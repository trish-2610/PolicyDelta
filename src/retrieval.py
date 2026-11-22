"""
Step 5: Hybrid search (keyword + meaning) and re-ranking

Why: vector search (Step 3/4) finds text with the same MEANING, but it can miss exact words like
"seventy-two hours", "section 29" or "Consent Manager". Keyword search (BM25) finds exact words but
misses meaning. Hybrid search uses BOTH. Then a re-ranker reads the question and each candidate
chunk together, and puts the best ones on top.

    1. Vector search : Qdrant, 384-number embeddings (meaning)                      -> top 20
    2. Keyword search: BM25 (counts matching words, rare words count more)          -> top 20
    3. Hybrid        : join both lists with RRF (Reciprocal Rank Fusion):
                       a chunk ranked high in either list gets a high combined score  -> top 20
    4. Re-rank       : a cross-encoder (ms-marco-MiniLM-L-6-v2, runs locally with FastEmbed)
                       reads question + chunk together and scores how well it answers -> final top 5

What is compared (same 24 golden questions as Steps 3 and 4):
    Version 1: vector only            (the baseline)
    Version 2: hybrid (vector + BM25)
    Version 3: hybrid + re-ranking

Input : Qdrant index (Step 4b), data/processed/chunks_B.jsonl, data/eval/golden_set.json
Decision (from the measured results): the app uses HYBRID search without re-ranking.
    Re-ranking did not improve accuracy on this data (right chunk first: 71% -> 46%)
    and made each question about 250x slower (8 ms -> 2,153 ms). It is kept here only as a measured experiment.

Output: data/eval/retrieval_results.json
Run   : python src\\retrieval.py
"""

## importing required libraries
import json
import re
import time
from pathlib import Path

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from embeddings import EmbeddingModel
from indexing import get_client, search as vector_search


## file paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
GOLDEN_FILE = ROOT / "data" / "eval" / "golden_set.json"
RESULTS_FILE = ROOT / "data" / "eval" / "retrieval_results.json"

## settings
CANDIDATES = 20       ## how many chunks each search returns before joining / re-ranking
TOP_K = 5             ## how many chunks we finally keep
RRF_K = 60            ## standard RRF constant: stops the very first rank from dominating
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = False  ## switched off for the app after measuring (see the note at the top)

## very common words that carry no meaning for keyword search
STOP_WORDS = set("""a an the of to in on for and or is are be by with as at from that this which what who whom
how when where why under shall may must can it its any such do does did has have had i my me we our
you your their there than then into about after before within upon""".split())


def tokenize(text):
    """This function splits text into lower-case words and removes common words"""

    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOP_WORDS]


class Retriever:
    """This class holds the three searches: vector, keyword (BM25) and the re-ranker"""

    ## loads everything once (the re-ranker only when asked, it is slow and not used by the app)
    def __init__(self, load_reranker=USE_RERANKER):
        self.chunks = [json.loads(line) for line in
                       (PROCESSED_DIR / "chunks_B.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.embedder = EmbeddingModel()
        self.client = get_client()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.reranker = None
        if load_reranker:
            print(f"Loading re-ranker {RERANK_MODEL} (first time: downloads about 80 MB)...")
            self.reranker = TextCrossEncoder(model_name=RERANK_MODEL)

    ## 1. vector search: returns chunk positions, best first
    def vector(self, question, top_k=CANDIDATES):
        q_vector = self.embedder.embed([question])[0]
        return [p.id for p in vector_search(self.client, q_vector, top_k=top_k)]

    ## 2. keyword search: returns chunk positions, best first
    def keyword(self, question, top_k=CANDIDATES):
        scores = self.bm25.get_scores(tokenize(question))
        return list(np.argsort(-scores)[:top_k])

    ## 3. hybrid: join both lists with Reciprocal Rank Fusion
    def hybrid(self, question, top_k=CANDIDATES):
        combined = {}
        for ranked_list in [self.vector(question), self.keyword(question)]:
            for rank, chunk_id in enumerate(ranked_list, start=1):
                combined[chunk_id] = combined.get(chunk_id, 0) + 1 / (RRF_K + rank)
        return sorted(combined, key=combined.get, reverse=True)[:top_k]

    ## 4. re-rank: the cross-encoder reads question + each chunk and scores it
    def rerank(self, question, chunk_ids, top_k=TOP_K):
        texts = [self.chunks[i]["text"] for i in chunk_ids]
        scores = list(self.reranker.rerank(question, texts))
        order = np.argsort(-np.array(scores))
        return [chunk_ids[i] for i in order[:top_k]]

    ## the search used by the app later: hybrid (re-ranking only if switched on), returns the chunks themselves
    def retrieve(self, question, top_k=TOP_K):
        if USE_RERANKER:
            ids = self.rerank(question, self.hybrid(question), top_k=top_k)
        else:
            ids = self.hybrid(question, top_k=top_k)
        return [self.chunks[i] for i in ids]


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def has_answer(chunk, src):
    return chunk["doc_id"] == src["doc_id"] and normalize(src["evidence"]) in normalize(chunk["text"])


def evaluate(name, search_function, questions, chunks):
    """This function runs one version on all questions and measures it"""

    hit_1 = hit_5 = phrases_found = total_phrases = 0
    reciprocal_ranks = []
    times = []
    misses = []

    for q in questions:
        t0 = time.perf_counter()
        top = [chunks[i] for i in search_function(q["question"])][:TOP_K]
        times.append(1000 * (time.perf_counter() - t0))

        ## position of the first chunk that has the answer (1 = top)
        first_hit = next((rank for rank, c in enumerate(top, start=1)
                          if any(has_answer(c, s) for s in q["sources"])), None)
        if first_hit == 1:
            hit_1 += 1
        if first_hit:
            hit_5 += 1
        else:
            misses.append(q["id"])
        reciprocal_ranks.append(1 / first_hit if first_hit else 0)

        for s in q["sources"]:
            total_phrases += 1
            if any(has_answer(c, s) for c in top):
                phrases_found += 1

    n = len(questions)
    return {
        "version": name,
        "hit@1": f"{hit_1} of {n} ({round(100 * hit_1 / n)}%)",
        "hit@5": f"{hit_5} of {n} ({round(100 * hit_5 / n)}%)",
        "answer_phrases_top5": f"{phrases_found} of {total_phrases} ({round(100 * phrases_found / total_phrases)}%)",
        "MRR": round(sum(reciprocal_ranks) / n, 3),
        "avg_ms": round(sum(times) / n, 1),
        "missed_questions": misses,
    }


## main function
def main():
    retriever = Retriever(load_reranker=True)   ## the experiment compares all versions, so load it here
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    questions = [q for q in golden if q["type"] in ("factual", "multi_section")]

    ## warm up (first call loads things into memory; we do not want to time that)
    retriever.rerank(questions[0]["question"], retriever.hybrid(questions[0]["question"]))

    versions = [
        ("1. Vector only (baseline)", lambda q: retriever.vector(q)),
        ("2. Hybrid (vector + BM25)", lambda q: retriever.hybrid(q)),
        ("3. Hybrid + re-ranking", lambda q: retriever.rerank(q, retriever.hybrid(q))),
    ]
    results = [evaluate(name, fn, questions, retriever.chunks) for name, fn in versions]

    ## extra: keyword search alone, to show why we need both
    keyword_only = evaluate("(extra) Keyword only (BM25)", lambda q: retriever.keyword(q), questions, retriever.chunks)

    RESULTS_FILE.write_text(json.dumps(results + [keyword_only], indent=2), encoding="utf-8")

    ## results table
    print(f"\nRETRIEVAL RESULTS ({len(questions)} golden questions, top {TOP_K} chunks)")
    print(f"{'version':<30}{'hit@1':>14}{'hit@5':>14}{'phrases top5':>16}{'MRR':>7}{'ms':>8}")
    for r in results + [keyword_only]:
        print(f"{r['version']:<30}{r['hit@1']:>14}{r['hit@5']:>14}{r['answer_phrases_top5']:>16}{r['MRR']:>7}{r['avg_ms']:>8}")

    print("\nQuestions still missed by version 3:", ", ".join(results[2]["missed_questions"]) or "none")
    print("\nHow to read it:")
    print("  hit@1 = right chunk is result no. 1   hit@5 = right chunk somewhere in the top 5")
    print("  MRR   = average of 1/position of the first right chunk (1.0 = always first)")
    print("  ms    = average time per question on your laptop")
    print(f"\nSaved: {RESULTS_FILE}")
    retriever.client.close()


if __name__ == "__main__":
    main()
