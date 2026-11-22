"""
Step 3: Embeddings (turn text into numbers that capture meaning)

What this file does:
    1. Turns every chunk into 384 numbers (an embedding) on YOUR laptop
       Model: sentence-transformers/all-MiniLM-L6-v2 (small, fast, open-source)
       Engine: FastEmbed (ONNX), a light engine that runs on a normal CPU, no PyTorch, no API, no cost
       The model (about 90 MB) is downloaded once, the first time, then reused
    2. Saves the embeddings to disk, so later steps reuse them and never redo this work
    3. Runs a first, simple search test (the BASELINE): for each golden question, find the 5 closest chunks
       and check if the right answer is among them. Done for both chunking ways (A and B).

Input : data/processed/chunks_A.jsonl, chunks_B.jsonl   (from Step 2)
        data/eval/golden_set.json                        (the test questions)
Output: data/processed/embeddings_A.npy, embeddings_B.npy
        data/eval/embedding_results.json
Run   : python src\\embeddings.py
"""

## importing required libraries
import json
import re
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


## folder paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
GOLDEN_FILE = ROOT / "data" / "eval" / "golden_set.json"
RESULTS_FILE = ROOT / "data" / "eval" / "embedding_results.json"

## settings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32         ## how many texts are embedded together
TOP_K = 5               ## how many chunks the search returns
MODEL_LIMIT_CHARS = 1000  ## the model reads about 256 tokens (roughly 1000 characters); text after that is ignored


class EmbeddingModel:
    """This class loads the embedding model and creates the embeddings"""

    ## initializes the embedding model (downloads it the first time only)
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading model {model_name} (first time: downloads about 90 MB)...")
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    ## This function creates the embeddings for a list of texts
    def embed(self, texts):
        vectors = np.array(list(self.model.embed(texts, batch_size=BATCH_SIZE)), dtype="float32")

        ## normalize: every vector gets length 1, so a dot product = cosine similarity
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def load_chunks(way_name):
    """This function loads the chunks from Step 2"""

    chunk_file = PROCESSED_DIR / f"chunks_{way_name}.jsonl"
    return [json.loads(line) for line in chunk_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text):
    """lower case and single spaces, so line breaks do not break the match"""

    return re.sub(r"\s+", " ", text).strip().lower()


def search(question_vector, chunk_vectors, top_k=TOP_K):
    """This function returns the positions of the top_k chunks most similar to the question"""

    scores = chunk_vectors @ question_vector        ## cosine similarity with every chunk
    return np.argsort(-scores)[:top_k]


def chunk_has_answer(chunk, src):
    """A chunk 'has the answer' if it is from the right document and contains the answer phrase"""

    return chunk["doc_id"] == src["doc_id"] and normalize(src["evidence"]) in normalize(chunk["text"])


def evaluate(questions, question_vectors, chunks, chunk_vectors):
    """This function measures how often the right chunk is in the top 1 and the top 5"""

    hit_1 = 0           ## questions where the FIRST result has the answer
    hit_5 = 0           ## questions where at least one of the top 5 has the answer
    phrases_found = 0   ## answer phrases (some questions have 2) found in the top 5
    total_phrases = 0

    for question, q_vector in zip(questions, question_vectors):
        top = [chunks[i] for i in search(q_vector, chunk_vectors)]

        if any(chunk_has_answer(top[0], src) for src in question["sources"]):
            hit_1 += 1
        if any(chunk_has_answer(c, src) for c in top for src in question["sources"]):
            hit_5 += 1
        for src in question["sources"]:
            total_phrases += 1
            if any(chunk_has_answer(c, src) for c in top):
                phrases_found += 1

    n = len(questions)
    return {
        "questions": n,
        "hit@1": f"{hit_1} of {n} ({round(100 * hit_1 / n)}%)",
        "hit@5": f"{hit_5} of {n} ({round(100 * hit_5 / n)}%)",
        "answer_phrases_in_top5": f"{phrases_found} of {total_phrases} ({round(100 * phrases_found / total_phrases)}%)",
    }


## main function
def main():
    embedder = EmbeddingModel()

    ## test questions: only the normal ones that have an answer page (factual + multi_section).
    ## PII questions are tested later, together with the PII guard (Step 6).
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    questions = [q for q in golden if q["type"] in ("factual", "multi_section")]
    print(f"Embedding {len(questions)} test questions...")
    question_vectors = embedder.embed([q["question"] for q in questions])

    results = {"model": MODEL_NAME, "dimensions": int(question_vectors.shape[1])}

    for way_name in ["A", "B"]:
        chunks = load_chunks(way_name)
        print(f"Embedding {len(chunks)} chunks of way {way_name}...")

        start_time = time.perf_counter()
        chunk_vectors = embedder.embed([c["text"] for c in chunks])
        seconds = round(time.perf_counter() - start_time, 1)

        ## save the embeddings (row i = chunk i in chunks_<way>.jsonl)
        np.save(PROCESSED_DIR / f"embeddings_{way_name}.npy", chunk_vectors)

        scores = evaluate(questions, question_vectors, chunks, chunk_vectors)
        scores["chunks"] = len(chunks)
        scores["embedding_seconds"] = seconds
        scores["chunks_longer_than_model_reads"] = sum(1 for c in chunks if c["n_chars"] > MODEL_LIMIT_CHARS)
        results[way_name] = scores

    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ## results table
    rows = [
        ("Chunks embedded", "chunks"),
        ("Time to embed (seconds)", "embedding_seconds"),
        ("Chunks longer than model reads (~1000 ch)", "chunks_longer_than_model_reads"),
        ("Right chunk is result no. 1 (hit@1)", "hit@1"),
        ("Right chunk in top 5 (hit@5)", "hit@5"),
        ("Answer phrases found in top 5", "answer_phrases_in_top5"),
    ]
    print(f"\nEMBEDDING RESULTS  (model: {MODEL_NAME}, {results['dimensions']} numbers per chunk)")
    print(f"Test: {len(questions)} golden questions (factual + multi-section), vector search only = BASELINE\n")
    print(f"{'':<44}{'Way A (fixed-size)':<22}Way B (by rule)")
    for label, key in rows:
        print(f"{label:<44}{str(results['A'][key]):<22}{results['B'][key]}")

    print("\nSaved: data/processed/embeddings_A.npy, data/processed/embeddings_B.npy")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
