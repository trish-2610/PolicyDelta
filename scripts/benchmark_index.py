"""
Step 4a: FAISS benchmark (HNSW vs IVF)

Why: to find the closest chunks, the simplest way is EXACT search: compare the question with every chunk.
That is perfect but gets slow when there are millions of chunks. HNSW and IVF are two "shortcut" indexes
that are much faster but may miss a few results. This file measures that trade-off.

    Exact (Flat): compare with every chunk                        (like reading every book in the library)
    HNSW        : a graph that jumps from neighbour to neighbour   (like asking people "who is closer?" until you arrive)
    IVF         : groups chunks into clusters, searches only a few (like going only to the right shelves of the library)

Two tests:
    Test 1 (REAL data)      : our 136 real chunks and the 24 golden questions
    Test 2 (SIMULATED scale): 50,000 vectors made from our real ones plus small random noise,
                              because 136 chunks are too few to show any speed difference.
                              These vectors are SIMULATED test data, not real documents.

What is compared: each index's top 5 results vs the EXACT top 5 (recall@5 = how many of the true top 5 it found)

Input : data/processed/embeddings_B.npy, chunks_B.jsonl (Step 3 and Step 2), data/eval/golden_set.json
Output: data/eval/index_benchmark.json
Run   : python scripts\\benchmark_index.py
"""

## importing required libraries
import json
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np

## so we can reuse the embedding model from Step 3
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embeddings import EmbeddingModel   # noqa: E402


## file paths
PROCESSED_DIR = ROOT / "data" / "processed"
GOLDEN_FILE = ROOT / "data" / "eval" / "golden_set.json"
RESULTS_FILE = ROOT / "data" / "eval" / "index_benchmark.json"

## settings
TOP_K = 5
SCALE_SIZE = 50_000       ## how many vectors in the simulated scale test
SCALE_QUERIES = 500       ## how many test searches in the simulated scale test
NOISE = 0.05              ## how much random noise is added to make new vectors from real ones
HNSW_M = 32               ## HNSW: how many neighbours each point is linked to
IVF_LISTS_SCALE = 256     ## IVF: how many clusters in the scale test (about the square root of 50,000)
IVF_LISTS_REAL = 4        ## IVF: how many clusters for our 136 real chunks (few points = few clusters)
SEED = 42                 ## fixed random seed, so every run gives the same simulated data


def build_exact(vectors):
    """Exact search: compares the question with every vector (the 'perfect' answer)"""

    index = faiss.IndexFlatIP(vectors.shape[1])    ## IP = inner product = cosine, because vectors are normalized
    index.add(vectors)
    return index


def build_hnsw(vectors):
    """HNSW: a graph of neighbours"""

    index = faiss.IndexHNSWFlat(vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)
    return index


def build_ivf(vectors, n_lists):
    """IVF: first groups the vectors into clusters (training), then stores each vector in its cluster"""

    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], n_lists, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index


def search_all(index, queries):
    """Searches one question at a time (like real users do). Returns results and average milliseconds per search"""

    faiss.omp_set_num_threads(1)          ## one CPU thread, so the timing is fair
    results = []
    start_time = time.perf_counter()
    for q in queries:
        _, ids = index.search(q.reshape(1, -1), TOP_K)
        results.append(ids[0])
    ms_per_query = 1000 * (time.perf_counter() - start_time) / len(queries)
    faiss.omp_set_num_threads(0)          ## back to all threads
    return np.array(results), ms_per_query


def recall_vs_exact(results, exact_results):
    """How many of the exact top 5 did this index also find? (1.0 = all of them)"""

    found = [len(set(r) & set(e)) / TOP_K for r, e in zip(results, exact_results)]
    return float(np.mean(found))


def index_size_mb(index):
    """How much memory the index takes"""

    return round(faiss.serialize_index(index).nbytes / 1_000_000, 1)


def run_test(vectors, queries, n_lists):
    """Builds exact, HNSW and IVF indexes, searches with every setting and measures everything"""

    rows = []

    ## exact search = the reference
    t0 = time.perf_counter()
    exact = build_exact(vectors)
    build_exact_s = time.perf_counter() - t0
    exact_results, exact_ms = search_all(exact, queries)
    rows.append({"index": "Exact (Flat)", "setting": "-", "build_s": round(build_exact_s, 2),
                 "recall@5": 1.0, "ms_per_query": round(exact_ms, 3), "size_mb": index_size_mb(exact),
                 "results": exact_results})

    ## HNSW: efSearch = how many neighbours it looks at while searching (more = slower but more accurate)
    t0 = time.perf_counter()
    hnsw = build_hnsw(vectors)
    build_hnsw_s = time.perf_counter() - t0
    for ef in [16, 64]:
        hnsw.hnsw.efSearch = ef
        res, ms = search_all(hnsw, queries)
        rows.append({"index": "HNSW", "setting": f"efSearch={ef}", "build_s": round(build_hnsw_s, 2),
                     "recall@5": round(recall_vs_exact(res, exact_results), 3), "ms_per_query": round(ms, 3),
                     "size_mb": index_size_mb(hnsw), "results": res})

    ## IVF: nprobe = how many clusters it opens while searching (more = slower but more accurate)
    t0 = time.perf_counter()
    ivf = build_ivf(vectors, n_lists)
    build_ivf_s = time.perf_counter() - t0
    nprobe_settings = [1, 2] if n_lists < 32 else [1, n_lists // 32, n_lists // 8]
    for nprobe in nprobe_settings:
        ivf.nprobe = nprobe
        res, ms = search_all(ivf, queries)
        rows.append({"index": "IVF", "setting": f"nprobe={nprobe} of {n_lists}", "build_s": round(build_ivf_s, 2),
                     "recall@5": round(recall_vs_exact(res, exact_results), 3), "ms_per_query": round(ms, 3),
                     "size_mb": index_size_mb(ivf), "results": res})
    return rows


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def golden_hit5(results, questions, chunks):
    """Real test only: in how many golden questions is a chunk with the answer in the top 5?"""

    hits = 0
    for ids, q in zip(results, questions):
        top = [chunks[i] for i in ids if i >= 0]
        if any(c["doc_id"] == s["doc_id"] and normalize_text(s["evidence"]) in normalize_text(c["text"])
               for c in top for s in q["sources"]):
            hits += 1
    return f"{hits} of {len(questions)}"


def print_table(title, rows, with_golden=False):
    print(f"\n{title}")
    header = f"{'index':<14}{'setting':<18}{'build s':>9}{'recall@5':>10}{'ms/query':>10}{'size MB':>9}"
    if with_golden:
        header += f"{'golden hit@5':>14}"
    print(header)
    for r in rows:
        line = (f"{r['index']:<14}{r['setting']:<18}{r['build_s']:>9}{r['recall@5']:>10}"
                f"{r['ms_per_query']:>10}{r['size_mb']:>9}")
        if with_golden:
            line += f"{r['golden_hit@5']:>14}"
        print(line)


## main function
def main():
    ## real data from Steps 2 and 3
    vectors = np.load(PROCESSED_DIR / "embeddings_B.npy").astype("float32")
    chunks = [json.loads(line) for line in (PROCESSED_DIR / "chunks_B.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    questions = [q for q in golden if q["type"] in ("factual", "multi_section")]

    embedder = EmbeddingModel()
    question_vectors = embedder.embed([q["question"] for q in questions])

    ## ---------- Test 1: REAL data ----------
    real_rows = run_test(vectors, question_vectors, IVF_LISTS_REAL)
    for r in real_rows:
        r["golden_hit@5"] = golden_hit5(r["results"], questions, chunks)

    ## ---------- Test 2: SIMULATED scale ----------
    print(f"\nMaking {SCALE_SIZE:,} SIMULATED vectors from the {len(vectors)} real ones (real vector + small noise)...")
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(vectors), SCALE_SIZE)
    big = vectors[picks] + rng.normal(0, NOISE, (SCALE_SIZE, vectors.shape[1])).astype("float32")
    big /= np.linalg.norm(big, axis=1, keepdims=True)

    query_picks = rng.integers(0, len(vectors), SCALE_QUERIES)
    big_queries = vectors[query_picks] + rng.normal(0, NOISE, (SCALE_QUERIES, vectors.shape[1])).astype("float32")
    big_queries /= np.linalg.norm(big_queries, axis=1, keepdims=True)

    print("Building indexes (HNSW on 50,000 vectors can take up to a minute)...")
    scale_rows = run_test(big, big_queries, IVF_LISTS_SCALE)

    ## save (without the raw result lists)
    clean = lambda rows: [{k: v for k, v in r.items() if k != "results"} for r in rows]
    RESULTS_FILE.write_text(json.dumps({"real_136_chunks": clean(real_rows),
                                        "simulated_50k_vectors": clean(scale_rows)}, indent=2), encoding="utf-8")

    print_table(f"TEST 1: REAL DATA ({len(vectors)} chunks, {len(questions)} golden questions)", real_rows, with_golden=True)
    print_table(f"TEST 2: SIMULATED SCALE ({SCALE_SIZE:,} vectors, {SCALE_QUERIES} searches)", scale_rows)

    print("\nHow to read it:")
    print("  recall@5 = share of the EXACT top 5 that the index also found (1.0 = perfect)")
    print("  ms/query = average time for one search, on one CPU thread")
    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()