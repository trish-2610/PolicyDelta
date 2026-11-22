"""
Step 4b: Indexing in Qdrant (the searchable database the app will use)

What this file does:
    1. Creates a Qdrant collection (like a table) in LOCAL mode: Qdrant runs inside Python and saves to
       data/qdrant. No server, no Docker.
    2. Stores every chunk as a "point" = its 384 numbers (from Step 3) + its info (document, title, pages,
       rule heading, primary/secondary, text). The info is called the "payload" and is used for citations.
    3. Remembers which embedding model built the index. If the model changes, the old numbers no longer
       match new questions, so the whole index is rebuilt automatically.
    4. Tests the database: searches the 24 golden questions through Qdrant and checks the results match
       Step 3, measures search time, and tests a filter (search only primary legal documents).

Input : data/processed/chunks_B.jsonl, embeddings_B.npy   (Steps 2 and 3)
        data/eval/golden_set.json
Output: data/qdrant/                       the database
        data/qdrant_index_info.json        which model and how many chunks built the index
        data/eval/indexing_results.json
Run   : python src\\indexing.py              build (skips if already up to date)
        python src\\indexing.py --rebuild    force a full rebuild
"""

## importing required libraries
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from embeddings import MODEL_NAME, EmbeddingModel


## file paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
QDRANT_DIR = ROOT / "data" / "qdrant"
INDEX_INFO_FILE = ROOT / "data" / "qdrant_index_info.json"
GOLDEN_FILE = ROOT / "data" / "eval" / "golden_set.json"
RESULTS_FILE = ROOT / "data" / "eval" / "indexing_results.json"

## settings
COLLECTION = "dpdp_chunks"
TOP_K = 5


def load_chunks_and_vectors():
    """This function loads the chunks (Step 2) and their embeddings (Step 3)"""

    chunks = [json.loads(line) for line in (PROCESSED_DIR / "chunks_B.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors = np.load(PROCESSED_DIR / "embeddings_B.npy").astype("float32")
    if len(chunks) != len(vectors):
        raise SystemExit(f"{len(chunks)} chunks but {len(vectors)} embeddings. Run Step 3 again: python src\\embeddings.py")
    return chunks, vectors


def get_client():
    """This function opens the local Qdrant database (used by this step and by later steps)"""

    return QdrantClient(path=str(QDRANT_DIR))


def index_is_up_to_date(n_chunks):
    """The index is up to date if it was built with the SAME model and the SAME number of chunks"""

    if not INDEX_INFO_FILE.exists() or not QDRANT_DIR.exists():
        return False, "no index yet"
    info = json.loads(INDEX_INFO_FILE.read_text(encoding="utf-8"))
    if info["model"] != MODEL_NAME:
        return False, f"embedding model changed ({info['model']} -> {MODEL_NAME})"
    if info["chunks"] != n_chunks:
        return False, f"number of chunks changed ({info['chunks']} -> {n_chunks})"
    return True, "same model and same chunks"


def build_index(chunks, vectors):
    """This function creates the collection and stores every chunk with its numbers and its info"""

    ## start clean: delete the old database folder
    if QDRANT_DIR.exists():
        shutil.rmtree(QDRANT_DIR)

    client = get_client()
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {key: chunk[key] for key in
                   ["chunk_id", "doc_id", "title", "source_type", "page_start", "page_end", "heading", "text"]}
        points.append(PointStruct(id=i, vector=vector.tolist(), payload=payload))
    client.upsert(collection_name=COLLECTION, points=points)

    ## remember which model and how many chunks built this index
    INDEX_INFO_FILE.write_text(json.dumps({"model": MODEL_NAME, "dimensions": int(vectors.shape[1]),
                                           "chunks": len(chunks), "collection": COLLECTION}, indent=2), encoding="utf-8")
    return client


def search(client, question_vector, top_k=TOP_K, only_primary=False):
    """This function searches Qdrant. only_primary=True searches only the legal texts (not the PIB explainer)"""

    query_filter = None
    if only_primary:
        query_filter = Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="primary"))])
    response = client.query_points(collection_name=COLLECTION, query=question_vector.tolist(),
                                   query_filter=query_filter, limit=top_k, with_payload=True)
    return response.points


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def has_answer(payload, src):
    return payload["doc_id"] == src["doc_id"] and normalize(src["evidence"]) in normalize(payload["text"])


## main function
def main():
    chunks, vectors = load_chunks_and_vectors()

    ## build only if needed
    up_to_date, reason = index_is_up_to_date(len(chunks))
    if "--rebuild" in sys.argv[1:]:
        up_to_date, reason = False, "--rebuild was asked"

    if up_to_date:
        print(f"Index is up to date ({reason}): skipping the build")
        client = get_client()
        build_seconds = 0.0
        action = "skipped"
    else:
        print(f"Building the index because: {reason}")
        t0 = time.perf_counter()
        client = build_index(chunks, vectors)
        build_seconds = round(time.perf_counter() - t0, 2)
        action = "built"

    stored = client.count(collection_name=COLLECTION).count

    ## test: search the golden questions through Qdrant
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    questions = [q for q in golden if q["type"] in ("factual", "multi_section")]
    embedder = EmbeddingModel()
    question_vectors = embedder.embed([q["question"] for q in questions])

    hit_5 = 0
    same_as_numpy = 0
    hit_5_primary = 0
    secondary_returned = 0
    times = []
    for q, q_vector in zip(questions, question_vectors):
        t0 = time.perf_counter()
        points = search(client, q_vector)
        times.append(1000 * (time.perf_counter() - t0))

        if any(has_answer(p.payload, s) for p in points for s in q["sources"]):
            hit_5 += 1

        ## do Qdrant's top 5 match a plain numpy exact search (Step 3 method)?
        numpy_top = set(np.argsort(-(vectors @ q_vector))[:TOP_K].tolist())
        if {p.id for p in points} == numpy_top:
            same_as_numpy += 1

        ## filter test: only primary (legal) documents
        primary_points = search(client, q_vector, only_primary=True)
        secondary_returned += sum(1 for p in primary_points if p.payload["source_type"] != "primary")
        if any(has_answer(p.payload, s) for p in primary_points for s in q["sources"]):
            hit_5_primary += 1

    n = len(questions)
    example = search(client, question_vectors[0], top_k=1)[0].payload
    results = {
        "action": action,
        "reason": reason,
        "build_seconds": build_seconds,
        "chunks_stored": stored,
        "golden_hit@5": f"{hit_5} of {n}",
        "same_top5_as_exact_numpy": f"{same_as_numpy} of {n}",
        "avg_search_ms": round(sum(times) / n, 2),
        "golden_hit@5_primary_only": f"{hit_5_primary} of {n}",
        "secondary_chunks_returned_by_primary_filter": secondary_returned,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ## results table
    print("\nINDEXING RESULTS (Qdrant, local mode)")
    print(f"{'Index':<46}{action} ({reason})")
    print(f"{'Build time (seconds)':<46}{build_seconds}")
    print(f"{'Chunks stored in Qdrant':<46}{stored} of {len(chunks)}")
    print(f"{'Golden questions: right chunk in top 5':<46}{hit_5} of {n}   (Step 3 gave 21 of 24)")
    print(f"{'Same top 5 as exact search (Step 3 method)':<46}{same_as_numpy} of {n}")
    print(f"{'Average search time (ms)':<46}{results['avg_search_ms']}")
    print(f"{'Filter test: only primary documents':<46}{hit_5_primary} of {n} found, "
          f"{secondary_returned} secondary chunks leaked (should be 0)")

    print("\nExample of what is stored with every chunk (used for citations):")
    print(f"  question : {questions[0]['question'][:80]}")
    print(f"  document : {example['title']}")
    print(f"  pages    : {example['page_start']}-{example['page_end']}")
    print(f"  rule     : {example['heading']}")
    print(f"  type     : {example['source_type']}")

    print(f"\nSaved: database in {QDRANT_DIR}")
    print(f"Saved: {RESULTS_FILE}")
    client.close()


if __name__ == "__main__":
    main()