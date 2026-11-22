"""
Step 8: RAGAS evaluation and the ablation table (the resume numbers)

What this file does:
    1. Answers the same 24 golden questions (factual + multi-section) with 3 versions of the system,
       adding one feature at a time (this is called an ABLATION):
            Version 1: vector search only
            Version 2: hybrid search (vector + BM25)
            Version 3: hybrid search + query rewriting fallback  (= the final system)
    2. Scores every answer with RAGAS, which uses an LLM as a judge. All 4 RAGAS metrics (0-1, higher is better):
            Generation (the LLM answer):
                Faithfulness      : is every claim in the answer supported by the retrieved chunks?
                Answer relevancy  : does the answer actually address the question?
            Retrieval (the search):
                Context recall    : did the search find all the information needed for the correct answer?
                Context precision : are the useful chunks ranked at the top?
    3. Measures latency (search + answer time per question)
    4. Prints the ablation table

The final system (V3) runs FIRST, so its report card is ready even if you stop early.
Saves progress after each version, so if the free API limit stops the run, just run it again:
finished versions are skipped.

Input : golden set, Qdrant index + chunks, GROQ_API_KEY in .env
Output: data/eval/ragas_v1.json, ragas_v2.json, ragas_v3.json   (answers + scores per version)
        data/eval/ablation_table.json
Run   : python src\\evaluation.py                  all 24 questions (slow: about 1-2 hours on the free API)
        python src\\evaluation.py --quick          first 10 questions only (about 20-40 minutes)
        python src\\evaluation.py --fresh          deletes saved progress and starts again
        (it always continues where it stopped)
"""

## importing required libraries
import json
import math
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from langchain_community.embeddings import FastEmbedEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from config import EVAL_DIR, GOLDEN_FILE, GROQ_API_KEY, ROUTER_MODEL
from embeddings import MODEL_NAME
from generation import RAGPipeline, TOP_K


## file paths
TABLE_FILE = EVAL_DIR / "ablation_table.json"

## settings
JUDGE_MODEL = ROUTER_MODEL      ## openai/gpt-oss-20b on Groq: judges the answers (a different model from the one answering)
QUICK_QUESTIONS = 10
VERSIONS = [                                     ## table order
    ("v1", "Vector search only"),
    ("v2", "+ Hybrid search (BM25)"),
    ("v3", "+ Query rewriting fallback"),
]
RUN_ORDER = ["v3", "v1", "v2"]                   ## final system first
METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
RAGAS_COLUMN = {                                 ## our short name -> the column name RAGAS uses
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_recall": "context_recall",
    "context_precision": "llm_context_precision_with_reference",
}


def get_judge():
    """The LLM judge for RAGAS: Groq, through its OpenAI-compatible address"""

    chat = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
        max_tokens=2048,
        reasoning_effort="low",
    )
    return LangchainLLMWrapper(chat)


def get_judge_embeddings():
    """Answer relevancy compares meanings, so RAGAS also needs an embedding model: the same local MiniLM (Step 3)"""

    return LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name=MODEL_NAME))


def answer_with_version(pipeline, version, question):
    """Search with one version of the system, then answer. Returns chunks, answer and seconds."""

    start = time.perf_counter()
    if version == "v1":
        ids = pipeline.retriever.vector(question, top_k=TOP_K)
    elif version == "v2":
        ids = pipeline.retriever.hybrid(question, top_k=TOP_K)
    else:
        ids, _, _, _ = pipeline.search_with_fallback(question)
    chunks = [pipeline.retriever.chunks[i] for i in ids]
    answer = pipeline.generate(question, chunks)
    return chunks, answer, round(time.perf_counter() - start, 2)


def mean(values):
    good = [v for v in values if v is not None and not math.isnan(v)]
    return (round(sum(good) / len(good), 3) if good else None), len(good)


def run_version(pipeline, judge, judge_embeddings, version, questions):
    """Answers all questions with one version and scores them with the 4 RAGAS metrics"""

    ## 1. answers
    records = []
    for n, q in enumerate(questions, start=1):
        chunks, answer, seconds = answer_with_version(pipeline, version, q["question"])
        records.append({"id": q["id"], "question": q["question"], "reference": q["expected_answer"],
                        "answer": answer, "contexts": [c["text"] for c in chunks], "seconds": seconds})
        print(f"   answered {n} of {len(questions)}  ({seconds} s)")

    ## 2. RAGAS scores
    print("   RAGAS is judging the answers (slow on the free API, please wait)...")
    dataset = EvaluationDataset.from_list([
        {"user_input": r["question"], "retrieved_contexts": r["contexts"],
         "response": r["answer"], "reference": r["reference"]} for r in records])
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=1),        ## 1 generated question per answer (Groq allows only 1 reply per call)
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=judge,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, timeout=240, max_retries=10, max_wait=60),
    )
    scores = result.to_pandas()

    summary = {"records": records, "avg_latency_s": round(sum(r["seconds"] for r in records) / len(records), 2)}
    scored_counts = []
    for metric in METRICS:
        column = RAGAS_COLUMN[metric] if RAGAS_COLUMN[metric] in scores.columns else None
        values = []
        for r, (_, row) in zip(records, scores.iterrows()):
            value = row[column] if column else float("nan")
            r[metric] = None if value is None or math.isnan(value) else float(value)
            values.append(r[metric])
        summary[metric], n_scored = mean(values)
        scored_counts.append(n_scored)
    summary["scored"] = f"{min(scored_counts)}/{len(records)}"
    return summary


## main function
def main():
    if "--fresh" in sys.argv[1:]:
        for key, _ in VERSIONS:
            (EVAL_DIR / f"ragas_{key}.json").unlink(missing_ok=True)
        print("Saved progress deleted, starting again.")

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    questions = [q for q in golden if q["type"] in ("factual", "multi_section")]
    if "--quick" in sys.argv[1:]:
        questions = questions[:QUICK_QUESTIONS]
        print(f"QUICK mode: {len(questions)} questions per version")

    pipeline = RAGPipeline()
    judge = get_judge()
    judge_embeddings = get_judge_embeddings()

    results = {}
    for key in RUN_ORDER:
        name = dict(VERSIONS)[key]
        saved_file = EVAL_DIR / f"ragas_{key}.json"
        if saved_file.exists():
            saved = json.loads(saved_file.read_text(encoding="utf-8"))
            if len(saved["records"]) == len(questions):
                print(f"\n{name}: already done, loading saved result")
                results[key] = saved
                continue
        print(f"\n{name}: running on {len(questions)} questions")
        result = run_version(pipeline, judge, judge_embeddings, key, questions)
        if any(result[m] is None for m in METRICS):
            print("   WARNING: the judge could not score this version (API limit?). Not saved: run again later.")
        else:
            saved_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        results[key] = result

    table = [{"version": name, **{m: results[key][m] for m in METRICS},
              "avg_latency_s": results[key]["avg_latency_s"], "scored": results[key]["scored"]}
             for key, name in VERSIONS if key in results]
    TABLE_FILE.write_text(json.dumps(table, indent=2), encoding="utf-8")

    ## ablation table
    print(f"\nRAGAS ABLATION TABLE ({len(questions)} golden questions, judge: {JUDGE_MODEL})")
    print(f"{'Version':<28}{'Faithful':>10}{'Ans.relev':>11}{'Ctx.recall':>12}{'Ctx.prec':>10}{'Latency s':>11}  scored")
    for row in table:
        print(f"{row['version']:<28}{str(row['faithfulness']):>10}{str(row['answer_relevancy']):>11}"
              f"{str(row['context_recall']):>12}{str(row['context_precision']):>10}{row['avg_latency_s']:>11}  {row['scored']}")

    print("\nHow to read it (0 to 1, higher is better):")
    print("  Faithful   = every claim in the answer is backed by the retrieved chunks (generation)")
    print("  Ans.relev  = the answer addresses the question (generation)")
    print("  Ctx.recall = the search found everything needed for the correct answer (retrieval)")
    print("  Ctx.prec   = the useful chunks are ranked at the top (retrieval)")
    print("  Latency s  = average seconds for search + answer (no cache)")
    print("  scored     = answers the judge could score in every metric (failed judge calls are left out)")
    print(f"\nSaved: {TABLE_FILE}")
    pipeline.retriever.client.close()


if __name__ == "__main__":
    main()