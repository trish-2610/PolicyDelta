"""
Step 9: FastAPI service (the project as a web API)

Endpoints:
    GET  /health   is the service running?
    POST /ask      ask a question -> answer + citations (runs the full pipeline from Step 7)
    POST /upload   add a new PDF -> cleaned, chunked, embedded and searchable (same pipeline as the first 5 documents)

Upload guard (security for files):
    - only .pdf files, at most 10 MB and 100 pages, and the file must really start with %PDF
    - must contain readable text (a scanned photo-PDF is rejected)
    - every page is scanned for hidden instructions ("ignore your instructions" ...): such a file is rejected
      (this is called INDIRECT prompt injection: the attack hides inside a document, not in the question)

Rate limit: each user (IP address) can ask at most 10 questions per minute.

Run (venv active, inside the project folder):
    uvicorn api.main:app
Then open http://127.0.0.1:8000/docs in the browser to try it.
"""

## importing required libraries
import json
import re
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

## so we can import the code in src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chunking      # noqa: E402
import indexing      # noqa: E402
import ingestion     # noqa: E402
import security      # noqa: E402
from generation import RAGPipeline   # noqa: E402

from api.schemas import AskRequest, AskResponse, UploadResponse   # noqa: E402


## settings
MAX_UPLOAD_MB = 10
MAX_PAGES = 100
MIN_TEXT_CHARS = 200            ## less text than this in the whole PDF = probably a scanned photo
RATE_LIMIT = 10                 ## questions per user per minute

SOURCES_FILE = ROOT / "data" / "sources.json"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

lock = threading.Lock()                     ## one upload / rebuild at a time
recent_requests = defaultdict(deque)        ## IP address -> times of its recent questions


@asynccontextmanager
async def lifespan(app):
    """Load the pipeline once when the server starts (models, index, BM25, cache)"""

    print("Loading the RAG pipeline...")
    app.state.pipeline = RAGPipeline()
    print("Ready: open http://127.0.0.1:8000/docs")
    yield
    app.state.pipeline.retriever.client.close()


app = FastAPI(title="PolicyDelta", description="Citation-grounded answers about India's DPDP Act, 2023 and Rules, 2025. "
                                               "Information, not legal advice.", lifespan=lifespan)


def check_rate_limit(ip):
    """Allows at most RATE_LIMIT questions per minute from one IP address"""

    now = time.time()
    times = recent_requests[ip]
    while times and now - times[0] > 60:
        times.popleft()
    if len(times) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail=f"Too many questions: at most {RATE_LIMIT} per minute.")
    times.append(now)


@app.get("/health")
def health():
    """Is the service running?"""

    return {"status": "ok", "chunks": len(app.state.pipeline.retriever.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    """Ask a question about the DPDP Act / Rules. Returns the answer with its sources."""

    check_rate_limit(request.client.host if request.client else "unknown")
    with lock:                                  ## wait if an upload is rebuilding the index
        result = app.state.pipeline.answer(body.question)
    return AskResponse(
        status=result["status"],
        answer=result["answer"],
        sources=[{k: s[k] for k in ("n", "title", "pages", "rule")} for s in result.get("sources", [])],
        disclaimer=result.get("disclaimer", ""),
        from_cache=result.get("from_cache", False),
        fallback_used=result.get("fallback_used", False),
        pii_masked=result.get("pii_masked", []),
        seconds=result["seconds"],
    )


def make_doc_id(filename):
    """'My Policy (v2).pdf' -> 'upload_my_policy_v2'"""

    stem = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_")[:50] or "document"
    return f"upload_{stem}"


def reject(message, start):
    return UploadResponse(status="rejected", message=message, seconds=round(time.perf_counter() - start, 2))


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...), title: str = Form("")):
    """Add a PDF. It goes through the same cleaning, chunking, embedding and indexing as the first 5 documents."""

    start = time.perf_counter()

    ## 1. upload guard: file type and size
    data = file.file.read()
    if not (file.filename or "").lower().endswith(".pdf"):
        return reject("Only .pdf files are accepted.", start)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return reject(f"The file is larger than {MAX_UPLOAD_MB} MB.", start)
    if not data.startswith(b"%PDF"):
        return reject("This file is not a real PDF (it does not start with %PDF).", start)

    doc_id = make_doc_id(file.filename)
    sources = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    if any(s["doc_id"] == doc_id for s in sources):
        return reject(f"A document named {doc_id} already exists.", start)

    with lock:
        pdf_path = RAW_DIR / f"{doc_id}.pdf"
        pdf_path.write_bytes(data)
        source = {"doc_id": doc_id, "title": title or Path(file.filename).stem, "doc_type": "upload",
                  "source_type": "uploaded", "published": time.strftime("%Y-%m-%d"), "url": "uploaded by user"}

        ## 2. cleaning (Step 1)
        try:
            document, stats = ingestion.process_pdf(source)
        except Exception as error:
            pdf_path.unlink(missing_ok=True)
            return reject(f"The PDF could not be read: {error}", start)

        ## 3. upload guard: page limit, readable text, hidden instructions
        problem = ""
        if stats["pdf_pages"] > MAX_PAGES:
            problem = f"The PDF has more than {MAX_PAGES} pages."
        elif stats["chars_after"] < MIN_TEXT_CHARS:
            problem = "No readable text found (it may be a scanned photo). Scanned PDFs are not supported."
        else:
            for page in document["pages"]:
                is_attack, matched = security.detect_injection(page["text"])
                if is_attack:
                    problem = f"Hidden instructions found on page {page['page']} ('{matched}'). File rejected."
                    break
        if problem:
            pdf_path.unlink(missing_ok=True)
            return reject(problem, start)

        ## 4. save and register the document
        ingestion.save_document(document)
        sources.append(source)
        SOURCES_FILE.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")

        ## 5. rebuild: chunking (Step 2) -> embeddings (Step 3) -> Qdrant index (Step 4)
        pipeline = app.state.pipeline
        documents = chunking.load_documents()
        chunks = chunking.make_chunks(documents, "B", chunking.way_b)
        chunking.save_chunks(chunks, "B")
        vectors = pipeline.retriever.embedder.embed([c["text"] for c in chunks])
        np.save(PROCESSED_DIR / "embeddings_B.npy", vectors)

        pipeline.retriever.client.close()               ## the local database allows one open connection
        indexing.build_index(chunks, vectors).close()

        ## 6. reload the pipeline (new chunks, new BM25, new index, and an EMPTY cache: old answers may now be incomplete)
        app.state.pipeline = RAGPipeline()

    return UploadResponse(status="added", message="The document is now searchable.", doc_id=doc_id,
                          pages_kept=stats["pages_kept"], chunks_total=len(chunks),
                          seconds=round(time.perf_counter() - start, 2))