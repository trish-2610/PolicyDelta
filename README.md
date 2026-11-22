# PolicyDelta

A question-answering system for India's data protection law (the DPDP Act, 2023 and the DPDP Rules, 2025). It answers only from the official government documents and shows the exact document, page and rule behind every answer.

> PolicyDelta gives information from the official documents. It is not legal advice.

---

## The problem

India's DPDP law is spread across several long government PDFs. Each gazette file has Hindi pages first and English after, with the same header repeated on every page. Finding one rule, such as how fast a data breach must be reported, means reading through many pages.

A general chatbot is not a safe shortcut: it can answer from memory, give no source, and users may type personal details such as an Aadhaar number into the question.

## What PolicyDelta does

- Answers questions **only from the official documents**, with citations such as *DPDP Rules 2025, page 26, rule 7*.
- Says **"The documents do not contain this information"** instead of guessing.
- **Hides personal data** (Aadhaar, PAN, phone, email) before anything reaches the AI.
- **Rejects off-topic and trick questions** before any search happens.
- Lets users **upload new PDFs**, which go through the same pipeline.

**Example**

```
Q: Within how many hours must a company report a data breach to the Board?
A: Within seventy-two hours of becoming aware of the breach [1].
   [1] Digital Personal Data Protection Rules, 2025, pages 26-26, rule 7
```

---

## Architecture

![PolicyDelta architecture](docs/architecture.png)

## How a question is answered

```
User question
   |
   v
Safety checks:  hide personal data -> block trick questions -> is it about the DPDP law?
   |                                                            (no -> polite refusal)
   v
Saved answer for this question?  (yes -> return it instantly)
   |
   v
Search the documents two ways: by meaning + by exact words  ->  best 5 pieces
   |
   v
Weak match?  (yes -> rewrite the question in legal terms and search again)
   |
   v
AI answers ONLY from those 5 pieces, with a citation after each fact
   |
   v
Code checks every citation points to a piece that was really given  ->  answer + sources
```

**How the documents were prepared:** Hindi pages and repeated headers were removed, the original page numbers were kept for citations, and the text was cut into one piece per legal rule. A count check confirmed nothing was lost: all 44 sections of the Act and all 23 rules of the Rules.

---

## Results

All numbers come from real runs on a 50-question test set that I wrote and checked against the document pages.

**Answer quality** (RAGAS, 24 questions, compared with plain meaning-only search)

| Metric | What it checks | Plain search | PolicyDelta |
|---|---|---|---|
| Context recall | Did the search find all the information needed? | 0.92 | **1.00** |
| Context precision | Were the useful pieces ranked at the top? | 0.70 | **0.77** |
| Faithfulness | Is the answer supported by the found text? | 0.92 | **0.92** |

**Safety**

| Check | Result |
|---|---|
| Off-topic and trick questions stopped | 16 of 16 |
| Genuine questions wrongly blocked | 0 of 34 |
| Questions with personal data masked correctly | 6 of 6 |
| Personal data sent to the AI | 0 |

**Search speed test** (FAISS, 50,000 simulated vectors)

| Method | Accuracy vs exact search | Time per search |
|---|---|---|
| Exact search | 100% | 9.4 ms |
| HNSW (chosen) | 99.4% | 0.17 ms (~54x faster) |
| IVF | 100% | 0.28 ms |

HNSW was chosen because it was the fastest, needs no training step (new documents can be added any time), and matched exact search on the real 136 pieces, where IVF missed some answers.

---

## Key design decisions

| Decision | Why |
|---|---|
| Cut documents **one piece per legal rule** | Each piece is a complete rule with its heading. Pieces starting mid-sentence fell from 85% to 18%, and citations can name the rule. |
| **Search by meaning and by exact words** (hybrid) | Meaning-based search handles paraphrases; keyword search (BM25) catches exact terms like "section 29". It costs only about 2 ms extra. |
| **Rewrite the question only when search is weak** | Rewriting every question did not help in tests and added about 2.5 s, so it runs only as a fallback. |
| **Removed the re-ranker** | It was trained on short web text, not legal text, and was ~250x slower on a CPU. It lowered first-place accuracy from 71% to 46%. |
| **Small local embedding model (ONNX)** | Same open-source model (all-MiniLM-L6-v2) without PyTorch: free, offline, fast on an 8 GB laptop. |
| **Qdrant** for the app, **FAISS** for the benchmark | FAISS stores only vectors. Qdrant also stores document, page and rule with each piece, which citations need. |
| **Rules first, AI second** for safety | Fixed rules are free and instant; a small AI model catches the cleverer cases. |

---

## Tech stack

Python, pypdf, FastEmbed (all-MiniLM-L6-v2, ONNX), Qdrant, FAISS, rank-bm25, Groq LLMs (gpt-oss-20b for routing and rewriting, gpt-oss-120b for answers), RAGAS, FastAPI.

---

## Project structure

```
PolicyDelta/
├── docs/architecture.png   architecture diagram
├── data/
│   ├── raw/                  the 5 official PDFs
│   ├── processed/            cleaned text, chunks, embeddings
│   ├── eval/                 test questions and results
│   └── sources.json          details of each document
├── scripts/
│   ├── build_golden_set.py   50 test questions, each checked against its page
│   └── benchmark_index.py    FAISS: exact vs HNSW vs IVF
├── src/
│   ├── ingestion.py          clean the PDFs
│   ├── chunking.py           cut by rule (vs fixed-size)
│   ├── embeddings.py         text -> vectors
│   ├── indexing.py           store in Qdrant
│   ├── retrieval.py          hybrid search
│   ├── security.py           personal data masking, trick-question rules
│   ├── query_processing.py   topic check and query rewriting
│   ├── generation.py         full answer pipeline with citations
│   ├── cache.py              saved answers (in-memory, Redis-ready)
│   └── evaluation.py         RAGAS evaluation
├── api/                      FastAPI service
└── tests/                    API tests
```

---

## Run it

Requires Python 3.12 and a free Groq API key.

```bash
python -m venv venv
venv\Scripts\activate                 # Windows
pip install -r requirements.txt
# put GROQ_API_KEY=... in a .env file

python src\ingestion.py               # clean the documents
python src\chunking.py                # cut into pieces
python src\embeddings.py              # create embeddings
python src\indexing.py                # build the Qdrant index

uvicorn api.main:app                  # then open http://127.0.0.1:8000/docs
```

**API**

| Endpoint | What it does |
|---|---|
| `POST /ask` | Question in, answer with sources out |
| `POST /upload` | Add a PDF (only real PDFs up to 10 MB / 100 pages; scanned files and files with hidden instructions are rejected) |
| `GET /health` | Service status |

Repeated questions are answered from a cache, and each user can ask up to 10 questions per minute.

---

## Limitations

- Small corpus (5 documents) and test set, so results may not hold for every document.
- RAGAS scores are marked by an AI judge and are estimates. Answer relevancy could not be completed on the free API tier.
- Personal names are not masked, and scanned PDFs are not supported (no OCR).
- The speed test uses simulated vectors, because 136 real pieces are too few to show a difference.

## Next steps

Docker and CI/CD, cloud deployment with a Qdrant server and Redis, a legal-domain re-ranker, and Hindi language support.
