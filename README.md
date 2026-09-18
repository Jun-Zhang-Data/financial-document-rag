Financial Document RAG

A hands-on RAG project for question answering over a small synthetic set of financial reports.

I built this project to practice and understand the full RAG flow: preparing document chunks, retrieving relevant evidence, passing retrieved context to an LLM, validating citations, evaluating retrieval quality, and exposing the pipeline through an API.

This is a portfolio and learning project. It is not a production-proven financial system. The corpus is synthetic, the evaluation set is small, and the repository has not been tested with real production traffic.

What this project focuses on

The parts I focused on most are:

document chunking and metadata such as company, year, and page;

lexical, dense, and hybrid retrieval;

metadata filtering before retrieval;

page-level deduplication so duplicate chunks from the same page do not occupy several top-k positions;

retrieval evaluation with Recall@k, Coverage@k, and MRR;

giving only retrieved context to the LLM for answer generation;

refusing to answer when there is not enough evidence;

checking that citations returned by the LLM came from the retrieved context;

serving the pipeline through FastAPI;

an optional PostgreSQL/pgvector-backed retrieval path.

Dataset and evaluation setup

The repository uses five short synthetic reports for three invented companies. The corpus contains 39 pages and 59 chunks.

I use synthetic data so that I can define page-level ground truth for evaluation and inspect retrieval failures directly. This also means the project does not demonstrate real PDF parsing or retrieval over real company filings.

The evaluation set contains 18 questions: 15 answerable questions and 3 intentionally unanswerable questions. The default evaluation uses top_k=4.

Retrieval results

retriever                           Recall@4    Coverage@4     MRR   latency (med/best3)
----------------------------------------------------------------------------------------
bm25                                    0.80          0.80    0.60       0.1ms
bm25 + page dedupe                      0.87          0.87    0.63       0.1ms
dense                                   0.87          0.67    0.50      22.8ms
dense + page dedupe                     0.87          0.73    0.50      20.7ms
hybrid                                  0.73          0.73    0.52      24.7ms
hybrid + page dedupe                    0.87          0.80    0.56      21.4ms

These numbers are specific to this small synthetic corpus. They should not be interpreted as evidence that BM25 is generally better than dense or hybrid retrieval.

On this dataset, BM25 with page deduplication performed best overall, so I use it as the default local retriever. One useful lesson from the experiment was that a more complex retrieval method is not automatically better; I prefer to compare alternatives with the same evaluation set.

Run the retrieval evaluation with:

PYTHONPATH=src python src/evaluate.py --retrievers bm25
PYTHONPATH=src python src/evaluate.py

Recall@k and Coverage@k

I use both metrics because they answer different questions.

Recall@4 asks whether at least one expected evidence page appears in the top four retrieved results.

Coverage@4 is stricter for questions that require more than one source. It checks whether all expected evidence pages are present in the top four.

For example, a question comparing two companies may require one page from each company. Retrieving only one of those pages can still give a partial retrieval hit, but it is not enough to fully support the comparison. Coverage helps expose that failure.

Why page deduplication matters

Some pages are split into overlapping chunks. Without deduplication, two chunks from the same page can both appear near the top of the ranking and use two of the available top-k positions.

The retrieval code can therefore keep only the highest-ranked chunk from each page before returning the final results.

On this evaluation set, enabling page deduplication improved the BM25 Recall@4 and Coverage@4 from 0.80 to 0.87.

End-to-end request flow

A query follows this general path:

Client
  |
  v
FastAPI /v1/query
  |
  v
RAGService
  |
  v
metadata filtering
(company / year when present)
  |
  v
retrieval
  |
  v
top-k evidence chunks
  |
  v
LLM generation using retrieved context
  |
  v
citation validation
  |
  v
JSON response

api.py handles the HTTP request and response. service.py coordinates the retrieval and generation steps. The retrieval modules find evidence, and generation.py formats the retrieved evidence for the LLM and validates citations in the returned answer.

Retrieval approaches

The in-memory evaluation path supports three retrieval approaches.

Lexical retrieval

The BM25 retriever ranks chunks using the words that appear in the query and document text. This works well when the question uses terminology similar to the reports.

Dense retrieval

Dense retrieval converts the query and chunks into embeddings and ranks them by semantic similarity. This can help when the question and the document express the same idea using different words.

Hybrid retrieval

The hybrid retriever combines lexical and dense rankings using Reciprocal Rank Fusion (RRF). RRF combines the relative ranks from the two retrievers instead of trying to compare their raw scores directly.

The repository also contains a PostgreSQL-backed path that combines PostgreSQL full-text search with pgvector dense retrieval. I treat this as a database-backed version of the retrieval service rather than as evidence that the system has been operated at production scale.

Generation and grounding

After retrieval, the selected chunks are formatted as numbered sources and included in the LLM input.

The generation instructions tell the model to:

answer only from the supplied context;

return INSUFFICIENT_EVIDENCE when the context is not enough;

cite factual claims using the supplied document/page citations;

avoid inventing citations.

If retrieval returns no evidence at all, the code does not call the LLM and returns an insufficient-evidence result instead.

Citation validation: what it does and does not prove

After the LLM returns an answer, the code extracts citations such as:

[Northstar Industrial 2024, p.18]

It then checks whether each cited page was actually included in the retrieved context provided to the model.

This protects against one specific failure mode: the model inventing a citation to a page it never received.

However, this check does not prove that the text on that page semantically supports every claim in the answer. For example, a model could cite the correct page but misstate a number from that page. Claim-level semantic groundedness evaluation is not implemented in this version of the project.

API

The FastAPI service exposes a query endpoint. A retrieval-only request does not require LLM credentials.

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

PYTHONPATH=src uvicorn api:app --app-dir src --host 0.0.0.0 --port 8080

Example retrieval-only request:

curl -X POST http://localhost:8080/v1/query \
  -H 'content-type: application/json' \
  -d '{
    "question":"Why did Northstar EBITDA margin fall in 2024?",
    "generate":false,
    "top_k":4
  }'

The response includes the retrieved sources, applied filter information, and retrieval latency. When generation is enabled and LLM credentials are configured, it also includes the generated answer, citation-validation result, token usage, and generation latency.

Repository structure

.
├── src/
│   ├── api.py                 FastAPI request/response layer
│   ├── service.py             coordinates retrieval and generation
│   ├── documents.py           document loading and chunking
│   ├── retrieval.py           BM25, dense and hybrid retrieval
│   ├── postgres_retrieval.py  optional PostgreSQL/pgvector retrieval
│   ├── generation.py          LLM generation and citation validation
│   ├── evaluate.py            retrieval evaluation
│   └── app.py                 CLI entry point
├── tests/
├── eval_questions.json
├── compose.yaml
├── Dockerfile
└── deploy/

The repository also contains Docker, CI, metrics, tracing, and Kubernetes examples. They are supporting infrastructure for experimentation; the core purpose of this project is to demonstrate and explore the RAG pipeline itself.

Tests

Run the offline test suite with:

pytest

The repository also contains integration and container-related checks for the database-backed path.

Current limitations

The main limitations I would address next are:

Synthetic corpus only. There is no real PDF ingestion or parsing. Real financial reports contain tables, footnotes, headers, page-numbering issues, and scanned content that this project does not cover.

Small evaluation set. Eighteen questions are useful for comparing configurations and detecting regressions, but they are not enough to make broad claims about retrieval quality.

No claim-level groundedness evaluation. Citation validation confirms that a cited page was supplied to the model, not that the page semantically supports every generated claim.

No production traffic or load testing. The repository should not be interpreted as evidence that the service has been operated under real user traffic or production SLAs.

No published LLM answer-quality results. The measured results in this README are retrieval results. I do not claim measured semantic answer quality from the current repository.

Simple authentication. The optional service API key is suitable for experimentation, not a complete production authentication or rate-limiting design.

Possible next experiments include reranking, query decomposition for multi-source questions, a larger evaluation set, and claim-level groundedness evaluation. I would evaluate each change against the existing baseline rather than assume it improves the system.

License

MIT — see LICENSE.
