# Financial Document RAG

A hands-on Retrieval-Augmented Generation (RAG) project for question answering over a small synthetic set of financial reports.

> [!NOTE]
> This is a portfolio and learning project. It is **not** a production-proven financial system.  
> The corpus is synthetic, the evaluation set is small, and the service has not been operated under real production traffic.

## Highlights

- BM25, dense, and hybrid retrieval
- Metadata filtering by company and year
- Page-level deduplication
- Retrieval evaluation with Recall@k, Coverage@k, and MRR
- LLM generation using retrieved context
- `INSUFFICIENT_EVIDENCE` refusal behavior
- Deterministic citation validation
- FastAPI API
- Optional PostgreSQL + pgvector retrieval path

## Architecture

```text
Client
  |
  v
FastAPI /v1/query
  |
  v
RAGService
  |
  v
Metadata filtering
(company / year)
  |
  v
Retrieval
(BM25 / Dense / Hybrid)
  |
  v
Page deduplication
  |
  v
Top-k evidence
  |
  v
LLM generation
  |
  v
Citation validation
  |
  v
JSON response
```

Main files:

| File | Responsibility |
|---|---|
| `src/api.py` | HTTP request and response layer |
| `src/service.py` | Coordinates retrieval and generation |
| `src/retrieval.py` | BM25, dense, and hybrid retrieval |
| `src/postgres_retrieval.py` | PostgreSQL/pgvector retrieval |
| `src/generation.py` | LLM generation and citation validation |
| `src/evaluate.py` | Retrieval evaluation |

## Dataset

The repository contains five short synthetic financial reports representing three invented companies.

| Item | Count |
|---|---:|
| Documents | 5 |
| Pages | 39 |
| Chunks | 59 |
| Evaluation questions | 18 |
| Answerable questions | 15 |
| Intentionally unanswerable questions | 3 |

I use synthetic data so I can define page-level ground truth and inspect retrieval failures directly.

This also means the project does **not** demonstrate real PDF ingestion or parsing.

## Retrieval evaluation

The default evaluation uses `top_k=4`.

| Retriever | Recall@4 | Coverage@4 | MRR |
|---|---:|---:|---:|
| BM25 | 0.80 | 0.80 | 0.60 |
| **BM25 + page dedupe** | **0.87** | **0.87** | **0.63** |
| Dense | 0.87 | 0.67 | 0.50 |
| Dense + page dedupe | 0.87 | 0.73 | 0.50 |
| Hybrid (dense + BM25) | 0.73 | 0.73 | 0.52 |
| Hybrid + page dedupe | 0.87 | 0.80 | 0.56 |

These numbers are specific to this small synthetic corpus. They are not evidence that BM25 is generally better than dense or hybrid retrieval.

On this dataset, **BM25 + page deduplication** gave the strongest overall combination of Recall, Coverage, and MRR, so I keep it as the default local baseline.

### Example retrieval failure

For:

> **Why did Northstar EBITDA margin fall in 2024?**

With `top_k=2`:

| Retriever | Rank 1 | Rank 2 | Page 18 retrieved? |
|---|---|---|---|
| BM25 | p.21 | p.4 | No |
| Dense | p.26 | p.18 | Yes |
| Hybrid | p.21 | p.18 | Yes |

Page 18 contains the answer-bearing evidence.

This was useful because it showed two things:

1. semantic retrieval can recover a page that lexical retrieval misses for a particular query;
2. improving one query does not necessarily mean a retriever performs better across the full evaluation set.

## Evaluation metrics

### Recall@k

`Recall@4` checks whether **at least one** expected evidence page appears in the top four results.

### Coverage@k

`Coverage@4` checks whether **all** expected evidence pages appear in the top four.

Coverage matters for multi-source questions. A question comparing two companies or years may require evidence from more than one page.

### MRR

MRR rewards retrieving the correct evidence near the top of the ranking.

## Why page deduplication matters

Some pages are split into overlapping chunks.

Without page deduplication:

```text
1. Page 18, chunk 0
2. Page 18, chunk 1
3. Page 16, chunk 0
4. Page 21, chunk 0
```

With page deduplication:

```text
1. Page 18
2. Page 16
3. Page 21
4. Another distinct page
```

On this evaluation set, BM25 Recall@4 and Coverage@4 both improved from `0.80` to `0.87` after page deduplication.

## Retrieval approaches

### BM25

Lexical retrieval based on words shared between the query and document text.

### Dense retrieval

Embeds the query and chunks into vectors and ranks them by semantic similarity.

### Hybrid retrieval

Combines BM25 and dense rankings with Reciprocal Rank Fusion (RRF).

RRF combines **rank positions** rather than directly combining BM25 and vector-similarity scores, which are on different scales.

## Generation and grounding

After retrieval, selected chunks are formatted as numbered sources and passed to the LLM.

The generation instructions require the model to:

- answer only from supplied context;
- return `INSUFFICIENT_EVIDENCE` when the context is insufficient;
- cite factual claims using supplied document/page citations;
- avoid inventing citations.

If retrieval returns no evidence, the code does not call the LLM.

## Citation validation

After generation, the code extracts citations such as:

```text
[Northstar Industrial 2024, p.18]
```

It checks whether each cited page was actually included in the retrieved context.

This can detect an invented citation to a page the model never received.

> [!IMPORTANT]
> Citation validation checks **provenance**, not full semantic correctness.  
> A model can cite the correct page and still misstate what the page says. Claim-level groundedness evaluation is not implemented in this version.

## Quick start

<details>
<summary><strong>Windows PowerShell</strong></summary>

### 1. Create an environment

```powershell
conda create -n financial-rag python=3.12 -y
conda activate financial-rag
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH="src"
```

### 2. Run retrieval without an LLM

```powershell
python src/app.py `
  --question "Why did Northstar EBITDA margin fall in 2024?" `
  --no-llm
```

### 3. Run the retrieval evaluation

```powershell
python src/evaluate.py --retrievers bm25 dense hybrid
```

### 4. Start the API

```powershell
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080/docs
```

</details>

<details>
<summary><strong>macOS / Linux</strong></summary>

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export PYTHONPATH=src
```

### 2. Run retrieval without an LLM

```bash
python src/app.py \
  --question "Why did Northstar EBITDA margin fall in 2024?" \
  --no-llm
```

### 3. Run the retrieval evaluation

```bash
python src/evaluate.py --retrievers bm25 dense hybrid
```

### 4. Start the API

```bash
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8080
```

</details>

## Retrieval-only API example

A retrieval-only request does not require LLM credentials.

<details>
<summary><strong>PowerShell example</strong></summary>

```powershell
$body = @{
    question = "Why did Northstar EBITDA margin fall in 2024?"
    generate = $false
    top_k = 4
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8080/v1/query" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

</details>

<details>
<summary><strong>curl example</strong></summary>

```bash
curl -X POST http://127.0.0.1:8080/v1/query \
  -H 'content-type: application/json' \
  -d '{
    "question": "Why did Northstar EBITDA margin fall in 2024?",
    "generate": false,
    "top_k": 4
  }'
```

</details>

## Repository structure

```text
.
├── src/
│   ├── api.py
│   ├── service.py
│   ├── documents.py
│   ├── retrieval.py
│   ├── postgres_retrieval.py
│   ├── generation.py
│   ├── evaluate.py
│   └── app.py
├── tests/
├── data/
├── deploy/
├── eval_questions.json
├── compose.yaml
├── Dockerfile
└── README.md
```

The repository also contains Docker, CI, metrics, tracing, and Kubernetes examples. I treat these as supporting infrastructure for experimentation rather than evidence that I have operated this service at production scale.

## Tests

```bash
python -m pytest
```

## Current limitations

- **Synthetic corpus only** — no real PDF ingestion or parsing.
- **Small evaluation set** — 18 questions are useful for comparison and regression testing, but not broad quality claims.
- **No claim-level groundedness evaluation** — citation validation checks provenance, not semantic correctness.
- **No production traffic or load testing.**
- **No published LLM answer-quality benchmark** — the measured results above are retrieval metrics.
- **Simple authentication** — the optional API key is not a complete production authentication or rate-limiting design.

Possible next experiments:

- reranking;
- query decomposition for multi-source questions;
- a larger evaluation set;
- claim-level groundedness evaluation.

## License

MIT — see [LICENSE](LICENSE).
