# Architecture

## Why this shape

The retrieval core is deliberately transparent: custom parsing, BM25, dense retrieval, RRF
fusion, citation validation, and an evaluation set. Those pieces are easy to reason about
and easy to benchmark, and the evaluation harness depends on being able to swap one of them
at a time.

The production path adds infrastructure around that core instead of hiding it behind a
large RAG framework.

```text
Client
  |
  v
FastAPI  ---- /metrics ----> Prometheus / Grafana
  |  \\---- OTLP traces ---> OpenTelemetry backend
  |
  +--> API-key gate (optional)
  |
  v
RAGService (one long-lived instance per pod)
  |
  +--> memory backend: BM25 / dense / RRF (dev + evaluation)
  |
  +--> PostgreSQL backend
         |-- metadata filters: B-tree
         |-- lexical search: tsvector + GIN
         |-- dense search: pgvector + HNSW cosine index
         `-- fusion: reciprocal rank fusion in SQL
  |
  v
OpenAI Responses API
  |-- grounding rules in `instructions`
  |-- retrieved documents in user input
  |-- store=false
  `-- deterministic citation validation after generation
```

## Production choices

### FastAPI + Uvicorn

The CLI remains useful for evaluation, but production consumers need a stable HTTP
contract, health/readiness probes, request IDs, auth, and machine-readable errors. One
Uvicorn process runs per container; Kubernetes scales containers rather than duplicating
the embedding model several times inside one pod.

### PostgreSQL + pgvector

PostgreSQL is the system of record for indexed chunks. Full-text search and vector search
live in the same database, which keeps metadata filters transactional and avoids an extra
vector-database service for this workload. Hybrid results are combined with RRF inside the
filtered candidate pool, matching the in-memory implementation — fusing over the whole
corpus and filtering afterwards changes the order, because RRF is rank-based.

Query embeddings are cached in a bounded per-instance LRU dict. A `functools.lru_cache` on
the method would key on `self` and keep the retriever and its model alive for as long as the
cache does.

The schema uses:

- B-tree index on `(company, year)`
- GIN index on generated `tsvector`
- HNSW cosine index on `vector(384)`
- unique key on `(source_file, page, chunk_index)` for idempotent ingestion

### Ingestion

`src/ingest.py` embeds chunks in a batch and upserts them. A `--replace` initial load drops
the HNSW index before the batch and rebuilds it afterwards, avoiding row-by-row graph
maintenance during a full refresh.

### Observability

Prometheus metrics are always local to the service and OpenTelemetry is opt-in through
`OTEL_EXPORTER_OTLP_ENDPOINT`. Logs are JSON and include a request ID. This lets a real
deployment use Grafana/Prometheus, Datadog, New Relic, Honeycomb, Azure Monitor, or another
OTLP-compatible backend without changing business logic.

## Scaling notes

The local SentenceTransformer preserves the existing benchmark. For much larger traffic,
separate embedding inference into its own service or use a managed embedding API so API
pods do not each hold a model in memory. The PostgreSQL retrieval interface can stay the
same.

For multi-tenant financial data, add a `tenant_id` column to every chunk, include it in all
filters, and use PostgreSQL row-level security. Do not rely on application-only tenant
filtering.
