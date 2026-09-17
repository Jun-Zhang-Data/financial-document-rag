# Financial Document RAG — a small, measured, explainable demo

A personal project that builds a retrieval-augmented generation pipeline over financial
reports without LangChain or LlamaIndex, so every step is visible and can be explained:

- document parsing with company / year / page metadata
- overlapping fixed-size chunking
- three retrieval strategies: BM25, dense embeddings, and a hybrid of the two
- metadata filtering that degrades gracefully instead of returning nothing
- grounded generation with citations
- deterministic citation validation
- retrieval evaluation with Recall@k, Coverage@k and MRR
- latency, token and cost reporting
- unit tests, most of which run without any ML dependency

The point of the project is not the size of the corpus. It is that every claim about
retrieval quality in this README comes from a command you can re-run.

## Measured results

From `python src/evaluate.py` on 18 questions (15 answerable, 3 deliberately
unanswerable), `top_k=4`, over a corpus of 5 documents / 39 pages / 59 chunks:

```text
retriever                           Recall@4    Coverage@4     MRR   latency (best of 3)
----------------------------------------------------------------------------------------
bm25                                    0.80          0.80    0.60       0.3ms
bm25 +pagededupe                        0.87          0.87    0.63       0.2ms
dense                                   0.87          0.67    0.50      26.6ms
dense +pagededupe                       0.87          0.73    0.50      19.3ms
hybrid(dense+bm25)                      0.87          0.80    0.59      18.5ms
hybrid(dense+bm25) +pagededupe          0.87          0.80    0.60      24.8ms
```

What this actually says, including the parts that are inconvenient:

- Dense retrieval beats BM25 on Recall@4 (0.87 vs 0.80) because it survives vocabulary
  mismatch. The question asks why the margin *fell*; the report says the margin *declined*.
  BM25 has no stemming and no synonyms, so it drops the target page out of the top four
  entirely, while the embedding model still finds it.
- BM25 beats dense on MRR (0.60 vs 0.50) and on Coverage@4. On questions phrased with the
  same words as the report ("net revenue retention", "customer acquisition cost payback
  period"), exact term matching puts the right page first; the embedding model instead
  spreads similar-sounding financial prose across several pages and retrieves the
  general business-overview page.
- The hybrid keeps dense recall and recovers BM25's coverage. That is the expected result,
  and it is the configuration the app uses by default.
- Page-level deduplication helps every retriever. Chunk overlap means two adjacent chunks
  of the same page score almost identically, so a naive top-4 can spend two slots on
  near-duplicate text — which is fatal for a comparison question that needs pages from two
  different reports.
- Neither retriever handles the question that names no company ("why did freight costs
  increase for the industrial pump manufacturer"): dense fills its top three with the
  freight company, which is exactly what that distractor was added to expose.
- BM25 is roughly a hundred times faster per query. Dense and hybrid are within
  measurement noise of each other because the single query encode (~20ms on CPU)
  dominates, and BM25 adds a fraction of a millisecond on top. On a 59-chunk corpus none
  of this matters; the point is that the embedding cost is not free and the table makes it
  explicit.

Latency is reported as the best of three runs per question, after five untimed warm-up
queries. Without that warm-up the first dense calls cost 166ms and fall to 17ms, which
silently charges model initialisation to whichever retriever runs first.

Coverage@4 is the metric that matters most here. Recall counts a question as answered if
*any* expected page is retrieved; Coverage requires *all* of them, which is what
"compare Northstar and BluePeak in 2024" genuinely needs.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env               # only needed for answer generation
```

Retrieval and evaluation need no API key. BM25 needs no model download at all:

```bash
python src/evaluate.py --retrievers bm25            # instant, pure Python
python src/evaluate.py                              # all three retrievers
python src/evaluate.py --retrievers bm25 --verbose  # per-question hit/miss detail
python -m pytest                                    # 64 tests
```

Ask questions:

```bash
python src/app.py --retriever bm25 --no-llm --question "Why did Northstar's EBITDA margin fall in 2024?"
python src/app.py                                   # interactive, hybrid retrieval, calls the LLM
python src/app.py --dedupe-pages --top-k 5
```

Answer generation needs `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`. Nothing else does.

## How it fits together

```text
data/*.txt
   │  parse headers and [PAGE n] markers          documents.py
   ▼
Chunk(company, year, page, text) + citation
   │  90-word windows, 15-word overlap
   ▼
metadata filter (all companies, all years)        retrieval.py
   │  relax if the filter empties the pool
   ▼
BM25 │ dense embeddings │ RRF hybrid
   │  optional page-level dedupe, top-k
   ▼
prompt with numbered, citation-tagged sources     generation.py
   ▼
LLM answer  →  citation validation, tokens, cost  app.py / evaluate.py
```

```text
.
├── data/                     5 synthetic annual reports, 39 pages
│   ├── northstar_2023.txt    industrial manufacturer
│   ├── northstar_2024.txt
│   ├── bluepeak_2023.txt     software company
│   ├── bluepeak_2024.txt
│   └── harborlight_2024.txt  freight company: a deliberate lexical distractor
├── src/
│   ├── documents.py          parsing, chunking, corpus stats
│   ├── retrieval.py          filters, BM25, dense, hybrid
│   ├── generation.py         prompt, LLM call, citation validation, cost
│   ├── app.py                CLI
│   └── evaluate.py           Recall@k, Coverage@k, MRR, answer grading
├── tests/                    64 tests, no API key needed
├── conftest.py               puts src/ on sys.path for the tests
├── pytest.ini
├── eval_questions.json       18-question gold set
├── REVIEW_01_initial.md      first code review of this project
├── requirements.txt          pinned
└── requirements-dev.txt
```

## Design decisions, and what they cost

### Parsing

Documents carry a header and page markers:

```text
COMPANY: Northstar Industrial
YEAR: 2024

[PAGE 18]
Adjusted EBITDA decreased to EUR 246 million ...
```

Company, year and page travel with every chunk, so a citation can be produced without a
second lookup. A malformed document raises `DocumentFormatError` naming the file and the
missing field rather than an `AttributeError` from a failed regex.

Real reports are PDFs. Parsing them (Docling, PyMuPDF) is a genuinely harder problem —
tables, multi-column layouts, headers and footers — and is deliberately out of scope. The
text format keeps the metadata contract explicit.

### Chunking

Fixed 90-word windows with 15 words of overlap. Overlap means a sentence sitting on a
boundary still appears whole in one of the two neighbours.

The cost is real and measurable: 39 pages become 59 chunks, 20 pages split into more than
one chunk, and adjacent chunks share text. That duplication is exactly why
`--dedupe-pages` improves Coverage@4. A production system would chunk on headings or
semantic boundaries instead.

### Metadata filtering

Filters are extracted from the question and applied as OR within a field, AND across
fields:

```text
"How did Northstar's margin change from 2023 to 2024?"
  → company in {Northstar Industrial} AND year in {2023, 2024}   → 29 candidate chunks
```

Three details matter, and each of them was a bug in the first version of this project:

1. *All* years are extracted, not just the first. Otherwise a cross-year question is
   silently restricted to 2023 and can never answer.
2. *All* companies are extracted, not just the first match. Otherwise a comparison
   question silently drops one issuer.
3. If a filter empties the candidate pool, it is relaxed step by step
   (company AND year → company → year → unfiltered) and the relaxation is reported:

```text
Metadata filter: company in {Northstar Industrial} (relaxed from: company in {Northstar Industrial} AND year in {2026})
Note: the requested filter matched nothing, so it was relaxed.
```

A hard filter that returns nothing is worse than no filter: the generator gets no
evidence, cannot even attempt a grounded answer, and the user pays for a call that was
guaranteed to fail. Relaxing and disclosing is the better failure mode.

### Retrieval

BM25 is implemented directly (Okapi, k1=1.5, b=0.75) so the lexical baseline has no hidden
behaviour. Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2` with embeddings
L2-normalised at encode time, which makes a dot product equal to cosine similarity.

The hybrid fuses the two with reciprocal rank fusion, `score = Σ 1/(60 + rank)`. RRF
combines *rankings*, which avoids having to normalise a BM25 score and a cosine similarity
onto a common scale — a comparison that has no principled answer.

A baseline exists because "we used embeddings" is not a result. Without BM25 in the table
there is no way to know whether the embedding model earns its latency, and on this corpus
the honest answer is "on recall yes, on ranking no".

### Generation

Only retrieved chunks are sent. The prompt requires the model to answer from context, to
reply `INSUFFICIENT_EVIDENCE` when the context does not support an answer, and to cite
using the exact citation printed above each source.

Hardening that the demo does need:

- if retrieval returned nothing, no request is made at all
- `temperature=0` for reproducibility, with a fallback for models that reject the
  parameter
- a 30-second timeout and two retries
- credentials are only required when a call is actually going to happen, so retrieval,
  evaluation and tests all run without them

### Citation validation

Every citation in the answer is parsed and checked against the citations that were
actually supplied:

```text
CITATION CHECK
Cited: [Northstar Industrial 2024, p.18]
Supported by retrieved context: 1/1
```

An answer citing `p.19` when only `p.18` was retrieved is flagged as `UNSUPPORTED`. This is
deterministic, costs nothing, and catches the most visible RAG failure: a plausible
citation pointing at a page the model never saw. It does not verify that the *claim*
matches the source — that needs a judge model or a human, and is listed below as future
work.

### Evaluation set

18 questions in `eval_questions.json`, chosen so the metric can fail:

- single-source questions
- cross-year questions requiring two reports
- cross-company comparison questions
- a question that names no company at all ("the industrial pump manufacturer"), where
  metadata filtering cannot help and the freight company competes lexically
- a stated negative ("Did BluePeak pay a dividend in 2024?" — the report says no dividend
  is proposed)
- three unanswerable questions (a year outside the corpus, a fact never stated) where the
  correct behaviour is refusal

`tests/test_eval_set.py` asserts that every expected page exists in the corpus, and that
every answerable question leaves a candidate pool larger than `2 × top_k`. That second
test exists because the first version of this project had a 9-chunk corpus in which
filtering left exactly 3 candidates for `top_k=3`: Recall@3 was 1.00 by construction and
would have stayed 1.00 with the embedding model deleted.

### Answer grading, latency and cost

`python src/evaluate.py --llm` grades generated answers: share of answers containing a
citation, share with no unsupported citation, and refusal rate on the three unanswerable
questions. It reports total tokens and, if prices are configured, estimated cost.

`app.py` reports retrieval, generation and end-to-end latency separately, plus input and
output tokens per query. Set `INPUT_PRICE_PER_1M_TOKENS` and `OUTPUT_PRICE_PER_1M_TOKENS`
in `.env` to turn tokens into an estimated dollar figure:

```text
cost = input tokens × input price + output tokens × output price
```

Sending four chunks instead of five whole reports is the point of RAG: it is what keeps
the input token count in the hundreds.

## Known limitations



- The corpus is synthetic. It exercises the pipeline honestly, but it is not evidence that
  the pipeline handles real PDF filings.
- 15 answerable questions is a small gold set. A single question changes Recall@4 by 0.07,
  so the differences between the retrievers in that table are directional, not
  statistically significant.
- The answer-grading path (`--llm`) has not been run here: this environment has no API
  key. The retrieval numbers above are reproduced output; the answer metrics are not.
- Everything is in memory and rebuilt at startup. There is no index persistence and no
  caching.
- Citation validation checks provenance, not correctness. A supported citation attached to
  a wrong claim would pass.
- The BM25 tokenizer has no stemming, and possessives ("Northstar's") split into two
  tokens, which measurably shifts lexical rankings.

## What production would need

1. real PDF parsing (Docling / PyMuPDF), including tables
2. a persistent index (pgvector, Elasticsearch, OpenSearch) instead of in-memory vectors
3. a cross-encoder reranker on top of hybrid retrieval
4. heading-aware or semantic chunking
5. a much larger gold set, and answer-quality grading by a judge model with human spot checks
6. a FastAPI service with request tracing
7. caching of embeddings and of repeated questions
8. monitoring of retrieval quality, latency and cost per query in production

