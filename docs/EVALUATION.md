# Evaluation: design notes

Notes on how retrieval is measured here, what the numbers turned out to be, and the
decisions the measurements changed. Everything below is reproducible with:

```bash
PYTHONPATH=src python src/evaluate.py --retrievers bm25    # no model download
PYTHONPATH=src python src/evaluate.py                      # bm25, dense, hybrid
PYTHONPATH=src python src/evaluate.py --verbose            # per-question detail
```

## The corpus is synthetic, and that is a deliberate trade

`data/` holds five short reports for three invented companies: Northstar Industrial (2023
and 2024), BluePeak Software (2023 and 2024), and Harborlight Logistics (2024). 39 pages,
59 chunks.

I wrote them rather than using real filings because the evaluation needs page-level ground
truth. To score Coverage@k I have to know exactly which page answers each question, and
labelling that by hand across real 10-Ks is slow and error-prone. Synthetic pages let me
control the retrieval problem directly: I know where every answer lives, I know which pages
are near-duplicates, and I can plant specific traps.

Harborlight exists only as a lexical distractor. It discusses container rates, freight cost
pass-through and energy prices — the same vocabulary Northstar uses to explain its margin
decline. A retriever that matches on surface words gets pulled towards it.

The cost of this choice is real: nothing here proves the pipeline survives a genuine
annual-report PDF, with tables, footnotes, page furniture and inconsistent page labels.
See the limitations in the README.

## Metrics

| Metric | Definition | Why it is here |
|---|---|---|
| Recall@k | any expected page is in the top k | the usual hit-rate; the most forgiving |
| Coverage@k | *every* expected page is in the top k | comparison and cross-year questions are only answerable if all the required pages arrive |
| MRR | 1 / rank of the best expected page | rewards putting the right page first, which is what matters when the context window is tight |

Coverage@k is the metric I care most about. "Compare the EBITDA margins of Northstar and
BluePeak in 2024" cannot be answered from one page, so a retriever that fetches one of the
two and stops has failed the question while still scoring a Recall@k hit.

`eval_questions.json` has 18 questions: 15 answerable, 3 unanswerable. The unanswerable
ones exist to test refusal rather than retrieval. The set includes cross-year questions, two
cross-company comparisons, a question that names no company at all, and a stated negative
("Did BluePeak pay a dividend in 2024?").

The pool has to be big enough for the metric to mean something. Every answerable question
leaves a candidate pool of more than 2 × top_k after metadata filtering, and
`tests/test_eval_set.py` asserts it. Without that guard a filter can narrow the pool to
roughly top_k, at which point recall is 1.00 by construction and the ranking is never
tested.

## Results

Measured on this corpus at top_k=4. The quality columns were bit-identical across three
consecutive runs.

```text
retriever                           Recall@4    Coverage@4     MRR   latency (med/best3)
----------------------------------------------------------------------------------------
bm25                                    0.80          0.80    0.60       0.1ms
bm25 +pagededupe                        0.87          0.87    0.63       0.1ms
dense                                   0.87          0.67    0.50      22.8ms
dense +pagededupe                       0.87          0.73    0.50      20.7ms
hybrid(dense+bm25)                      0.73          0.73    0.52      24.7ms
hybrid(dense+bm25) +pagededupe          0.87          0.80    0.56      21.4ms
```

`bm25 +pagededupe` wins or ties every quality metric at roughly a two-hundredth of the
latency, so it is the default for both the CLI and the service. On a corpus of 59 chunks
whose questions largely reuse the reports' own vocabulary, a 22MB embedding model does not
earn its cost. I would not generalise that to a large or jargon-heavy corpus — it is a
result about this data, which is exactly why the harness exists.

Per-question detail behind the table (15 answerable questions):

```text
bm25          12/15   misses margin-fall, cross-year, comparison
bm25 +dd      13/15   misses cross-year, comparison
dense         13/15   misses revenue-2024, freight-no-company
dense +dd     13/15   misses revenue-2024, freight-no-company
hybrid        11/15   misses revenue-2024, cross-year, comparison, freight-no-company
hybrid +dd    13/15   misses cross-year, freight-no-company
```

What the individual failures show:

- **Dense survives vocabulary mismatch.** The question asks why the margin *fell*; the
  report says it *declined*. BM25 has no stemming and no synonyms, so that page drops out
  of its top four. This is the one question where the embedding model clearly pays for
  itself.
- **BM25 wins when the question quotes the report.** "What was Northstar's revenue in 2024?"
  is a lexical match on a page dense ranks fifth.
- **Neither handles the question that names no company.** "Why did freight costs increase
  for the industrial pump manufacturer?" has no company token to filter on, and both
  retrievers drift to Harborlight. That is precisely what the distractor document was added
  to expose, and it stays in the set as a documented failure.
- **Page dedupe improves every retriever.** Chunk overlap means two adjacent chunks of one
  page score almost identically, so a naive top-4 can spend two slots on near-duplicate
  text and crowd out the second document a comparison needs. Dedupe is the measurable cost
  of the overlap I chose in chunking.

## Rank fusion has to happen inside the filtered pool

RRF fuses *ranks*, not scores, which is why it can combine a BM25 score with a cosine
similarity without inventing a shared scale. It also makes fusion sensitive to what is in
the pool: rank depends on how many other documents are present.

My first implementation ranked all 59 chunks, fused, and applied the metadata filter
afterwards — while the single retrievers filtered first and then ranked. Those are not the
same operation. Each sub-retriever demotes pool members by a different number of
intervening non-pool chunks, so the fused order changes. Comparing both orders across the
evaluation set, 14 of 18 questions produced a different top-4, and on the Northstar
margin-fall question the corpus-wide version pushed the expected page out of the top four
entirely.

Fixing it cost the hybrid its apparent advantage: Recall@4 went from 0.87 to 0.73, and BM25
became the best configuration. A result that only existed because ranking happened before
filtering was not a result. Ranking is now a first-class operation —
`BaseRetriever.rank(query, candidate_ids)` orders a pool, `HybridRetriever.rank` fuses
sub-retriever ranks *within* that pool, and `HybridRetriever.score_all` raises rather than
offer a second, inconsistent path. The regression test in `tests/test_retrieval.py` adds
documents that the filter excludes and asserts the surviving order does not move.

## Latency methodology

Latency is the median across questions of the best of three timed runs, after an untimed
warm-up. Two things had to be fixed before the column meant anything:

The warm-up is adaptive. A fixed five-call warm-up was not enough on my machine: the first
dense query costs an order of magnitude more than a warm one, so whichever retriever was
timed first absorbed the remainder of model and thread-pool initialisation. `warm_up()` now
repeats queries until latency stops improving.

The hybrid reuses the dense and BM25 instances instead of building its own. Constructing a
second `SentenceTransformer` in the same process encodes the corpus twice and then contends
for the same CPU threads, which made the hybrid appear faster than the dense retriever
inside it.

Even with both fixed, the dense and hybrid figures move by up to a factor of two between
runs on a loaded laptop, while the quality metrics do not move at all. Treat the latency
column as an order of magnitude, not a benchmark: BM25 is sub-millisecond, anything
involving the embedding model is tens of milliseconds. That two-orders-of-magnitude gap is
the part of the measurement I would defend.

Both defects were in the measurement code rather than the retrievers, which is why
`tests/test_evaluate.py` exists. A harness that reports the wrong number is worse than no
harness, because it is believed. The BM25 row of the table above is also asserted as a test,
so a change that moves the published numbers fails the build instead of quietly disagreeing
with the documentation.

## Answer grading

`--llm` additionally grades generated answers: the share containing at least one citation,
the share with no unsupported citation, the refusal rate on the three unanswerable
questions, tokens consumed, and an estimated cost if prices are configured.

Citation validation is deterministic and cheap: citations are parsed back out of the answer
and compared against the citations actually supplied as context, so a plausible reference to
a page the model never saw is reported as unsupported. It checks provenance, not
correctness — a citation can be real and the sentence attached to it still wrong.

I have not run this path. It needs an API key I do not have configured, so every answer
metric in this repository is unmeasured. The retrieval numbers above are all reproduced
command output.
