"""Retrieval and answer evaluation.

Retrieval metrics run offline with no API key and, for BM25, with no model download:

    python src/evaluate.py --retrievers bm25
    python src/evaluate.py                      # bm25, dense and hybrid side by side
    python src/evaluate.py --llm                # also grade answers (needs OPENAI_API_KEY)

Metrics
    Recall@k      question counts as a hit if any expected source is in the top k
    Coverage@k    every expected source is in the top k (matters for comparison questions)
    MRR           1 / rank of the best-ranked expected source, averaged over questions
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from documents import corpus_stats, load_chunks
from generation import generate_answer, pricing
from retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    BaseRetriever,
    HybridRetriever,
    build_retriever,
)

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval_questions.json"

SourceKey = tuple[str, str, str]


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def expected_keys(question: dict) -> list[SourceKey]:
    return [(s["company"], s["year"], s["page"]) for s in question["expected_sources"]]


def positive_int(value: str) -> int:
    """argparse type that rejects zero and negatives with a readable message."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers")
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25", "dense", "hybrid"],
        choices=["bm25", "dense", "hybrid"],
    )
    parser.add_argument("--top-k", type=positive_int, default=4)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="also grade generated answers: refusal on unanswerable questions, citation validity",
    )
    parser.add_argument(
        "--llm-retriever",
        default="bm25",
        choices=["bm25", "dense", "hybrid"],
        help="which retriever feeds the answer grading (default: bm25, the service default)",
    )
    parser.add_argument("--verbose", action="store_true", help="print per-question detail")
    return parser.parse_args()


def warm_up(retriever: BaseRetriever, max_calls: int = 30, tolerance: float = 0.25) -> int:
    """Run untimed queries until latency stops improving; return how many it took.

    A fixed warm-up count is not enough. The first dense call costs an order of magnitude
    more than a warm one, and how many calls that takes to decay depends on the machine.
    With a fixed count the first timed retriever absorbs the remainder of model and
    thread-pool initialisation, which made dense look several times slower than the hybrid
    that contains it.
    """
    timings: list[float] = []

    for _ in range(max_calls):
        start = time.perf_counter()
        retriever.retrieve("warm up", top_k=1)
        timings.append(time.perf_counter() - start)

        if len(timings) >= 5 and max(timings[-3:]) <= min(timings) * (1 + tolerance):
            break

    return len(timings)


def evaluate_retrieval(
    retriever: BaseRetriever,
    questions: Sequence[dict],
    top_k: int,
    verbose: bool,
    dedupe_pages: bool = False,
) -> dict[str, float]:
    answerable = [q for q in questions if q["answerable"]]

    hits = 0
    covered = 0
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    # Untimed warm-up: the first dense queries pay for lazy model initialisation and
    # thread-pool spin-up, which would otherwise be charged to whichever retriever runs
    # first.
    warm_up(retriever)

    for question in questions:
        # Best of three: the wall clock in a shared environment is noisy, and the minimum
        # is the closest thing to the cost of the work itself.
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            results = retriever.retrieve(
                question["question"], top_k=top_k, dedupe_pages=dedupe_pages
            )
            best = min(best, time.perf_counter() - start)
        latencies.append(best)

        if not question["answerable"]:
            continue

        retrieved = [r.chunk.source_key for r in results]
        expected = expected_keys(question)

        found = [key for key in expected if key in retrieved]
        hit = bool(found)
        # "all" questions (comparisons, cross-year) only count as covered when every
        # required page is present. For "any" questions the expected pages are
        # interchangeable, so coverage is the same thing as a hit.
        full = len(found) == len(expected) if question["match"] == "all" else hit

        hits += int(hit)
        covered += int(full)

        ranks = [retrieved.index(key) + 1 for key in found]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)

        if verbose:
            status = "HIT " if hit else "MISS"
            if hit and not full:
                status = "PART"
            print(f"  [{status}] {question['question']}")
            print(f"         expected: {expected}")
            print(f"         retrieved: {retrieved}")

    n = len(answerable)

    return {
        "recall": hits / n,
        "coverage": covered / n,
        "mrr": sum(reciprocal_ranks) / n,
        "median_latency_ms": 1000 * statistics.median(latencies),
    }


def evaluate_answers(
    retriever: BaseRetriever,
    questions: Sequence[dict],
    top_k: int,
) -> dict[str, float | None]:
    """Grade generated answers. Requires an API key and spends tokens.

    ``cost_usd`` is None unless token prices are configured, which is why the values are
    optional.
    """
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    cited_ok = 0
    no_unsupported = 0
    refused_correctly = 0
    input_tokens = 0
    output_tokens = 0
    latencies: list[float] = []

    for question in questions:
        retrieved = retriever.retrieve(question["question"], top_k=top_k)

        start = time.perf_counter()
        result = generate_answer(question["question"], retrieved)
        latencies.append(time.perf_counter() - start)

        input_tokens += result.usage.input_tokens
        output_tokens += result.usage.output_tokens

        if question["answerable"]:
            cited_ok += int(result.citations.has_citations)
            no_unsupported += int(result.citations.is_valid)
            marker = "cited" if result.citations.has_citations else "NO CITATION"
            if result.citations.unsupported:
                marker = f"UNSUPPORTED {result.citations.unsupported}"
        else:
            refused_correctly += int(result.refused)
            marker = "refused" if result.refused else "DID NOT REFUSE"

        print(f"  [{marker}] {question['question']}")

    input_price, output_price = pricing()
    total_cost = None
    if input_price is not None and output_price is not None:
        total_cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    return {
        "answers_with_citations": cited_ok / len(answerable),
        "answers_without_unsupported_citations": no_unsupported / len(answerable),
        "correct_refusals": refused_correctly / len(unanswerable) if unanswerable else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": total_cost,
        "mean_latency_s": sum(latencies) / len(latencies),
    }


def shared_retriever(
    kind: str,
    built: dict[str, BaseRetriever],
    chunks,
    embedding_model: str,
) -> BaseRetriever:
    """Build a retriever, reusing already-built components.

    The hybrid must reuse the existing dense and BM25 instances rather than construct new
    ones. Two SentenceTransformer models in one process encode the corpus twice and then
    contend for the same CPU threads, which made the reported latencies swing by 3x
    between runs while the quality metrics stayed identical.
    """
    if kind in built:
        return built[kind]

    if kind == "hybrid":
        dense = shared_retriever("dense", built, chunks, embedding_model)
        bm25 = shared_retriever("bm25", built, chunks, embedding_model)
        built[kind] = HybridRetriever([dense, bm25])
    else:
        built[kind] = build_retriever(kind, chunks, embedding_model)

    return built[kind]


def main() -> None:
    args = parse_args()
    load_dotenv()

    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    chunks = load_chunks()
    stats = corpus_stats(chunks)
    questions = load_questions()
    answerable = [q for q in questions if q["answerable"]]

    print("CORPUS")
    print(
        f"{stats['documents']} documents, {stats['pages']} pages, {stats['chunks']} chunks; "
        f"{stats['split_pages']} pages produced more than one chunk; "
        f"mean {stats['mean_chunk_words']} words per chunk (max {stats['max_chunk_words']})"
    )
    print(
        f"\nEVALUATION SET\n{len(questions)} questions: {len(answerable)} answerable, "
        f"{len(questions) - len(answerable)} unanswerable; top_k={args.top_k}"
    )

    rows: list[tuple[str, dict[str, float]]] = []
    built: dict[str, BaseRetriever] = {}

    for kind in args.retrievers:
        try:
            retriever = shared_retriever(kind, built, chunks, embedding_model)
        except ImportError as exc:
            print(f"\nSkipping '{kind}' retriever: {exc}")
            continue

        for dedupe in (False, True):
            label = f"{retriever.name}{' +pagededupe' if dedupe else ''}"
            if args.verbose:
                print(f"\nRETRIEVER: {label}")
            metrics = evaluate_retrieval(
                retriever, questions, args.top_k, args.verbose and not dedupe, dedupe
            )
            rows.append((label, metrics))

    if not rows:
        print("\nNo retrievers could be evaluated.")
        return

    k = args.top_k
    print("\nRETRIEVAL RESULTS")
    header = f"{'retriever':<32}{f'Recall@{k}':>12}{f'Coverage@{k}':>14}{'MRR':>8}{'latency (med/best3)':>22}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(
            f"{name:<32}{m['recall']:>12.2f}{m['coverage']:>14.2f}{m['mrr']:>8.2f}"
            f"{m['median_latency_ms']:>10.1f}ms"
        )

    print(
        "\nLatency is the median across questions of the best of three timed runs, after an "
        "adaptive untimed warm-up (queries are repeated until latency stops improving)."
    )
    print(
        "\nRecall counts a question as a hit if any expected page is in the top k. "
        "Coverage requires every expected page, which is what comparison questions need. "
        "MRR rewards putting the right page first. '+pagededupe' keeps only the best "
        "chunk per page, so overlapping neighbours cannot crowd out a second source."
    )

    if not args.llm:
        print("\nAnswer grading skipped. Re-run with --llm to grade citations and refusals.")
        return

    retriever = shared_retriever(args.llm_retriever, built, chunks, embedding_model)

    print(f"\nANSWER GRADING (retriever: {retriever.name})")
    answer_metrics = evaluate_answers(retriever, questions, args.top_k)

    print("\nANSWER RESULTS")
    print(f"Answers containing at least one citation: {answer_metrics['answers_with_citations']:.2f}")
    print(
        "Answers with no unsupported citation:     "
        f"{answer_metrics['answers_without_unsupported_citations']:.2f}"
    )
    print(f"Correct refusals on unanswerable questions: {answer_metrics['correct_refusals']:.2f}")
    print(
        f"Tokens: {answer_metrics['input_tokens']} in, {answer_metrics['output_tokens']} out"
    )
    if answer_metrics["cost_usd"] is None:
        print("Cost: set INPUT_PRICE_PER_1M_TOKENS and OUTPUT_PRICE_PER_1M_TOKENS to estimate.")
    else:
        print(f"Estimated cost for the run: ${answer_metrics['cost_usd']:.4f}")
    print(f"Mean generation latency: {answer_metrics['mean_latency_s']:.2f}s")


if __name__ == "__main__":
    main()
