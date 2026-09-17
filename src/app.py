"""Interactive RAG demo over the financial documents in ./data.

Usage:
    python src/app.py                                  # interactive, hybrid retrieval
    python src/app.py --retriever bm25                 # lexical only, no model download
    python src/app.py --question "..." --top-k 5       # one-shot
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

from dotenv import load_dotenv

from documents import corpus_stats, load_chunks
from generation import GenerationResult, generate_answer, pricing
from retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    BaseRetriever,
    RetrievalResult,
    build_retriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Financial document RAG demo")
    parser.add_argument(
        "--retriever",
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="retrieval strategy (default: hybrid = dense + BM25 via reciprocal rank fusion)",
    )
    parser.add_argument("--top-k", type=int, default=4, help="chunks passed to the LLM")
    parser.add_argument(
        "--dedupe-pages",
        action="store_true",
        help="keep only the best-scoring chunk per source page",
    )
    parser.add_argument("--question", help="ask one question and exit")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="show retrieval only, do not call the LLM",
    )
    return parser.parse_args()


def print_retrieval(retriever: BaseRetriever, question: str, retrieved: List[RetrievalResult]) -> None:
    filter_result = retriever.filter_for(question)

    print("\nRETRIEVAL")
    print(f"Strategy: {retriever.name}")
    print(f"Metadata filter: {filter_result.describe()}")
    print(f"Candidate chunks after filtering: {len(filter_result.candidate_ids)}")

    if filter_result.relaxed:
        print("Note: the requested filter matched nothing, so it was relaxed.")

    if not retrieved:
        print("No chunks retrieved.")
        return

    for result in retrieved:
        print(f"\n{result.rank}. {result.chunk.citation} | score={result.score:.4f}")
        print(f"   {result.chunk.text}")


def print_generation(result: GenerationResult, retrieval_s: float, generation_s: float) -> None:
    print("\nANSWER")
    print(result.answer)

    report = result.citations

    print("\nCITATION CHECK")
    if not result.called_llm:
        print("Skipped: no evidence retrieved, so no LLM call was made.")
    elif not report.has_citations:
        print("No citations found in the answer." + ("" if result.refused else " Expected at least one."))
    else:
        print(f"Cited: {', '.join(report.cited)}")
        print(f"Supported by retrieved context: {len(report.supported)}/{len(report.cited)}")
        if report.unsupported:
            print(f"UNSUPPORTED (not in context): {', '.join(report.unsupported)}")

    print("\nLATENCY")
    print(f"Retrieval: {retrieval_s:.2f}s")
    print(f"LLM generation: {generation_s:.2f}s")
    print(f"End-to-end: {retrieval_s + generation_s:.2f}s")

    print("\nTOKENS AND COST")
    if not result.called_llm:
        print("No LLM call, so no tokens were spent.")
        return

    usage = result.usage
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    input_price, output_price = pricing()
    cost = usage.cost_usd(input_price, output_price)

    if cost is None:
        print(
            "Cost: not calculated. Set INPUT_PRICE_PER_1M_TOKENS and "
            "OUTPUT_PRICE_PER_1M_TOKENS in .env to estimate it."
        )
    else:
        print(f"Estimated cost: ${cost:.6f}")


def ask(
    question: str,
    retriever: BaseRetriever,
    top_k: int,
    call_llm: bool,
    dedupe_pages: bool = False,
) -> None:
    print("\nQUESTION")
    print(question)

    start = time.perf_counter()
    retrieved = retriever.retrieve(question, top_k=top_k, dedupe_pages=dedupe_pages)
    after_retrieval = time.perf_counter()

    print_retrieval(retriever, question, retrieved)

    if not call_llm:
        print(f"\nLATENCY\nRetrieval: {after_retrieval - start:.2f}s")
        return

    result = generate_answer(question, retrieved)
    end = time.perf_counter()

    print_generation(result, after_retrieval - start, end - after_retrieval)


def main() -> None:
    args = parse_args()
    load_dotenv()

    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    chunks = load_chunks()
    stats = corpus_stats(chunks)

    print("Financial document RAG demo")
    print(
        f"Corpus: {stats['documents']} documents, {stats['pages']} pages, "
        f"{stats['chunks']} chunks "
        f"({stats['split_pages']} pages split into multiple chunks, "
        f"mean {stats['mean_chunk_words']} words per chunk)"
    )

    retriever = build_retriever(args.retriever, chunks, embedding_model)
    call_llm = not args.no_llm

    if args.question:
        ask(args.question, retriever, args.top_k, call_llm, args.dedupe_pages)
        return

    print("Type 'quit' to exit.")

    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        ask(question, retriever, args.top_k, call_llm, args.dedupe_pages)


if __name__ == "__main__":
    main()
