"""Tests for the evaluation harness itself.

The measurement code deserves tests for the same reason the retrieval code does: two of
the numbers this project reports were once wrong because of defects in here, not in the
retrievers.
"""

from __future__ import annotations

import argparse

import pytest

from documents import DATA_DIR, load_chunks
from evaluate import (
    evaluate_retrieval,
    load_questions,
    positive_int,
    shared_retriever,
    warm_up,
)
from retrieval import BM25Retriever, HybridRetriever


@pytest.fixture(scope="module")
def corpus():
    return load_chunks(DATA_DIR)


@pytest.mark.parametrize("value", ["0", "-1", "-2"])
def test_positive_int_rejects_non_positive_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="must be at least 1"):
        positive_int(value)


def test_positive_int_accepts_positive_values():
    assert positive_int("4") == 4


def test_shared_retriever_reuses_an_existing_instance(corpus):
    built = {}
    first = shared_retriever("bm25", built, corpus, "unused-model")
    second = shared_retriever("bm25", built, corpus, "unused-model")

    assert first is second


def test_hybrid_reuses_the_already_built_components(corpus):
    """Regression test: the hybrid must not construct a second embedding model.

    Two SentenceTransformers in one process encode the corpus twice and contend for the
    same CPU threads, which is what made the reported latencies swing between runs. A
    stand-in for the dense retriever is enough to prove reuse without downloading a model.
    """
    dense_stub = BM25Retriever(corpus)
    dense_stub.name = "dense"
    bm25 = BM25Retriever(corpus)

    built = {"dense": dense_stub, "bm25": bm25}
    hybrid = shared_retriever("hybrid", built, corpus, "unused-model")

    assert isinstance(hybrid, HybridRetriever)
    assert hybrid.retrievers[0] is dense_stub
    assert hybrid.retrievers[1] is bm25


def test_warm_up_stops_early_when_latency_is_already_flat(corpus):
    """BM25 needs no warm-up, so the adaptive loop must not run all 30 calls."""
    calls = warm_up(BM25Retriever(corpus), max_calls=30)

    assert 5 <= calls < 30


def test_bm25_metrics_match_the_published_table(corpus):
    """Pin the numbers the README and design notes report.

    The published table is a claim about this corpus and this evaluation set. If a change
    moves it, that should be a failing test and a deliberate documentation update rather
    than a quiet drift.
    """
    questions = load_questions()
    bm25 = BM25Retriever(corpus)

    plain = evaluate_retrieval(bm25, questions, top_k=4, verbose=False)
    deduped = evaluate_retrieval(bm25, questions, top_k=4, verbose=False, dedupe_pages=True)

    assert (plain["recall"], plain["coverage"], plain["mrr"]) == pytest.approx(
        (0.80, 0.80, 0.60), abs=0.005
    )
    assert (deduped["recall"], deduped["coverage"], deduped["mrr"]) == pytest.approx(
        (0.87, 0.87, 0.63), abs=0.005
    )
