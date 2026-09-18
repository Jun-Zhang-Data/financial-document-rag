"""Data-integrity tests for the evaluation set.

A gold set that points at pages which do not exist would silently deflate every metric, so
the ground truth is checked against the corpus.
"""

import pytest

from documents import DATA_DIR, load_chunks
from evaluate import expected_keys, load_questions


@pytest.fixture(scope="module")
def source_keys():
    return {chunk.source_key for chunk in load_chunks(DATA_DIR)}


@pytest.fixture(scope="module")
def questions():
    return load_questions()


def test_questions_have_the_required_fields(questions):
    for question in questions:
        assert question["question"].strip()
        assert question["match"] in {"any", "all"}
        assert isinstance(question["answerable"], bool)
        assert isinstance(question["expected_sources"], list)


def test_every_expected_source_exists_in_the_corpus(questions, source_keys):
    missing = [
        (question["question"], key)
        for question in questions
        for key in expected_keys(question)
        if key not in source_keys
    ]

    assert missing == []


def test_answerable_questions_have_at_least_one_expected_source(questions):
    for question in questions:
        if question["answerable"]:
            assert question["expected_sources"], question["question"]


def test_unanswerable_questions_have_no_expected_source(questions):
    for question in questions:
        if not question["answerable"]:
            assert question["expected_sources"] == [], question["question"]


def test_set_is_large_enough_to_be_informative(questions):
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    multi_source = [q for q in questions if len(q["expected_sources"]) > 1]

    assert len(answerable) >= 12
    assert len(unanswerable) >= 3
    assert len(multi_source) >= 3


def test_questions_are_unique(questions):
    texts = [q["question"] for q in questions]

    assert len(texts) == len(set(texts))


def test_top_k_cannot_trivially_cover_the_candidate_pool(questions):
    """Guard against trivially small candidate pools that make Recall@k meaningless.
    """
    from retrieval import BM25Retriever

    retriever = BM25Retriever(load_chunks(DATA_DIR))
    top_k = 4

    for question in questions:
        if not question["answerable"]:
            continue
        pool = retriever.filter_for(question["question"]).candidate_ids
        assert len(pool) > 2 * top_k, (question["question"], len(pool))
