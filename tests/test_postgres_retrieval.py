import pytest

from config import Settings
from postgres_retrieval import PostgresHybridRetriever, _filter_attempts, _where_clause
from retrieval import QueryFilters


def test_database_filter_relaxation_matches_memory_order():
    original = QueryFilters(companies=("Northstar Industrial",), years=("2024",))
    assert _filter_attempts(original) == [
        original,
        QueryFilters(companies=("Northstar Industrial",)),
        QueryFilters(years=("2024",)),
        QueryFilters(),
    ]


def test_database_where_clause_is_parameterized():
    where, params = _where_clause(
        QueryFilters(companies=("Northstar Industrial", "BluePeak Software"), years=("2024",))
    )
    assert where == "company = ANY(%s) AND year = ANY(%s)"
    assert params == [["Northstar Industrial", "BluePeak Software"], ["2024"]]


class _StubModel:
    """Stands in for a SentenceTransformer, counting how often it is asked to encode."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.calls = 0

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        self.calls += 1
        return [[0.1] * self.dimension for _ in texts]


@pytest.fixture
def retriever():
    settings = Settings(embedding_dimension=4)
    instance = PostgresHybridRetriever(settings)
    instance.model = _StubModel(settings.embedding_dimension)
    return instance


def test_query_embeddings_are_cached(retriever):
    first = retriever._embed_query("EBITDA margin")
    second = retriever._embed_query("EBITDA margin")

    assert first == second
    assert retriever.model.calls == 1


def test_query_embedding_cache_is_bounded(retriever):
    retriever._query_cache_size = 3

    for i in range(5):
        retriever._embed_query(f"question {i}")

    assert len(retriever._query_cache) == 3
    # Least recently used entries are evicted first.
    assert list(retriever._query_cache) == ["question 2", "question 3", "question 4"]


def test_embedding_dimension_mismatch_is_reported(retriever):
    retriever.model = _StubModel(dimension=7)

    with pytest.raises(RuntimeError, match="produced 7 dimensions"):
        retriever._embed_query("EBITDA margin")
