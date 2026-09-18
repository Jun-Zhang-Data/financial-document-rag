import os

import pytest

from config import Settings
from postgres_retrieval import PostgresHybridRetriever

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="requires the CI pgvector service",
)
def test_pgvector_hybrid_retrieval_returns_expected_financial_page():
    settings = Settings(
        retrieval_backend="postgres",
        database_url=os.environ["DATABASE_URL"],
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
    )
    retriever = PostgresHybridRetriever(settings)
    retriever.start()
    try:
        results = retriever.retrieve(
            "Northstar 2024 adjusted EBITDA margin declined freight energy",
            top_k=4,
            dedupe_pages=True,
        )
        assert results
        assert all(r.chunk.company == "Northstar Industrial" for r in results)
        assert all(r.chunk.year == "2024" for r in results)
        assert any(r.chunk.page == "18" for r in results)
    finally:
        retriever.close()
