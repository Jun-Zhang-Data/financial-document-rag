"""Persistent hybrid retrieval using PostgreSQL full-text search + pgvector.

This backend keeps the original project's transparent RRF strategy but moves indexing
and nearest-neighbour search into PostgreSQL. It is loaded lazily so the core unit tests
and BM25 demo still run without database drivers or a database server.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from config import Settings
from documents import Chunk
from retrieval import QueryFilters, RetrievalResult, parse_filters


@dataclass
class DatabaseFilterResult:
    filters: QueryFilters
    applied: QueryFilters
    relaxed: bool
    candidate_count: int

    def describe(self) -> str:
        if not self.relaxed:
            return self.applied.describe()
        return f"{self.applied.describe()} (relaxed from: {self.filters.describe()})"


def _filter_attempts(filters: QueryFilters) -> list[QueryFilters]:
    attempts = [filters]
    if filters.companies and filters.years:
        attempts.append(QueryFilters(companies=filters.companies))
        attempts.append(QueryFilters(years=filters.years))
    if QueryFilters() not in attempts:
        attempts.append(QueryFilters())
    return attempts


def _where_clause(filters: QueryFilters) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if filters.companies:
        clauses.append("company = ANY(%s)")
        params.append(list(filters.companies))
    if filters.years:
        clauses.append("year = ANY(%s)")
        params.append(list(filters.years))
    return (" AND ".join(clauses) if clauses else "TRUE", params)


def ensure_schema(connection: Any, embedding_dimension: int) -> None:
    """Create the pgvector-backed schema and indexes if they do not already exist."""
    dimension = int(embedding_dimension)
    if not 1 <= dimension <= 4096:
        raise ValueError("embedding_dimension must be between 1 and 4096")

    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS document_chunks (
        id BIGSERIAL PRIMARY KEY,
        company TEXT NOT NULL,
        year TEXT NOT NULL,
        page TEXT NOT NULL,
        source_file TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding vector({dimension}) NOT NULL,
        textsearch tsvector GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(content, ''))
        ) STORED,
        UNIQUE (source_file, page, chunk_index)
    );

    CREATE INDEX IF NOT EXISTS document_chunks_company_year_idx
        ON document_chunks (company, year);
    CREATE INDEX IF NOT EXISTS document_chunks_textsearch_idx
        ON document_chunks USING GIN (textsearch);
    CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
    """
    with connection.cursor() as cur:
        cur.execute(ddl)
    connection.commit()


class PostgresHybridRetriever:
    name = "postgres-hybrid(pgvector+fts)"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool: Any = None
        self.model: Any = None
        self.companies: list[str] = []
        self._query_cache: OrderedDict[str, str] = OrderedDict()
        self._query_cache_size = 512

    def start(self) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - depends on production extra
            raise RuntimeError(
                "PostgreSQL backend requires psycopg pool support. Install the production dependencies."
            ) from exc

        from sentence_transformers import SentenceTransformer

        self.pool = ConnectionPool(
            conninfo=self.settings.database_url,
            min_size=self.settings.db_pool_min_size,
            max_size=self.settings.db_pool_max_size,
            open=True,
        )
        with self.pool.connection() as conn:
            ensure_schema(conn, self.settings.embedding_dimension)
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT company FROM document_chunks ORDER BY company")
                self.companies = [row[0] for row in cur.fetchall()]

        self.model = SentenceTransformer(self.settings.embedding_model)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()

    def corpus_stats(self) -> dict[str, Any]:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    count(DISTINCT source_file) AS documents,
                    count(DISTINCT (company, year, page)) AS pages,
                    count(*) AS chunks,
                    count(DISTINCT (company, year, page)) FILTER (WHERE chunk_index > 0) AS split_pages,
                    coalesce(round(avg(array_length(regexp_split_to_array(content, '\\s+'), 1)), 1), 0),
                    coalesce(max(array_length(regexp_split_to_array(content, '\\s+'), 1)), 0)
                FROM document_chunks
                """
            )
            row = cur.fetchone()
        return {
            "documents": int(row[0]),
            "pages": int(row[1]),
            "chunks": int(row[2]),
            "split_pages": int(row[3]),
            "mean_chunk_words": float(row[4]),
            "max_chunk_words": int(row[5]),
        }

    def filter_for(self, query: str) -> DatabaseFilterResult:
        filters = parse_filters(query, self.companies)
        with self.pool.connection() as conn, conn.cursor() as cur:
            for attempt in _filter_attempts(filters):
                where, params = _where_clause(attempt)
                cur.execute(f"SELECT count(*) FROM document_chunks WHERE {where}", params)
                count = int(cur.fetchone()[0])
                if count:
                    return DatabaseFilterResult(
                        filters=filters,
                        applied=attempt,
                        relaxed=attempt != filters,
                        candidate_count=count,
                    )
        return DatabaseFilterResult(filters, QueryFilters(), True, 0)

    def _embed_query(self, query: str) -> str:
        """Embed a query string, caching recent results.

        The cache is a bounded per-instance dict rather than ``functools.lru_cache`` on the
        method: a decorator here would key on ``self`` and keep the retriever (and its
        model) alive for as long as the cache lives.
        """
        cached = self._query_cache.get(query)
        if cached is not None:
            self._query_cache.move_to_end(query)
            return cached

        vector = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        if len(vector) != self.settings.embedding_dimension:
            raise RuntimeError(
                f"embedding model produced {len(vector)} dimensions but "
                f"EMBEDDING_DIMENSION={self.settings.embedding_dimension}"
            )
        literal = "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"

        self._query_cache[query] = literal
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)

        return literal

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        dedupe_pages: bool = False,
        filter_result: DatabaseFilterResult | None = None,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        filter_result = filter_result or self.filter_for(query)
        if not filter_result.candidate_count:
            return []

        where, filter_params = _where_clause(filter_result.applied)
        embedding = self._embed_query(query)
        candidate_k = max(top_k * 4, self.settings.postgres_candidate_k)
        output_limit = min(candidate_k, top_k * 6 if dedupe_pages else top_k)

        sql = f"""
        WITH filtered AS (
            SELECT id, company, year, page, source_file, chunk_index, content, embedding, textsearch
            FROM document_chunks
            WHERE {where}
        ),
        query_text AS (
            SELECT websearch_to_tsquery('english', %s) AS q
        ),
        dense AS (
            SELECT id,
                   row_number() OVER (ORDER BY embedding <=> %s::vector, id) AS rank
            FROM filtered
            ORDER BY embedding <=> %s::vector, id
            LIMIT %s
        ),
        lexical AS (
            SELECT f.id,
                   row_number() OVER (
                       ORDER BY ts_rank_cd(f.textsearch, q.q) DESC, f.id
                   ) AS rank
            FROM filtered f
            CROSS JOIN query_text q
            WHERE f.textsearch @@ q.q
            ORDER BY ts_rank_cd(f.textsearch, q.q) DESC, f.id
            LIMIT %s
        ),
        combined AS (
            SELECT id, rank FROM dense
            UNION ALL
            SELECT id, rank FROM lexical
        ),
        fused AS (
            SELECT id, sum(1.0 / (%s + rank)) AS score
            FROM combined
            GROUP BY id
        )
        SELECT f.company, f.year, f.page, f.source_file, f.chunk_index, f.content, fused.score
        FROM fused
        JOIN filtered f USING (id)
        ORDER BY fused.score DESC, f.id
        LIMIT %s
        """
        params = [
            *filter_params,
            query,
            embedding,
            embedding,
            candidate_k,
            candidate_k,
            self.settings.rrf_k,
            output_limit,
        ]

        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[RetrievalResult] = []
        seen_pages: set[tuple[str, str, str]] = set()
        for row in rows:
            chunk = Chunk(
                company=row[0],
                year=row[1],
                page=row[2],
                source_file=row[3],
                chunk_index=int(row[4]),
                text=row[5],
            )
            if dedupe_pages and chunk.source_key in seen_pages:
                continue
            seen_pages.add(chunk.source_key)
            results.append(
                RetrievalResult(chunk=chunk, score=float(row[6]), rank=len(results) + 1)
            )
            if len(results) >= top_k:
                break
        return results
