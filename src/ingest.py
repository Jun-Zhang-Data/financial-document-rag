"""Ingest the local document corpus into PostgreSQL/pgvector.

Usage:
    PYTHONPATH=src python src/ingest.py --replace
"""

from __future__ import annotations

import argparse

from config import get_settings
from documents import load_chunks
from postgres_retrieval import ensure_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest financial documents into pgvector")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete existing chunks before loading the local corpus",
    )
    return parser.parse_args()


def vector_literal(values) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def main() -> None:
    args = parse_args()
    settings = get_settings()

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install production dependencies to run PostgreSQL ingestion") from exc

    from sentence_transformers import SentenceTransformer

    chunks = load_chunks()
    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if embeddings.shape[1] != settings.embedding_dimension:
        raise RuntimeError(
            f"embedding model produced {embeddings.shape[1]} dimensions but "
            f"EMBEDDING_DIMENSION={settings.embedding_dimension}"
        )

    with psycopg.connect(settings.database_url) as conn:
        ensure_schema(conn, settings.embedding_dimension)
        with conn.cursor() as cur:
            if args.replace:
                # Initial bulk loads are faster without maintaining the HNSW graph row by row.
                cur.execute("DROP INDEX IF EXISTS document_chunks_embedding_hnsw_idx")
                cur.execute("TRUNCATE document_chunks RESTART IDENTITY")

            rows = [
                (
                    chunk.company,
                    chunk.year,
                    chunk.page,
                    chunk.source_file,
                    chunk.chunk_index,
                    chunk.text,
                    vector_literal(embedding),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
            cur.executemany(
                """
                INSERT INTO document_chunks
                    (company, year, page, source_file, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (source_file, page, chunk_index) DO UPDATE SET
                    company = EXCLUDED.company,
                    year = EXCLUDED.year,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
        conn.commit()

    print(f"Ingested {len(chunks)} chunks into PostgreSQL/pgvector.")


if __name__ == "__main__":
    main()
