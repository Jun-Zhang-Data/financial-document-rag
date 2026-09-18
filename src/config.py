"""Application configuration loaded from environment variables.

The CLI can keep using its legacy flags, while the API and deployment paths use a
single validated settings object. Secrets are never read from committed files in
production; Pydantic Settings reads them from the process environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from retrieval import DEFAULT_EMBEDDING_MODEL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    app_name: str = Field(default="financial-document-rag", alias="APP_NAME")
    app_env: Literal["local", "dev", "staging", "prod", "test"] = Field(
        default="local", alias="APP_ENV"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    retrieval_backend: Literal["memory", "postgres"] = Field(
        default="memory", alias="RETRIEVAL_BACKEND"
    )
    retriever_kind: Literal["bm25", "dense", "hybrid"] = Field(
        default="bm25", alias="RAG_RETRIEVER"
    )
    top_k: int = Field(default=4, ge=1, le=20, alias="RAG_TOP_K")
    dedupe_pages: bool = Field(default=True, alias="RAG_DEDUPE_PAGES")

    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, ge=1, le=4096, alias="EMBEDDING_DIMENSION")

    database_url: str = Field(
        default="postgresql://rag:rag@postgres:5432/rag", alias="DATABASE_URL"
    )
    db_pool_min_size: int = Field(default=1, ge=1, le=20, alias="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=10, ge=1, le=100, alias="DB_POOL_MAX_SIZE")
    postgres_candidate_k: int = Field(default=50, ge=4, le=1000, alias="POSTGRES_CANDIDATE_K")
    rrf_k: int = Field(default=60, ge=1, le=1000, alias="RRF_K")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="", alias="OPENAI_MODEL")
    input_price_per_1m_tokens: float | None = Field(
        default=None, alias="INPUT_PRICE_PER_1M_TOKENS"
    )
    output_price_per_1m_tokens: float | None = Field(
        default=None, alias="OUTPUT_PRICE_PER_1M_TOKENS"
    )

    service_api_key: str = Field(default="", alias="SERVICE_API_KEY")
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")
    otel_service_name: str = Field(default="financial-document-rag", alias="OTEL_SERVICE_NAME")
    otel_exporter_otlp_endpoint: str = Field(default="", alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper().strip()

    @field_validator("db_pool_max_size")
    @classmethod
    def pool_max_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("DB_POOL_MAX_SIZE must be positive")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
