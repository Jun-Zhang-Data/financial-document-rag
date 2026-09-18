"""Long-lived RAG service used by the HTTP API.

The expensive retriever/model objects are constructed once during application startup,
not once per request. The same service interface supports the transparent in-memory demo
backend and the production PostgreSQL/pgvector backend.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from config import Settings
from documents import corpus_stats, load_chunks
from generation import GenerationResult, generate_answer
from observability import RAG_QUERIES, RAG_QUERY_LATENCY
from retrieval import RetrievalResult, build_retriever

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    question: str
    retrieved: list[RetrievalResult]
    filter_description: str
    filter_relaxed: bool
    answer: GenerationResult | None
    retrieval_ms: float
    generation_ms: float


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend_name = settings.retrieval_backend
        self.retriever: Any = None
        self.stats: dict[str, Any] = {}
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def start(self) -> None:
        if self.settings.retrieval_backend == "memory":
            chunks = load_chunks()
            self.stats = corpus_stats(chunks)
            self.retriever = build_retriever(
                self.settings.retriever_kind,
                chunks,
                self.settings.embedding_model,
            )
        else:
            from postgres_retrieval import PostgresHybridRetriever

            self.retriever = PostgresHybridRetriever(self.settings)
            self.retriever.start()
            self.stats = self.retriever.corpus_stats()

        self._ready = True
        logger.info(
            "RAG service ready",
            extra={"backend": self.backend_name, "retrieved": self.stats.get("chunks", 0)},
        )

    def close(self) -> None:
        close = getattr(self.retriever, "close", None)
        if callable(close):
            close()
        self._ready = False

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        generate: bool = True,
        dedupe_pages: bool | None = None,
    ) -> QueryResult:
        if not self.ready:
            raise RuntimeError("RAG service is not ready")

        top_k = top_k or self.settings.top_k
        dedupe = self.settings.dedupe_pages if dedupe_pages is None else dedupe_pages

        started = time.perf_counter()
        filter_result = self.retriever.filter_for(question)
        retrieved = self.retriever.retrieve(
            question,
            top_k=top_k,
            dedupe_pages=dedupe,
            filter_result=filter_result,
        )
        after_retrieval = time.perf_counter()

        answer: GenerationResult | None = None
        if generate:
            answer = generate_answer(
                question,
                retrieved,
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
            )
        finished = time.perf_counter()

        retrieval_ms = (after_retrieval - started) * 1000
        generation_ms = (finished - after_retrieval) * 1000
        outcome = "ok"
        if answer is not None and answer.refused:
            outcome = "refused"
        elif not retrieved:
            outcome = "no_results"

        RAG_QUERIES.labels(self.backend_name, outcome).inc()
        RAG_QUERY_LATENCY.labels(self.backend_name, str(generate).lower()).observe(finished - started)
        logger.info(
            "RAG query completed",
            extra={
                "backend": self.backend_name,
                "question_length": len(question),
                "retrieved": len(retrieved),
                "latency_ms": round((finished - started) * 1000, 2),
            },
        )

        return QueryResult(
            question=question,
            retrieved=retrieved,
            filter_description=filter_result.describe(),
            filter_relaxed=filter_result.relaxed,
            answer=answer,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )
