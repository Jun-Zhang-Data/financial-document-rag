"""Production HTTP API for the financial-document RAG service."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from config import Settings, get_settings
from observability import (
    configure_logging,
    configure_metrics,
    configure_opentelemetry,
    configure_request_ids,
)
from service import RAGService


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    generate: bool = True
    dedupe_pages: bool | None = None


class SourceResponse(BaseModel):
    rank: int
    score: float
    company: str
    year: str
    page: str
    citation: str
    text: str


class UsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class QueryResponse(BaseModel):
    question: str
    answer: str | None
    refused: bool | None
    model: str | None
    citations_valid: bool | None
    sources: list[SourceResponse]
    filter: str
    filter_relaxed: bool
    retrieval_ms: float
    generation_ms: float
    usage: UsageResponse | None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    service = RAGService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        app.state.rag = service
        yield
        service.close()

    app = FastAPI(
        title="Financial Document RAG API",
        version="1.0.0",
        docs_url="/docs" if settings.app_env != "prod" else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rag = service

    configure_request_ids(app)
    configure_metrics(app, settings)
    configure_opentelemetry(app, settings)

    def require_api_key(
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = settings.service_api_key
        if not expected:
            return
        # Constant-time comparison: a timing-distinguishable check on a shared secret is
        # cheap to fix and awkward to explain later.
        if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")

    @app.get("/health/live", tags=["health"])
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request) -> dict[str, Any]:
        rag: RAGService = request.app.state.rag
        if not rag.ready:
            raise HTTPException(status_code=503, detail="RAG service is not ready")
        return {"status": "ok", "backend": rag.backend_name, "corpus": rag.stats}

    @app.get("/v1/info", dependencies=[Depends(require_api_key)], tags=["rag"])
    def info(request: Request) -> dict[str, Any]:
        rag: RAGService = request.app.state.rag
        return {
            "name": settings.app_name,
            "environment": settings.app_env,
            "backend": rag.backend_name,
            "retriever": getattr(rag.retriever, "name", settings.retriever_kind),
            "corpus": rag.stats,
        }

    @app.post(
        "/v1/query",
        response_model=QueryResponse,
        dependencies=[Depends(require_api_key)],
        tags=["rag"],
    )
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        rag: RAGService = request.app.state.rag
        try:
            result = rag.query(
                payload.question,
                top_k=payload.top_k,
                generate=payload.generate,
                dedupe_pages=payload.dedupe_pages,
            )
        except RuntimeError as exc:
            # Missing LLM credentials/configuration is a dependency/configuration problem,
            # not a malformed client request.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        sources = [
            SourceResponse(
                rank=item.rank,
                score=item.score,
                company=item.chunk.company,
                year=item.chunk.year,
                page=item.chunk.page,
                citation=item.chunk.citation,
                text=item.chunk.text,
            )
            for item in result.retrieved
        ]

        answer = result.answer
        return QueryResponse(
            question=result.question,
            answer=answer.answer if answer else None,
            refused=answer.refused if answer else None,
            model=answer.model if answer else None,
            citations_valid=answer.citations.is_valid if answer else None,
            sources=sources,
            filter=result.filter_description,
            filter_relaxed=result.filter_relaxed,
            retrieval_ms=round(result.retrieval_ms, 3),
            generation_ms=round(result.generation_ms, 3),
            usage=(
                UsageResponse(
                    input_tokens=answer.usage.input_tokens,
                    output_tokens=answer.usage.output_tokens,
                    total_tokens=answer.usage.total_tokens,
                )
                if answer
                else None
            ),
        )

    return app


app = create_app()
