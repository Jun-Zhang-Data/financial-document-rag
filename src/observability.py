"""Logging, metrics and optional OpenTelemetry wiring for the API service."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from config import Settings

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")

REQUESTS = Counter(
    "rag_http_requests_total",
    "HTTP requests handled by the RAG API",
    ("method", "path", "status"),
)
LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency for the RAG API",
    ("method", "path"),
)
RAG_QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "End-to-end RAG query latency",
    ("backend", "generate"),
)
RAG_QUERIES = Counter(
    "rag_queries_total",
    "RAG queries by backend and outcome",
    ("backend", "outcome"),
)


class JsonFormatter(logging.Formatter):
    """Small JSON formatter to keep logs machine-readable without another dependency."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(),
        }
        for field in ("backend", "question_length", "retrieved", "latency_ms"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def configure_metrics(app: FastAPI, settings: Settings) -> None:
    if not settings.metrics_enabled:
        return

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable):
        started = time.perf_counter()
        status = 500
        path = request.url.path
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(request.method, path).observe(elapsed)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def configure_request_ids(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = REQUEST_ID.set(request_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            REQUEST_ID.reset(token)


def configure_opentelemetry(app: FastAPI, settings: Settings) -> None:
    """Enable OTLP tracing only when configured and optional packages are installed."""
    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover - exercised in production images
        logging.getLogger(__name__).warning("OpenTelemetry packages not installed: %s", exc)
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
