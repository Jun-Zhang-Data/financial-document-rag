from fastapi.testclient import TestClient

from api import create_app
from config import Settings


def test_health_and_retrieval_only_query():
    settings = Settings(
        app_env="test",
        retrieval_backend="memory",
        retriever_kind="bm25",
        metrics_enabled=False,
        service_api_key="",
    )

    with TestClient(create_app(settings)) as client:
        live = client.get("/health/live")
        assert live.status_code == 200
        assert live.json() == {"status": "ok"}

        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["corpus"]["chunks"] == 59

        response = client.post(
            "/v1/query",
            json={
                "question": "Why did Northstar's EBITDA margin fall in 2024?",
                "generate": False,
                "top_k": 4,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] is None
        assert body["sources"]
        assert all(source["company"] == "Northstar Industrial" for source in body["sources"])
        assert body["filter_relaxed"] is False
        assert response.headers["x-request-id"]


def test_api_key_is_enforced_when_configured():
    settings = Settings(
        app_env="test",
        retrieval_backend="memory",
        retriever_kind="bm25",
        metrics_enabled=False,
        service_api_key="secret",
    )

    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/info").status_code == 401
        assert client.get("/v1/info", headers={"x-api-key": "secret"}).status_code == 200
