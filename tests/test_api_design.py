"""API design tests: uniform error shape and /v1 dual-mount versioning."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_app_error_returns_error_and_code(client) -> None:
    resp = client.post(
        "/documents",
        files={"file": ("image.png", b"\x89PNG", "image/png")},
    )
    assert resp.status_code == 415
    body = resp.json()
    assert set(body) == {"error", "code"}
    assert body["code"] == "unsupported_media_type"
    assert body["error"]


def test_validation_error_returns_code(client) -> None:
    resp = client.post("/agent", json={"message": ""})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "validation_error"
    assert body["error"]


def test_v1_health_ok(client) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_v1_dual_mount_keeps_unversioned_paths(client) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    # Both the original and the /v1-prefixed variant are registered.
    assert "/chat" in paths
    assert "/v1/chat" in paths
    assert "/documents" in paths
    assert "/v1/documents" in paths


def test_v1_error_shape_matches(client) -> None:
    resp = client.post(
        "/v1/documents",
        files={"file": ("image.png", b"\x89PNG", "image/png")},
    )
    assert resp.status_code == 415
    body = resp.json()
    assert body["code"] == "unsupported_media_type"


def test_unexpected_error_returns_code() -> None:
    """A 500 from an unhandled exception still yields the uniform {error, code} body."""
    app = create_app(Settings(llm_provider="echo", embedder="hash", vector_store="memory"))

    @app.get("/_boom")
    async def _boom() -> None:  # pragma: no cover - trivial raise
        raise RuntimeError("kaboom")

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/_boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    assert body["error"]


def test_health_unversioned_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
