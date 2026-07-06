"""Smoke tests: the app boots and exposes the expected surface."""

from __future__ import annotations

EXPECTED_ROUTES = {
    "/health",
    "/documents",
    "/chat",
    "/chat/stream",
    "/jobs",
    "/jobs/{job_id}",
    "/payments",
    "/payments/{payment_id}",
    "/payments/{payment_id}/confirm",
    "/agent",
    "/guardrails/redact",
    "/guardrails/citations",
    "/telegram/message",
}


def test_all_expected_routes_registered(client) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    missing = EXPECTED_ROUTES - paths
    assert not missing, f"missing routes: {missing}"


def test_openapi_schema_generates(client) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"]


def test_health_reports_active_config(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "echo"
