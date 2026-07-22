"""Smoke tests for the /agent endpoint (offline echo provider)."""

from __future__ import annotations


def test_agent_endpoint_returns_result(client) -> None:
    resp = client.post("/agent", json={"message": "what is 2+2?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert isinstance(body["steps"], list)
    assert body["finished"] is True


def test_agent_rejects_empty_message(client) -> None:
    resp = client.post("/agent", json={"message": ""})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "validation_error"
    assert body["error"]
