"""Tests for the /telegram/message simulation endpoint."""

from __future__ import annotations


def test_telegram_message_returns_reply(client) -> None:
    client.post(
        "/documents",
        files={"file": ("facts.txt", b"Paris is the capital of France.", "text/plain")},
    )
    resp = client.post("/telegram/message", json={"text": "What is the capital of France?"})
    assert resp.status_code == 200
    assert resp.json()["reply"]


def test_telegram_message_rejects_empty(client) -> None:
    assert client.post("/telegram/message", json={"text": ""}).status_code == 422
