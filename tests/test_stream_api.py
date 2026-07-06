"""Tests for the /chat/stream SSE endpoint."""

from __future__ import annotations


def test_chat_stream_emits_sse_events(client) -> None:
    client.post(
        "/documents",
        files={"file": ("facts.txt", b"The Eiffel Tower is in Paris, France.", "text/plain")},
    )
    resp = client.post("/chat/stream", json={"message": "Where is the Eiffel Tower?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert "event: token" in body
    assert "event: done" in body
