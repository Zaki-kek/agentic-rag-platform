"""End-to-end: ingest a document, then chat and stream grounded over it."""

from __future__ import annotations


def test_ingest_then_chat_then_stream(client) -> None:
    corpus = b"The Eiffel Tower is in Paris, France. The Louvre is also in Paris."

    up = client.post("/documents", files={"file": ("kb.txt", corpus, "text/plain")})
    assert up.status_code == 200
    assert up.json()["chunks"] >= 1

    chat = client.post("/chat", json={"message": "Where is the Eiffel Tower?"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["citations"]
    assert any("Eiffel" in c["preview"] for c in body["citations"])

    stream = client.post("/chat/stream", json={"message": "Where is the Eiffel Tower?"})
    assert stream.status_code == 200
    assert "text/event-stream" in stream.headers["content-type"]
    assert "event: token" in stream.text
    assert "event: done" in stream.text
