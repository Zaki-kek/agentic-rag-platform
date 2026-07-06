"""End-to-end API tests via TestClient (offline stack)."""

from __future__ import annotations


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "echo"


def test_ingest_then_chat_returns_citations(client) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("facts.txt", b"The Eiffel Tower is located in Paris, France.", "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json()["chunks"] >= 1

    chat = client.post("/chat", json={"message": "Where is the Eiffel Tower?"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["provider"] == "echo"
    assert body["citations"], "expected at least one citation"
    assert "Eiffel Tower" in body["citations"][0]["preview"]


def test_chat_rejects_empty_message(client) -> None:
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422
