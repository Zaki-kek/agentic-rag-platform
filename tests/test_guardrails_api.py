"""Tests for the /guardrails endpoints."""

from __future__ import annotations


def test_redact_endpoint_masks_email(client) -> None:
    resp = client.post("/guardrails/redact", json={"text": "reach me at alice@example.com please"})
    assert resp.status_code == 200
    redacted = resp.json()["redacted"]
    assert "alice@example.com" not in redacted


def test_citations_endpoint_flags_out_of_range(client) -> None:
    resp = client.post("/guardrails/citations", json={"answer": "see [1] and [5]", "num_sources": 3})
    assert resp.status_code == 200
    assert resp.json()["problems"]  # [5] is out of range


def test_citations_endpoint_ok_when_valid(client) -> None:
    resp = client.post("/guardrails/citations", json={"answer": "see [1] and [2]", "num_sources": 3})
    assert resp.status_code == 200
    assert resp.json()["problems"] == []
