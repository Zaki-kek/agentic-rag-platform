"""Tests for the payment stub API."""

from __future__ import annotations


def test_payment_lifecycle(client) -> None:
    created = client.post("/payments", json={"amount": 500})
    assert created.status_code == 200
    payment = created.json()
    assert payment["status"] == "pending"
    assert payment["confirmation_url"]

    payment_id = payment["id"]
    confirmed = client.post(f"/payments/{payment_id}/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "paid"

    assert client.get(f"/payments/{payment_id}").json()["status"] == "paid"


def test_confirm_missing_payment_returns_404(client) -> None:
    assert client.post("/payments/nope/confirm").status_code == 404


def test_create_payment_rejects_nonpositive_amount(client) -> None:
    assert client.post("/payments", json={"amount": 0}).status_code == 422
