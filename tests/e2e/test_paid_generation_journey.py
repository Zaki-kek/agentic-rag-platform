"""End-to-end: confirm a payment, then run a generation job to completion."""

from __future__ import annotations


def test_payment_then_generation_job_completes(client) -> None:
    pay = client.post("/payments", json={"amount": 500})
    assert pay.status_code == 200
    payment_id = pay.json()["id"]
    assert client.post(f"/payments/{payment_id}/confirm").json()["status"] == "paid"

    job = client.post("/jobs", json={"data": [10, 20, 30, 40]})
    assert job.status_code == 200
    body = job.json()
    assert body["status"] == "done"
    assert body["progress"] == 1.0
    assert body["completed_stages"] == ["compute", "draft"]
    assert body["context"]["stats"]["mean"] == 25.0

    polled = client.get(f"/jobs/{body['job_id']}")
    assert polled.status_code == 200
    assert polled.json()["progress"] == 1.0
