"""End-to-end tests for the generation job API (offline echo provider)."""

from __future__ import annotations


def test_create_and_get_job(client) -> None:
    resp = client.post("/jobs", json={"data": [10, 20, 30]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["progress"] == 1.0
    assert body["completed_stages"] == ["compute", "draft"]
    assert body["context"]["stats"]["mean"] == 20.0
    assert body["context"]["draft"]  # the (echoed) narrative

    job_id = body["job_id"]
    got = client.get(f"/jobs/{job_id}")
    assert got.status_code == 200
    assert got.json()["job_id"] == job_id


def test_get_missing_job_returns_404(client) -> None:
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_job_rejects_empty_data(client) -> None:
    assert client.post("/jobs", json={"data": []}).status_code == 422


def test_idempotency_key_returns_same_job(client) -> None:
    payload = {"data": [1, 2, 3], "idempotency_key": "abc-123"}

    first = client.post("/jobs", json=payload)
    second = client.post("/jobs", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    # Same key -> same job, no duplicate created.
    assert first.json()["job_id"] == second.json()["job_id"]
    assert first.json() == second.json()


def test_different_idempotency_keys_create_distinct_jobs(client) -> None:
    a = client.post("/jobs", json={"data": [1], "idempotency_key": "k-a"})
    b = client.post("/jobs", json={"data": [1], "idempotency_key": "k-b"})

    assert a.json()["job_id"] != b.json()["job_id"]


def test_no_idempotency_key_creates_new_job_each_time(client) -> None:
    a = client.post("/jobs", json={"data": [1, 2]})
    b = client.post("/jobs", json={"data": [1, 2]})

    assert a.json()["job_id"] != b.json()["job_id"]
