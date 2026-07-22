"""Tests for the /metrics endpoint and the in-memory metrics registry."""

from __future__ import annotations

import re

from app.observability.metrics import Counter, Registry


def test_metrics_endpoint_ok_and_nonempty(client) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.text.strip()
    assert resp.headers["content-type"].startswith("text/")


def test_metrics_expose_http_requests_total(client) -> None:
    body = client.get("/metrics").text
    assert "# TYPE http_requests_total counter" in body
    assert re.search(r"^http_requests_total \d", body, re.MULTILINE)


def _read_counter(client) -> float:
    body = client.get("/metrics").text
    match = re.search(r"^http_requests_total (\S+)$", body, re.MULTILINE)
    assert match is not None, body
    return float(match.group(1))


def test_counter_grows_monotonically_across_requests(client) -> None:
    # Prime the counter, then measure the delta over N extra requests.
    baseline = _read_counter(client)
    extra = 3
    for _ in range(extra):
        client.get("/health")
    after = _read_counter(client)
    # Each /metrics read is itself a request, so the delta is at least `extra`.
    assert after >= baseline + extra


def test_counter_unit_semantics() -> None:
    counter = Counter("demo_total", "demo")
    assert counter.value == 0.0
    counter.inc()
    counter.inc(2)
    assert counter.value == 3.0


def test_registry_render_prometheus_format() -> None:
    registry = Registry()
    registry.counter("widgets_total", "widgets made").inc(5)
    text = registry.render()
    assert "# TYPE widgets_total counter" in text
    assert "widgets_total 5" in text
    assert text.endswith("\n")


def test_registry_counter_is_idempotent() -> None:
    registry = Registry()
    first = registry.counter("shared_total")
    second = registry.counter("shared_total")
    assert first is second
