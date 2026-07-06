"""Tests for the vendor-neutral tracing package (offline, no langfuse needed)."""

from __future__ import annotations

import pytest

from app.observability import (
    InMemoryTracer,
    LangfuseTracer,
    NoOpTracer,
    Span,
    Tracer,
    build_tracer,
)


def test_noop_span_is_a_context_manager_and_records_nothing() -> None:
    tracer = NoOpTracer()
    with tracer.span("work", foo="bar") as handle:
        handle.set(extra=1)  # must be callable and harmless
    # NoOp keeps no state; only public surface is the (empty) protocol.
    assert not hasattr(tracer, "spans")
    assert isinstance(tracer, Tracer)


def test_inmemory_records_span_name_and_initial_attributes() -> None:
    tracer = InMemoryTracer()
    with tracer.span("retrieve", k=4):
        pass

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert isinstance(span, Span)
    assert span.name == "retrieve"
    assert span.attributes == {"k": 4}
    assert span.duration_ms is not None
    assert span.duration_ms >= 0.0


def test_inmemory_records_attributes_set_via_handle() -> None:
    tracer = InMemoryTracer()
    with tracer.span("generate", model="echo") as handle:
        handle.set(tokens=12)
        handle.set(tokens=13, cached=True)  # later .set overwrites/merges

    span = tracer.spans[0]
    assert span.attributes == {"model": "echo", "tokens": 13, "cached": True}


def test_handle_set_returns_self_for_chaining() -> None:
    tracer = InMemoryTracer()
    with tracer.span("chain") as handle:
        assert handle.set(a=1).set(b=2) is handle

    assert tracer.spans[0].attributes == {"a": 1, "b": 2}


def test_nested_spans_are_both_recorded_in_completion_order() -> None:
    tracer = InMemoryTracer()
    with tracer.span("outer", level=0):
        with tracer.span("inner", level=1) as inner:
            inner.set(done=True)

    names = [s.name for s in tracer.spans]
    # The inner span completes (and is recorded) before the outer one.
    assert names == ["inner", "outer"]
    inner_span = tracer.spans[0]
    assert inner_span.attributes == {"level": 1, "done": True}


def test_inmemory_records_span_even_when_block_raises() -> None:
    tracer = InMemoryTracer()
    with pytest.raises(ValueError):
        with tracer.span("boom"):
            raise ValueError("kaboom")

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "boom"
    assert tracer.spans[0].duration_ms is not None


def test_build_tracer_returns_the_right_types() -> None:
    assert isinstance(build_tracer(), NoOpTracer)
    assert isinstance(build_tracer("none"), NoOpTracer)
    assert isinstance(build_tracer("memory"), InMemoryTracer)
    assert isinstance(build_tracer("langfuse"), LangfuseTracer)


def test_build_tracer_unknown_name_falls_back_to_noop() -> None:
    assert isinstance(build_tracer("does-not-exist"), NoOpTracer)


def test_langfuse_tracer_degrades_to_noop_without_dependency() -> None:
    # langfuse is not installed in the test environment, so the client is None
    # and spans are silently dropped rather than crashing.
    tracer = LangfuseTracer()
    assert tracer._client is None
    with tracer.span("call", provider="anthropic") as handle:
        handle.set(ok=True)  # must not raise
    assert isinstance(tracer, Tracer)
