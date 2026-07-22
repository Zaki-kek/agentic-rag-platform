"""Vendor-neutral tracing package (no-op by default, optional Langfuse).

Importing this package exposes the tracer abstraction and factory. The default
``build_tracer()`` returns a zero-cost no-op tracer, so callers can instrument
code unconditionally without pulling in any observability dependency.
"""

from app.observability.metrics import REGISTRY, Counter, Registry, http_requests_total
from app.observability.tracer import (
    InMemoryTracer,
    LangfuseTracer,
    NoOpTracer,
    Span,
    SpanHandle,
    Tracer,
    build_tracer,
)

__all__ = [
    "Span",
    "SpanHandle",
    "Tracer",
    "NoOpTracer",
    "InMemoryTracer",
    "LangfuseTracer",
    "build_tracer",
    "Counter",
    "Registry",
    "REGISTRY",
    "http_requests_total",
]
