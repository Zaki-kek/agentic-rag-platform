"""Vendor-neutral tracing with optional Langfuse, no-op by default.

Use a tracer to time and annotate spans of work without coupling the codebase
to any observability vendor. Build one with ``build_tracer(name)``; the default
``"none"`` is a zero-cost no-op, ``"memory"`` records spans in process for
tests/debug, and ``"langfuse"`` reports to Langfuse when it (and its keys) are
present, degrading to no-op otherwise.

Example:
    >>> tracer = build_tracer("memory")
    >>> with tracer.span("retrieve", k=4) as handle:
    ...     handle.set(hits=2)
    >>> tracer.spans[0].name
    'retrieve'
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.core import get_logger

logger = get_logger(__name__)

__all__ = [
    "Span",
    "SpanHandle",
    "Tracer",
    "NoOpTracer",
    "InMemoryTracer",
    "LangfuseTracer",
    "build_tracer",
]


@dataclass
class Span:
    """A single finished (or in-flight) unit of traced work.

    Attributes:
        name: Human-readable span name, e.g. ``"retrieve"``.
        attributes: Arbitrary key/value metadata attached to the span.
        duration_ms: Wall-clock duration in milliseconds, or ``None`` until the
            span has exited.
    """

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None


class SpanHandle:
    """Mutable handle yielded inside a ``tracer.span(...)`` block.

    Call :meth:`set` to attach attributes while the span is open. The handle
    wraps a :class:`Span`; the owning tracer is responsible for timing and for
    recording/exporting the span on exit.
    """

    def __init__(self, span: Span) -> None:
        self._span = span

    @property
    def span(self) -> Span:
        """Return the underlying :class:`Span` object."""
        return self._span

    def set(self, **attrs: Any) -> SpanHandle:
        """Attach (or overwrite) attributes on the span.

        Args:
            **attrs: Key/value metadata to merge into the span attributes.

        Returns:
            This handle, to allow fluent chaining.
        """
        self._span.attributes.update(attrs)
        return self


@runtime_checkable
class Tracer(Protocol):
    """Minimal tracing interface every tracer implementation provides."""

    name: str

    def span(self, name: str, **attributes: Any) -> Any:
        """Open a span context manager yielding a :class:`SpanHandle`.

        Args:
            name: Name of the span.
            **attributes: Initial attributes recorded on the span.

        Returns:
            A context manager that yields a :class:`SpanHandle`.
        """
        ...


class NoOpTracer:
    """Tracer that does nothing; the default, zero-overhead implementation."""

    name = "none"

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanHandle]:
        """Yield a throwaway :class:`SpanHandle` without recording anything.

        Args:
            name: Name of the span (ignored beyond the yielded handle).
            **attributes: Initial attributes on the throwaway span.

        Yields:
            A :class:`SpanHandle` whose mutations are not retained.
        """
        yield SpanHandle(Span(name=name, attributes=dict(attributes)))


class InMemoryTracer:
    """Tracer that records finished spans in ``.spans`` for tests and debugging."""

    name = "memory"

    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanHandle]:
        """Time the block and append the finished :class:`Span` to ``.spans``.

        The span is recorded even if the wrapped block raises, so failures are
        observable; the exception is re-raised after timing.

        Args:
            name: Name of the span.
            **attributes: Initial attributes on the span.

        Yields:
            A :class:`SpanHandle` for attaching further attributes via ``.set``.
        """
        span = Span(name=name, attributes=dict(attributes))
        handle = SpanHandle(span)
        start = time.perf_counter()
        try:
            yield handle
        finally:
            span.duration_ms = (time.perf_counter() - start) * 1000.0
            self.spans.append(span)


class LangfuseTracer:
    """Tracer backed by Langfuse, lazily imported and fully optional.

    If the ``langfuse`` package is not installed or no client can be created
    (missing keys/config), the tracer logs a warning once and behaves exactly
    like :class:`NoOpTracer`. It never raises on construction or use.
    """

    name = "langfuse"

    def __init__(self, **kwargs: Any) -> None:
        self._client = self._make_client(**kwargs)

    @staticmethod
    def _make_client(**kwargs: Any) -> Any | None:
        """Build a Langfuse client, returning ``None`` if unavailable.

        Args:
            **kwargs: Forwarded to the Langfuse client constructor.

        Returns:
            A Langfuse client instance, or ``None`` when the dependency or its
            configuration is missing.
        """
        try:
            from langfuse import Langfuse  # local import keeps the dep optional
        except ImportError:
            logger.warning("langfuse is not installed; LangfuseTracer degrades to no-op")
            return None
        try:
            return Langfuse(**kwargs)
        except Exception as exc:  # noqa: BLE001 - never let observability crash callers
            logger.warning("Could not initialise Langfuse client (%s); degrading to no-op", exc)
            return None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanHandle]:
        """Time the block and report the span to Langfuse when available.

        Args:
            name: Name of the span.
            **attributes: Initial attributes on the span.

        Yields:
            A :class:`SpanHandle` for attaching further attributes via ``.set``.
        """
        span = Span(name=name, attributes=dict(attributes))
        handle = SpanHandle(span)
        start = time.perf_counter()
        try:
            yield handle
        finally:
            span.duration_ms = (time.perf_counter() - start) * 1000.0
            self._report(span)

    def _report(self, span: Span) -> None:
        """Best-effort export of a finished span to Langfuse.

        Args:
            span: The finished span to report. Any failure is logged and
                swallowed so tracing never breaks the caller.
        """
        if self._client is None:
            return
        try:
            self._client.span(
                name=span.name,
                metadata=span.attributes,
                start_time=None,
                end_time=None,
            )
        except Exception as exc:  # noqa: BLE001 - observability must not crash callers
            logger.warning("Failed to report span '%s' to Langfuse (%s)", span.name, exc)


_BUILDERS: dict[str, Any] = {
    "none": lambda **kwargs: NoOpTracer(),
    "memory": lambda **kwargs: InMemoryTracer(),
    "langfuse": lambda **kwargs: LangfuseTracer(**kwargs),
}


def build_tracer(name: str = "none", **kwargs: Any) -> Tracer:
    """Build a tracer by name, defaulting to the no-op tracer.

    Args:
        name: One of ``"none"``, ``"memory"`` or ``"langfuse"``. Unknown names
            log a warning and fall back to the no-op tracer.
        **kwargs: Extra keyword arguments forwarded to the selected tracer
            (currently only consumed by :class:`LangfuseTracer`).

    Returns:
        A :class:`Tracer` implementation.
    """
    builder = _BUILDERS.get(name)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS))
        logger.warning("Unknown tracer '%s' (known: %s); using no-op", name, known)
        return NoOpTracer()
    return builder(**kwargs)  # type: ignore[no-any-return]
