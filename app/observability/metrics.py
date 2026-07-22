"""Tiny in-memory metrics registry (Prometheus text format, no dependencies).

This module provides a minimal, dependency-free metrics facility: a monotonic
:class:`Counter` and a process-wide :data:`REGISTRY` that can render all
registered counters in the Prometheus text exposition format. It is intentionally
small - just enough to back a ``/metrics`` endpoint without pulling in
``prometheus_client`` or any other observability dependency.
"""

from __future__ import annotations

from threading import Lock

__all__ = ["Counter", "Registry", "REGISTRY", "http_requests_total"]


class Counter:
    """A monotonically increasing counter.

    Attributes:
        name: Metric name exposed in the Prometheus text format.
        help: Human-readable description of the metric.
    """

    def __init__(self, name: str, help: str = "") -> None:
        self.name = name
        self.help = help
        self._value = 0.0
        self._lock = Lock()

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter.

        Args:
            amount: Non-negative value to add. Defaults to ``1.0``.

        Raises:
            ValueError: When ``amount`` is negative (counters never decrease).
        """
        if amount < 0:
            raise ValueError("counter increment must be non-negative")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        """Current counter value."""
        with self._lock:
            return self._value


class Registry:
    """A collection of counters that renders Prometheus text format."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._lock = Lock()

    def counter(self, name: str, help: str = "") -> Counter:
        """Return the registered counter for ``name``, creating it if absent."""
        with self._lock:
            counter = self._counters.get(name)
            if counter is None:
                counter = Counter(name, help)
                self._counters[name] = counter
            return counter

    def render(self) -> str:
        """Render all counters in the Prometheus text exposition format.

        Returns:
            A newline-terminated string of ``# TYPE`` headers and samples.
        """
        with self._lock:
            counters = list(self._counters.values())
        lines: list[str] = []
        for counter in counters:
            if counter.help:
                lines.append(f"# HELP {counter.name} {counter.help}")
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(f"{counter.name} {counter.value}")
        return "\n".join(lines) + "\n"


REGISTRY = Registry()
http_requests_total = REGISTRY.counter(
    "http_requests_total", "Total number of HTTP requests handled."
)
