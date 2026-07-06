"""Validation of inline ``[n]`` citation markers against an available source count."""

from __future__ import annotations

import re

from app.core import get_logger

logger = get_logger(__name__)

# Matches inline citation markers like [1], [12]; ignores [ranges] or [non-numeric].
_MARKER = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, num_sources: int) -> list[str]:
    """Validate inline ``[n]`` citation markers in an answer.

    Every ``[n]`` marker found in ``answer`` must reference a source index in the
    range ``1 <= n <= num_sources``. Markers outside that range are reported as
    problems. When ``num_sources`` is zero, any marker is out of range.

    Args:
        answer: The generated answer text containing inline ``[n]`` markers.
        num_sources: The number of available sources (citation indices ``1..N``).

    Returns:
        A list of human-readable problem descriptions. An empty list means every
        marker is valid (or there were no markers to check).
    """
    if num_sources < 0:
        return [f"num_sources must be non-negative, got {num_sources}"]

    problems: list[str] = []
    seen: set[int] = set()
    for m in _MARKER.finditer(answer):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        if n < 1 or n > num_sources:
            problems.append(f"citation [{n}] is out of range (valid range: 1..{num_sources})")

    if problems:
        logger.debug("validate_citations: %d problem(s) over %d source(s)", len(problems), num_sources)
    return problems
