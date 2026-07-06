"""Deterministic computation.

The pattern: never let the LLM invent numbers. Any figure, statistic or table is
computed here in plain Python, and the model is only allowed to write prose around
these verified values. A quality gate then checks the prose did not alter them.
"""

from __future__ import annotations

import statistics


def compute_summary_stats(values: list[float]) -> dict[str, float]:
    """Return deterministic summary statistics for a list of numbers."""
    if not values:
        raise ValueError("values must be non-empty")
    n = len(values)
    return {
        "count": float(n),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "stdev": round(statistics.stdev(values), 2) if n > 1 else 0.0,
    }
