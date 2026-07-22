"""Composite faithfulness score and a strict pairwise separation (ROC-AUC).

Two building blocks for turning the individual proxy metrics into a research
signal:

* :func:`composite_faithfulness` - a weighted combination of ``grounding``,
  ``keyword_recall`` and ``numbers_preserved`` into one faithfulness number.
  Because it is a convex combination of ``[0, 1]`` inputs it is itself in
  ``[0, 1]`` and monotone non-decreasing in every component.
* :func:`separation` - a strict pairwise ROC-AUC (equivalently the normalised
  Mann-Whitney U statistic): the fraction of ``(good, bad)`` score pairs the
  metric orders correctly, counting ties as ``0.5``. A metric that perfectly
  ranks good above bad scores 1.0; a metric that carries no signal (all ties,
  e.g. a leaked proxy that saturates at 1.0 for everything) scores 0.5.

Pure ``numpy`` - no ``scipy``/``sklearn`` - so it runs offline.
"""

from __future__ import annotations

import numpy as np

# Default composite weights. Grounding (faithfulness to context) carries the
# most weight; keyword coverage and numeric preservation split the rest.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "grounding": 0.5,
    "keyword_recall": 0.25,
    "numbers_preserved": 0.25,
}


def composite_faithfulness(
    grounding: float,
    keyword_recall: float,
    numbers_preserved: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted combination of the three proxy metrics into one score.

    Args:
        grounding: Grounding/faithfulness proxy in ``[0, 1]``.
        keyword_recall: Keyword-coverage proxy in ``[0, 1]``.
        numbers_preserved: Numeric-preservation proxy in ``[0, 1]``.
        weights: Optional non-negative weights keyed by ``"grounding"``,
            ``"keyword_recall"`` and ``"numbers_preserved"``. Missing keys
            default to ``0.0``; defaults to :data:`_DEFAULT_WEIGHTS`.

    Returns:
        The weighted mean of the components, in ``[0, 1]``. Returns ``0.0`` when
        the weights sum to zero (guard).
    """
    used = weights or _DEFAULT_WEIGHTS
    components = {
        "grounding": grounding,
        "keyword_recall": keyword_recall,
        "numbers_preserved": numbers_preserved,
    }
    total_weight = sum(used.get(name, 0.0) for name in components)
    if total_weight <= 0.0:
        return 0.0
    weighted = sum(used.get(name, 0.0) * value for name, value in components.items())
    return weighted / total_weight


def separation(good_scores: list[float], bad_scores: list[float]) -> float:
    """Strict pairwise ROC-AUC of ``good`` scores over ``bad`` scores.

    This is the fraction of all ``(good, bad)`` pairs for which the good score
    is strictly greater than the bad score, with ties contributing ``0.5``. It
    equals the area under the ROC curve and the normalised Mann-Whitney U
    statistic: 1.0 means the metric ranks every good answer above every bad
    one, 0.5 means it has no discriminative power (e.g. all scores tie).

    Args:
        good_scores: Metric scores for the gold-``good`` answers.
        bad_scores: Metric scores for the gold-``bad`` answers.

    Returns:
        The separation in ``[0, 1]``. Returns ``0.0`` when either group is
        empty (nothing to compare).
    """
    good = np.asarray(good_scores, dtype=np.float64).reshape(-1)
    bad = np.asarray(bad_scores, dtype=np.float64).reshape(-1)
    if good.size == 0 or bad.size == 0:
        return 0.0
    diff = good[:, None] - bad[None, :]
    wins = float((diff > 0.0).sum()) + 0.5 * float((diff == 0.0).sum())
    return wins / float(good.size * bad.size)
