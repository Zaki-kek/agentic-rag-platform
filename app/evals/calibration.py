"""Calibration diagnostics for confidence-scored judgements (pure numpy).

A judge that reports a ``confidence`` is *calibrated* when that confidence
matches the empirical probability of being correct: among answers it flags with
0.8 confidence, roughly 80% should indeed be correct. The Expected Calibration
Error (ECE) is the standard binned summary of that gap.

Everything here is pure ``numpy`` (no ``scipy``/``sklearn``): confidences and
correctness flags are bucketed into equal-width bins, and the ECE is the
count-weighted mean of ``|accuracy - mean_confidence|`` over non-empty bins.
All values stay in ``[0, 1]`` and an empty input is guarded to ``0.0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability table.

    Attributes:
        lower: Inclusive lower edge of the confidence bin, in ``[0, 1]``.
        upper: Upper edge of the confidence bin, in ``[0, 1]``.
        mean_confidence: Mean predicted confidence of the samples in the bin.
        accuracy: Empirical accuracy (mean correctness) of the samples.
        count: Number of samples that fell into the bin.
    """

    lower: float
    upper: float
    mean_confidence: float
    accuracy: float
    count: int


def _as_float_array(values: list[float]) -> np.ndarray:
    """Coerce a list of numbers to a 1-D float64 array.

    Args:
        values: The numbers to coerce.

    Returns:
        A 1-D ``float64`` ``numpy`` array (empty if ``values`` is empty).
    """
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _bin_edges(n_bins: int) -> np.ndarray:
    """Return ``n_bins + 1`` equal-width edges spanning ``[0, 1]``.

    Args:
        n_bins: Number of bins (clamped to at least 1).

    Returns:
        A 1-D array of bin edges from ``0.0`` to ``1.0`` inclusive.
    """
    return np.linspace(0.0, 1.0, max(n_bins, 1) + 1)


def _bin_mask(confidences: np.ndarray, edges: np.ndarray, index: int) -> np.ndarray:
    """Boolean mask of confidences that fall into bin ``index``.

    The bins are half-open ``[lo, hi)`` except the last, which is closed
    ``[lo, hi]`` so a confidence of exactly ``1.0`` is counted.

    Args:
        confidences: 1-D array of confidence values.
        edges: Bin edges from :func:`_bin_edges`.
        index: Zero-based bin index.

    Returns:
        A boolean array selecting the samples in the bin.
    """
    lower, upper = edges[index], edges[index + 1]
    last = index == len(edges) - 2
    if last:
        return (confidences >= lower) & (confidences <= upper)
    return (confidences >= lower) & (confidences < upper)


def expected_calibration_error(
    confidences: list[float], correct: list[float], n_bins: int = 10
) -> float:
    """Expected Calibration Error of confidences against correctness flags.

    Confidences are bucketed into ``n_bins`` equal-width bins over ``[0, 1]``.
    For each non-empty bin the absolute gap between empirical accuracy and mean
    confidence is weighted by the bin's share of samples; the ECE is the sum of
    those weighted gaps.

    Args:
        confidences: Predicted confidences, each in ``[0, 1]``.
        correct: Correctness flags aligned to ``confidences`` (``1``/``0`` or
            ``True``/``False``).
        n_bins: Number of equal-width confidence bins.

    Returns:
        The ECE in ``[0, 1]``. Returns ``0.0`` for empty input (guard).

    Raises:
        ValueError: If ``confidences`` and ``correct`` have different lengths.
    """
    conf = _as_float_array(confidences)
    corr = _as_float_array(correct)
    if conf.size != corr.size:
        raise ValueError(f"length mismatch: {conf.size} confidences vs {corr.size} correct")
    if conf.size == 0:
        return 0.0

    edges = _bin_edges(n_bins)
    total = float(conf.size)
    error = 0.0
    for index in range(len(edges) - 1):
        mask = _bin_mask(conf, edges, index)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(corr[mask].mean())
        mean_conf = float(conf[mask].mean())
        error += (count / total) * abs(accuracy - mean_conf)
    return error


def reliability_table(
    confidences: list[float], correct: list[float], n_bins: int = 10
) -> list[ReliabilityBin]:
    """Per-bin reliability breakdown behind the ECE.

    Args:
        confidences: Predicted confidences, each in ``[0, 1]``.
        correct: Correctness flags aligned to ``confidences``.
        n_bins: Number of equal-width confidence bins.

    Returns:
        One :class:`ReliabilityBin` per non-empty bin, in ascending confidence
        order. Empty input yields an empty list.

    Raises:
        ValueError: If ``confidences`` and ``correct`` have different lengths.
    """
    conf = _as_float_array(confidences)
    corr = _as_float_array(correct)
    if conf.size != corr.size:
        raise ValueError(f"length mismatch: {conf.size} confidences vs {corr.size} correct")
    if conf.size == 0:
        return []

    edges = _bin_edges(n_bins)
    table: list[ReliabilityBin] = []
    for index in range(len(edges) - 1):
        mask = _bin_mask(conf, edges, index)
        count = int(mask.sum())
        if count == 0:
            continue
        table.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                mean_confidence=float(conf[mask].mean()),
                accuracy=float(corr[mask].mean()),
                count=count,
            )
        )
    return table
