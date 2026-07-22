"""Tests for the Expected Calibration Error and reliability table (pure numpy).

The invariants pinned here are the ones that make ECE meaningful: a perfectly
calibrated predictor has near-zero error, an over-confident-but-often-wrong one
has a large error, the result always lands in ``[0, 1]``, and empty input is
guarded to ``0.0`` rather than crashing. All computation is offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.evals.calibration import (
    ReliabilityBin,
    expected_calibration_error,
    reliability_table,
)


def _synthetic_calibrated(n_per_bin: int = 200) -> tuple[list[float], list[float]]:
    """Build an almost-perfectly-calibrated confidence/correctness sample.

    For each of ten confidence levels ``p`` it draws ``n_per_bin`` correctness
    flags that are ``1`` with probability ``p``; empirically accuracy tracks
    confidence, so the ECE is tiny.

    Args:
        n_per_bin: Samples generated per confidence level.

    Returns:
        A ``(confidences, correct)`` pair aligned elementwise.
    """
    rng = np.random.default_rng(0)
    confidences: list[float] = []
    correct: list[float] = []
    for p in np.linspace(0.05, 0.95, 10):
        confidences.extend([float(p)] * n_per_bin)
        correct.extend((rng.random(n_per_bin) < p).astype(float).tolist())
    return confidences, correct


def test_perfect_calibrator_has_near_zero_ece() -> None:
    confidences, correct = _synthetic_calibrated()
    ece = expected_calibration_error(confidences, correct)
    # Sampling noise keeps it just above zero, but a calibrated predictor is
    # well under a loose 0.05 ceiling.
    assert ece < 0.05


def test_overconfident_calibrator_has_large_ece() -> None:
    # Always claims 0.99 confidence but is right only half the time.
    confidences = [0.99] * 100
    correct = [1.0, 0.0] * 50
    ece = expected_calibration_error(confidences, correct)
    # The gap is |0.5 accuracy - 0.99 confidence| ~ 0.49.
    assert ece > 0.4


def test_ece_beats_perfect_when_miscalibrated() -> None:
    good_conf, good_correct = _synthetic_calibrated()
    good_ece = expected_calibration_error(good_conf, good_correct)
    bad_ece = expected_calibration_error([0.99] * 100, [1.0, 0.0] * 50)
    # The whole point: a miscalibrated predictor scores strictly worse.
    assert bad_ece > good_ece


def test_ece_in_unit_interval() -> None:
    confidences, correct = _synthetic_calibrated()
    ece = expected_calibration_error(confidences, correct)
    assert 0.0 <= ece <= 1.0


def test_empty_input_is_guarded_to_zero() -> None:
    assert expected_calibration_error([], []) == 0.0
    assert reliability_table([], []) == []


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.6], [1.0])


def test_reliability_table_bins_and_counts() -> None:
    # Two confidence clusters land in two different bins.
    confidences = [0.15, 0.15, 0.85, 0.85]
    correct = [0.0, 1.0, 1.0, 1.0]
    table = reliability_table(confidences, correct, n_bins=10)
    assert all(isinstance(row, ReliabilityBin) for row in table)
    assert len(table) == 2
    total_count = sum(row.count for row in table)
    assert total_count == len(confidences)
    # The low-confidence bin observed 50% accuracy, the high-confidence 100%.
    low, high = table[0], table[1]
    assert low.accuracy == 0.5
    assert high.accuracy == 1.0


def test_confidence_of_one_lands_in_last_bin() -> None:
    # A confidence of exactly 1.0 must be counted (closed last bin), not dropped.
    table = reliability_table([1.0], [1.0], n_bins=10)
    assert sum(row.count for row in table) == 1
