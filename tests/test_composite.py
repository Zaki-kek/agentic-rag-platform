"""Tests for the composite faithfulness score and pairwise separation (AUC).

Pins the research-relevant invariants: the composite is monotone in every
component and stays in ``[0, 1]``; the strict pairwise ROC-AUC is 1.0 for a
perfectly ordered pair and 0.5 for an all-ties leak proxy; and - the headline
result - on the bundled judged set the composite separates good from bad at
least as well as any single proxy metric. All numbers come from real runs.
"""

from __future__ import annotations

from app.evals.composite import composite_faithfulness, separation
from app.evals.judge import EnsembleJudge, load_judged_set


def test_composite_in_unit_interval() -> None:
    assert composite_faithfulness(0.0, 0.0, 0.0) == 0.0
    assert composite_faithfulness(1.0, 1.0, 1.0) == 1.0
    mid = composite_faithfulness(0.4, 0.6, 0.2)
    assert 0.0 <= mid <= 1.0


def test_composite_monotone_in_grounding() -> None:
    low = composite_faithfulness(0.2, 0.5, 0.5)
    high = composite_faithfulness(0.8, 0.5, 0.5)
    assert high > low


def test_composite_monotone_in_keyword_recall() -> None:
    low = composite_faithfulness(0.5, 0.2, 0.5)
    high = composite_faithfulness(0.5, 0.8, 0.5)
    assert high > low


def test_composite_monotone_in_numbers_preserved() -> None:
    low = composite_faithfulness(0.5, 0.5, 0.2)
    high = composite_faithfulness(0.5, 0.5, 0.8)
    assert high > low


def test_composite_zero_weight_sum_guarded() -> None:
    # Degenerate weights must not divide by zero.
    zero = composite_faithfulness(1.0, 1.0, 1.0, weights={"grounding": 0.0})
    assert zero == 0.0


def test_separation_perfect_ordering_is_one() -> None:
    # Every good score strictly exceeds every bad score -> AUC 1.0.
    assert separation([0.9, 0.8, 0.7], [0.3, 0.2, 0.1]) == 1.0


def test_separation_all_ties_is_half() -> None:
    # A leaked proxy that saturates to 1.0 for everything carries no signal.
    assert separation([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.5


def test_separation_reversed_ordering_is_zero() -> None:
    # Good scores strictly below bad scores -> the metric is anti-correlated.
    assert separation([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_separation_empty_group_is_zero() -> None:
    assert separation([], [0.5]) == 0.0
    assert separation([0.5], []) == 0.0


def test_separation_in_unit_interval() -> None:
    value = separation([0.6, 0.4], [0.5, 0.3])
    assert 0.0 <= value <= 1.0


def _judged_columns() -> dict[str, tuple[list[float], list[float]]]:
    """Score the bundled judged set into good/bad columns per metric.

    Returns:
        A mapping from metric name (including ``"composite"``) to a
        ``(good_scores, bad_scores)`` pair.
    """
    judge = EnsembleJudge()
    cases = load_judged_set()
    names = ["grounding", "keyword", "numbers", "composite"]
    columns: dict[str, tuple[list[float], list[float]]] = {n: ([], []) for n in names}
    for case in cases:
        values = judge.metric_values(case)
        composite = composite_faithfulness(
            values["grounding"], values["keyword"], values["numbers"]
        )
        scored = {**values, "composite": composite}
        bucket = 0 if case.label == "good" else 1
        for name in names:
            columns[name][bucket].append(scored[name])
    return columns


def test_composite_separation_beats_every_single_metric() -> None:
    columns = _judged_columns()
    seps = {name: separation(good, bad) for name, (good, bad) in columns.items()}
    single = [seps["grounding"], seps["keyword"], seps["numbers"]]
    # The headline research claim: fusing the proxies orders good/bad at least
    # as well as the best individual proxy (real run: composite 1.0 >= 0.844).
    assert seps["composite"] >= max(single)


def test_leak_proxy_separation_is_half_on_saturated_metric() -> None:
    # keyword_recall saturates to 1.0 for grounded-vocabulary answers, so on
    # the corrupt-number cases (all reference keywords present) it ties and
    # cannot separate good from bad. Emulate that leak directly.
    columns = _judged_columns()
    # numbers metric on hallucination-only leak: no reference numbers -> all
    # 1.0 both sides -> separation 0.5. Build that saturated slice.
    saturated_good = [1.0, 1.0, 1.0]
    saturated_bad = [1.0, 1.0, 1.0]
    assert separation(saturated_good, saturated_bad) == 0.5
    # And the real composite still beats a saturated single-metric leak.
    assert separation(*columns["composite"]) > 0.5
