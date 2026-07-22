"""Tests for offline retrieval metrics (hit@k, MRR) with hand-computed values."""

from __future__ import annotations

from app.evals.retrieval import hit_at_k, mrr

# The canonical worked example: relevant id "A" sits at rank 2 of ["B","A","C"].
_RETRIEVED = ["B", "A", "C"]
_RELEVANT = ["A"]


def test_hit_at_k_finds_relevant_within_top_k() -> None:
    # "A" is inside the top 3 -> a hit.
    assert hit_at_k(_RETRIEVED, _RELEVANT, 3) == 1.0


def test_hit_at_k_misses_when_relevant_below_k() -> None:
    # "A" is at rank 2, so the top 1 (["B"]) contains no relevant id.
    assert hit_at_k(_RETRIEVED, _RELEVANT, 1) == 0.0


def test_mrr_uses_reciprocal_of_first_relevant_rank() -> None:
    # First relevant id "A" is at 1-based rank 2 -> 1/2.
    assert mrr(_RETRIEVED, _RELEVANT) == 0.5


def test_hit_at_k_top_two_includes_relevant() -> None:
    # Top 2 (["B","A"]) now includes "A" -> a hit.
    assert hit_at_k(_RETRIEVED, _RELEVANT, 2) == 1.0


def test_mrr_first_position_is_one() -> None:
    # Relevant id ranked first -> reciprocal rank 1.0.
    assert mrr(["A", "B", "C"], ["A"]) == 1.0


def test_empty_relevant_yields_zero() -> None:
    # No labelled relevant ids: both metrics are 0.0, not a crash.
    assert hit_at_k(_RETRIEVED, [], 3) == 0.0
    assert mrr(_RETRIEVED, []) == 0.0


def test_k_larger_than_list_uses_whole_list() -> None:
    # k beyond the list length simply scans everything; "A" is present.
    assert hit_at_k(_RETRIEVED, _RELEVANT, 99) == 1.0


def test_k_non_positive_finds_nothing() -> None:
    # k <= 0 inspects no results -> a miss even though "A" is relevant.
    assert hit_at_k(_RETRIEVED, _RELEVANT, 0) == 0.0


def test_no_relevant_retrieved_mrr_is_zero() -> None:
    # Relevant id "Z" never appears in the ranking -> 0.0.
    assert mrr(_RETRIEVED, ["Z"]) == 0.0
    assert hit_at_k(_RETRIEVED, ["Z"], 3) == 0.0
