"""Tests for the offline RAG evaluation harness (no keys, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import AppError
from app.evals.metrics import grounding, keyword_recall, numbers_preserved
from app.evals.models import GoldenCase
from app.evals.runner import EvalRunner, load_golden_set

# asyncio_mode="auto" runs the async tests below with no per-test marker; the
# synchronous metric tests stay plain functions, so no module-level mark.

_GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "app" / "evals" / "golden_set.json"


# --- metric unit tests (hand-computed expectations) ---------------------------


def test_keyword_recall_counts_case_insensitive_hits() -> None:
    # "paris" present, "Rome" absent -> 1 of 2.
    assert keyword_recall("The tower stands in PARIS.", ["Paris", "Rome"]) == 0.5


def test_keyword_recall_empty_requirements_is_perfect() -> None:
    assert keyword_recall("anything", []) == 1.0


def test_grounding_fraction_of_content_words_in_context() -> None:
    # Content words (len >= 3) in answer: "the", "eiffel", "tower", "stands",
    # "paris". "stands" is absent from the context -> 4 of 5 grounded.
    answer = "The Eiffel Tower stands in Paris"
    contexts = ["The Eiffel Tower is a landmark located in Paris, France."]
    assert grounding(answer, contexts) == pytest.approx(4 / 5)


def test_grounding_empty_answer_is_perfect() -> None:
    assert grounding("", ["some context"]) == 1.0


def test_numbers_preserved_matches_verbatim_only() -> None:
    # 100 present, 42 absent -> 1 of 2. Integer floats compare without ".0".
    assert numbers_preserved("Water boils at 100 degrees.", [100.0, 42.0]) == 0.5


def test_numbers_preserved_no_false_substring_match() -> None:
    # "42" must not match inside "426".
    assert numbers_preserved("The code is 426.", [42.0]) == 0.0


def test_numbers_preserved_none_required_is_perfect() -> None:
    assert numbers_preserved("no numbers here", []) == 1.0


# --- runner end-to-end tests --------------------------------------------------


async def _fake_answer_fn(question: str) -> tuple[str, list[str]]:
    """Canned answers + contexts keyed by question (fully deterministic)."""
    table: dict[str, tuple[str, list[str]]] = {
        "good": (
            "The Eiffel Tower is located in Paris, France.",
            ["The Eiffel Tower is located in Paris, France."],
        ),
        "bad": (
            "I am not sure about that.",
            ["The Eiffel Tower is located in Paris, France."],
        ),
    }
    return table[question]


async def test_runner_computes_metrics_averages_and_pass_rate() -> None:
    cases = [
        GoldenCase(question="good", expected_keywords=["Paris", "France"]),
        GoldenCase(question="bad", expected_keywords=["Paris", "France"]),
    ]
    runner = EvalRunner(_fake_answer_fn)
    report = await runner.run(cases)

    assert len(report.cases) == 2

    good, bad = report.cases
    assert good.passed is True
    assert good.metrics["keyword_recall"] == 1.0
    assert good.metrics["grounding"] == pytest.approx(1.0)

    # "bad" answer mentions neither keyword and is ungrounded -> fails.
    assert bad.passed is False
    assert bad.metrics["keyword_recall"] == 0.0

    # Averages are the per-case means; pass_rate is 1 of 2.
    assert report.averages["keyword_recall"] == pytest.approx(0.5)
    assert report.pass_rate == pytest.approx(0.5)


async def test_runner_respects_custom_thresholds() -> None:
    # grounding for "good" is 1.0; require above that to force a fail.
    cases = [GoldenCase(question="good", expected_keywords=["Paris"])]
    runner = EvalRunner(_fake_answer_fn)
    report = await runner.run(cases, thresholds={"grounding": 1.01})

    assert report.cases[0].passed is False
    assert report.pass_rate == 0.0


async def test_runner_empty_cases_yields_zeroed_report() -> None:
    runner = EvalRunner(_fake_answer_fn)
    report = await runner.run([])

    assert report.cases == []
    assert report.pass_rate == 0.0
    assert report.averages["keyword_recall"] == 0.0
    assert report.averages["grounding"] == 0.0
    assert report.averages["numbers_preserved"] == 0.0


# --- golden set loading -------------------------------------------------------


def test_load_golden_set_parses_bundled_file() -> None:
    cases = load_golden_set(_GOLDEN_SET_PATH)
    assert len(cases) >= 3
    assert all(isinstance(c, GoldenCase) for c in cases)
    assert all(c.question and c.expected_keywords for c in cases)


def test_load_golden_set_missing_file_raises_app_error() -> None:
    with pytest.raises(AppError):
        load_golden_set(Path("/nonexistent/does_not_exist.json"))
