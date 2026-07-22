"""Tests for the ensemble LLM-judge with honest self-consistency.

These tests pin the qualitative invariants the judge must satisfy (not brittle
exact metric numbers): the good/bad separation of *confidence*, and the fact
that different weak judges catch different failure modes (a corrupted number vs
a word-level hallucination). Everything runs offline and deterministically.
"""

from __future__ import annotations

from app.evals.judge import (
    EnsembleJudge,
    JudgedCase,
    JudgeResult,
    load_judged_set,
)


def _by_category() -> tuple[list[JudgedCase], list[JudgedCase], list[JudgedCase]]:
    """Split the bundled judged set into good, corrupt-number and hallucination.

    Returns:
        A ``(good, corrupt, hallucination)`` triple. ``corrupt`` cases carry
        reference numbers (the broken-number attack); ``hallucination`` cases
        carry no reference numbers (the word-level attack).
    """
    cases = load_judged_set()
    good = [c for c in cases if c.label == "good"]
    corrupt = [c for c in cases if c.label == "bad" and c.reference_numbers]
    hallucination = [c for c in cases if c.label == "bad" and not c.reference_numbers]
    return good, corrupt, hallucination


def test_judged_set_has_all_three_categories() -> None:
    good, corrupt, hallucination = _by_category()
    # The set is only meaningful if each adversarial axis is represented.
    assert len(good) >= 4
    assert len(corrupt) >= 2
    assert len(hallucination) >= 2


def test_judge_is_deterministic() -> None:
    judge = EnsembleJudge()
    case = load_judged_set()[0]
    first = judge.judge(case)
    second = judge.judge(case)
    assert isinstance(first, JudgeResult)
    assert first.score == second.score
    assert first.confidence == second.confidence
    assert first.votes == second.votes
    assert first.metrics == second.metrics


def test_score_and_confidence_in_unit_interval() -> None:
    judge = EnsembleJudge()
    for case in load_judged_set():
        result = judge.judge(case)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 < result.confidence <= 1.0


def test_good_cases_have_unanimous_confidence() -> None:
    judge = EnsembleJudge()
    good, _, _ = _by_category()
    for case in good:
        result = judge.judge(case)
        # A genuinely grounded answer with correct numbers and keywords makes
        # every weak judge agree, so confidence is maximal.
        assert result.confidence == 1.0
        assert result.predicted_label == "good"


def test_corrupt_number_splits_on_numeric_axis() -> None:
    judge = EnsembleJudge()
    _, corrupt, _ = _by_category()
    for case in corrupt:
        result = judge.judge(case)
        # The answer stays grounded by words but the number is broken: the
        # numeric judge dissents while the grounding judge still approves, so
        # the ensemble is no longer unanimous.
        assert result.metrics["numbers"] == 0.0
        assert result.metrics["grounding"] > 0.0
        assert result.confidence < 1.0


def test_hallucination_splits_on_grounding_axis() -> None:
    judge = EnsembleJudge()
    _, _, hallucination = _by_category()
    for case in hallucination:
        result = judge.judge(case)
        # Word-level hallucination: assert the grounding invariant (markedly
        # low), NOT numbers_preserved, which is vacuously 1.0 when there are no
        # reference numbers. Grounding disagreement drives the confidence drop.
        assert result.metrics["grounding"] < 0.3
        assert result.confidence < 1.0


def test_confidence_separates_good_from_adversarial() -> None:
    judge = EnsembleJudge()
    good, corrupt, hallucination = _by_category()
    good_conf = [judge.judge(c).confidence for c in good]
    bad_conf = [judge.judge(c).confidence for c in corrupt + hallucination]
    # The whole point of the ensemble: unanimous on clean answers, split on
    # adversarial ones. Minimum good-confidence beats maximum bad-confidence.
    assert min(good_conf) > max(bad_conf)


def test_custom_weights_do_not_break_scoring() -> None:
    judge = EnsembleJudge(weights={"grounding": 2.0, "numbers": 1.0, "keyword": 1.0})
    for case in load_judged_set():
        result = judge.judge(case)
        assert 0.0 <= result.score <= 1.0
