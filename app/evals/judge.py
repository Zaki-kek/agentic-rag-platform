"""Ensemble LLM-judge with honest, offline self-consistency.

This module scores RAG answers with an *ensemble of independent weak judges*
built on top of the deterministic proxy metrics in :mod:`app.evals.metrics`.
It deliberately avoids a fake "temperature" self-consistency trick: the string
proxies are order-symmetric, so re-sampling them would agree trivially. Instead
each weak judge inspects a *different* signal (grounding, numeric preservation,
keyword coverage with a length/truncation penalty) and casts an independent
good/bad vote. Disagreement is therefore real - different judges catch
different failure modes:

* a *corrupted-number* answer stays grounded by words yet fails the numeric
  judge (``numbers_preserved == 0.0`` while ``grounding > 0``);
* a *hallucinated-by-words* answer reads fluently but fails the grounding and
  keyword judges (``grounding`` low), while the numeric judge abstains
  vacuously when no reference numbers exist.

Everything here is pure Python: no network, no API keys, fully deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.evals.metrics import grounding, keyword_recall, numbers_preserved

# Default vote thresholds. A weak judge votes "good" when its metric meets or
# exceeds the threshold. These are intentionally simple and shared so that the
# only thing distinguishing the judges is *which* signal they read.
_DEFAULT_GROUNDING_THRESHOLD = 0.5
_DEFAULT_NUMBERS_THRESHOLD = 0.5
_DEFAULT_KEYWORD_THRESHOLD = 0.5

# Answers shorter than this many content characters are treated as truncated /
# empty and penalised by the keyword judge so a one-word stub cannot pass on a
# lucky keyword hit.
_MIN_ANSWER_CHARS = 8


@dataclass(frozen=True)
class JudgedCase:
    """A single hand-written judge case with a gold ``good``/``bad`` label.

    Attributes:
        question: The question the answer responds to.
        answer: The answer text under evaluation (hand-written, not echoed).
        label: The gold label, ``"good"`` or ``"bad"``.
        contexts: Retrieved context strings the answer should be grounded in.
        reference_numbers: Numbers a faithful answer must reproduce verbatim.
        expected_keywords: Keywords a correct answer is expected to mention.
    """

    question: str
    answer: str
    label: str
    contexts: list[str] = field(default_factory=list)
    reference_numbers: list[float] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JudgeResult:
    """The verdict of the ensemble on one answer.

    Attributes:
        score: Weighted mean of the judges' good-votes, in ``[0, 1]``.
        confidence: Fraction of judges agreeing with the majority vote, in
            ``(0, 1]``. ``1.0`` means unanimous; lower means the judges split.
        predicted_label: ``"good"`` when ``score >= 0.5`` else ``"bad"``.
        votes: Judge name to its binary good-vote (``1`` good, ``0`` bad).
        metrics: Judge name to the raw metric value it read, in ``[0, 1]``.
    """

    score: float
    confidence: float
    predicted_label: str
    votes: dict[str, int]
    metrics: dict[str, float]


class EnsembleJudge:
    """Ensemble of independent weak judges over the proxy metrics.

    Each judge reads one signal and votes ``good``/``bad`` against a threshold.
    The ensemble ``score`` is the weighted mean of the good-votes and the
    ``confidence`` is the share of judges agreeing with the majority - a real,
    non-trivial self-consistency signal because the judges look at different
    axes and genuinely disagree on adversarial cases.
    """

    def __init__(
        self,
        *,
        grounding_threshold: float = _DEFAULT_GROUNDING_THRESHOLD,
        numbers_threshold: float = _DEFAULT_NUMBERS_THRESHOLD,
        keyword_threshold: float = _DEFAULT_KEYWORD_THRESHOLD,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Build the ensemble.

        Args:
            grounding_threshold: Vote threshold for the grounding judge.
            numbers_threshold: Vote threshold for the numeric-preservation judge.
            keyword_threshold: Vote threshold for the keyword-coverage judge.
            weights: Optional per-judge weights (keys ``"grounding"``,
                ``"numbers"``, ``"keyword"``); defaults to equal weighting.
        """
        self._grounding_threshold = grounding_threshold
        self._numbers_threshold = numbers_threshold
        self._keyword_threshold = keyword_threshold
        self._weights = weights or {"grounding": 1.0, "numbers": 1.0, "keyword": 1.0}

    def _keyword_score(self, case: JudgedCase) -> float:
        """Keyword coverage with a length/truncation penalty.

        Args:
            case: The case being judged.

        Returns:
            Keyword recall in ``[0, 1]``, forced to ``0.0`` when the answer is
            too short to be a real answer (empty or truncated).
        """
        if len(case.answer.strip()) < _MIN_ANSWER_CHARS:
            return 0.0
        return keyword_recall(case.answer, case.expected_keywords)

    def metric_values(self, case: JudgedCase) -> dict[str, float]:
        """Compute the raw metric each judge reads for ``case``.

        Args:
            case: The case being judged.

        Returns:
            A mapping of judge name to its metric value in ``[0, 1]``.
        """
        return {
            "grounding": grounding(case.answer, case.contexts),
            "numbers": numbers_preserved(case.answer, case.reference_numbers),
            "keyword": self._keyword_score(case),
        }

    def judge(self, case: JudgedCase) -> JudgeResult:
        """Score one case with the full ensemble.

        Args:
            case: The case being judged.

        Returns:
            A :class:`JudgeResult` with score, confidence, votes and metrics.
        """
        metrics = self.metric_values(case)
        thresholds = {
            "grounding": self._grounding_threshold,
            "numbers": self._numbers_threshold,
            "keyword": self._keyword_threshold,
        }
        votes = {name: (1 if metrics[name] >= thresholds[name] else 0) for name in metrics}

        total_weight = sum(self._weights.get(name, 1.0) for name in votes)
        weighted_good = sum(self._weights.get(name, 1.0) * vote for name, vote in votes.items())
        score = weighted_good / total_weight if total_weight > 0 else 0.0

        good_votes = sum(votes.values())
        bad_votes = len(votes) - good_votes
        confidence = max(good_votes, bad_votes) / len(votes)

        predicted_label = "good" if score >= 0.5 else "bad"
        return JudgeResult(
            score=score,
            confidence=confidence,
            predicted_label=predicted_label,
            votes=votes,
            metrics=metrics,
        )


# The judged set bundled with the package; hand-written good/bad pairs.
_DEFAULT_JUDGED_SET = Path(__file__).resolve().parent / "judged_set.json"


def load_judged_set(path: Path | None = None) -> list[JudgedCase]:
    """Load hand-written judged cases from a JSON file.

    Args:
        path: Optional path to the JSON file; defaults to the bundled
            ``judged_set.json``.

    Returns:
        The parsed list of :class:`JudgedCase` objects, in file order.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    target = path or _DEFAULT_JUDGED_SET
    raw = json.loads(target.read_text(encoding="utf-8"))
    return [
        JudgedCase(
            question=item["question"],
            answer=item["answer"],
            label=item["label"],
            contexts=item.get("contexts", []),
            reference_numbers=[float(n) for n in item.get("reference_numbers", [])],
            expected_keywords=item.get("expected_keywords", []),
        )
        for item in raw
    ]
