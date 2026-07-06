"""Evaluation runner: scores an answer function against a golden set."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.core import AppError, get_logger
from app.evals.metrics import grounding, keyword_recall, numbers_preserved
from app.evals.models import CaseResult, EvalReport, GoldenCase

logger = get_logger(__name__)

# Signature of the function under test: question -> (answer, contexts_used).
AnswerFn = Callable[[str], Awaitable[tuple[str, list[str]]]]

# Metric names produced for every case, used to seed averages deterministically.
_METRIC_NAMES = ("keyword_recall", "grounding", "numbers_preserved")

_DEFAULT_THRESHOLD = 0.5


class EvalRunner:
    """Run an async answer function over golden cases and aggregate metrics.

    The runner is provider-agnostic: it only needs an ``answer_fn`` that maps a
    question to ``(answer, contexts)``. This keeps evaluation fully offline -
    the caller can pass a real RAG pipeline or a canned fake.
    """

    def __init__(self, answer_fn: AnswerFn) -> None:
        """Initialise the runner.

        Args:
            answer_fn: Async callable returning the answer and the contexts it
                used for a given question.
        """
        self._answer_fn = answer_fn

    async def run(self, cases: list[GoldenCase], thresholds: dict[str, float] | None = None) -> EvalReport:
        """Evaluate every case and return an aggregate report.

        Args:
            cases: The golden cases to evaluate.
            thresholds: Optional per-metric pass thresholds. Metrics without an
                entry use the default threshold of ``0.5``.

        Returns:
            An :class:`EvalReport` with per-case results, metric averages and
            the overall pass rate.
        """
        thresholds = thresholds or {}
        results: list[CaseResult] = []
        for case in cases:
            answer, contexts = await self._answer_fn(case.question)
            metrics = {
                "keyword_recall": keyword_recall(answer, case.expected_keywords),
                "grounding": grounding(answer, contexts),
                "numbers_preserved": numbers_preserved(answer, case.reference_numbers),
            }
            passed = all(score >= thresholds.get(name, _DEFAULT_THRESHOLD) for name, score in metrics.items())
            results.append(CaseResult(question=case.question, metrics=metrics, passed=passed))

        averages = self._average_metrics(results)
        pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0
        logger.info("Evaluated %d case(s); pass_rate=%.3f", len(results), pass_rate)
        return EvalReport(cases=results, averages=averages, pass_rate=pass_rate)

    @staticmethod
    def _average_metrics(results: list[CaseResult]) -> dict[str, float]:
        """Mean of each metric across all results.

        Args:
            results: The scored per-case results.

        Returns:
            Metric name to mean score; ``0.0`` for every metric when empty.
        """
        if not results:
            return {name: 0.0 for name in _METRIC_NAMES}
        return {name: sum(r.metrics[name] for r in results) / len(results) for name in _METRIC_NAMES}


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    """Load and validate a golden set from a JSON file.

    Args:
        path: Path to a JSON file containing a list of golden-case objects.

    Returns:
        The parsed golden cases.

    Raises:
        AppError: If the file is missing or does not contain a JSON list.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        logger.error("Golden set not found: %s", file_path)
        raise AppError(f"Golden set not found: {file_path}") from exc

    data = json.loads(raw)
    if not isinstance(data, list):
        raise AppError(f"Golden set must be a JSON list, got {type(data).__name__}")
    return [GoldenCase.model_validate(item) for item in data]
