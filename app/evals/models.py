"""Pydantic models for the RAG evaluation harness."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    """A single labelled evaluation case.

    Attributes:
        question: The question posed to the answering function.
        expected_keywords: Keywords a correct answer is expected to contain.
        reference_numbers: Numbers a faithful answer must preserve verbatim.
    """

    question: str
    expected_keywords: list[str]
    reference_numbers: list[float] = Field(default_factory=list)


class CaseResult(BaseModel):
    """The scored outcome of evaluating one :class:`GoldenCase`.

    Attributes:
        question: The question that was evaluated.
        metrics: Metric name to score in ``[0, 1]``.
        passed: Whether every metric met its threshold.
    """

    question: str
    metrics: dict[str, float]
    passed: bool


class EvalReport(BaseModel):
    """Aggregate report over a set of evaluated cases.

    Attributes:
        cases: Per-case results in evaluation order.
        averages: Metric name to mean score across all cases.
        pass_rate: Fraction of cases that passed, in ``[0, 1]``.
    """

    cases: list[CaseResult]
    averages: dict[str, float]
    pass_rate: float
