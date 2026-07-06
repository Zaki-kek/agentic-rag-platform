"""Evals package: a dependency-free harness for scoring RAG answers."""

from app.evals.metrics import grounding, keyword_recall, numbers_preserved
from app.evals.models import CaseResult, EvalReport, GoldenCase
from app.evals.runner import AnswerFn, EvalRunner, load_golden_set

__all__ = [
    "grounding",
    "keyword_recall",
    "numbers_preserved",
    "CaseResult",
    "EvalReport",
    "GoldenCase",
    "AnswerFn",
    "EvalRunner",
    "load_golden_set",
]
