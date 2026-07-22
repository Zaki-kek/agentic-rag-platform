"""Evals package: a dependency-free harness for scoring RAG answers."""

from app.evals.calibration import (
    ReliabilityBin,
    expected_calibration_error,
    reliability_table,
)
from app.evals.composite import composite_faithfulness, separation
from app.evals.judge import (
    EnsembleJudge,
    JudgedCase,
    JudgeResult,
    load_judged_set,
)
from app.evals.metrics import grounding, keyword_recall, numbers_preserved
from app.evals.models import CaseResult, EvalReport, GoldenCase
from app.evals.retrieval import hit_at_k, mrr
from app.evals.runner import AnswerFn, EvalRunner, load_golden_set

__all__ = [
    "grounding",
    "keyword_recall",
    "numbers_preserved",
    "hit_at_k",
    "mrr",
    "CaseResult",
    "EvalReport",
    "GoldenCase",
    "AnswerFn",
    "EvalRunner",
    "load_golden_set",
    "EnsembleJudge",
    "JudgedCase",
    "JudgeResult",
    "load_judged_set",
    "expected_calibration_error",
    "reliability_table",
    "ReliabilityBin",
    "composite_faithfulness",
    "separation",
]
