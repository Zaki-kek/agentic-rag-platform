"""Quality gates: checks an output must pass before a stage is accepted.

A gate inspects the job context and returns pass/fail with a reason. Gates are the
guardrails between LLM stages - a failed gate triggers a retry of the stage.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.generation.models import GateResult


class QualityGate(Protocol):
    """A named check over the job context."""

    name: str

    def check(self, context: dict[str, Any]) -> GateResult: ...


class NonEmptyGate:
    """Passes when a context field exists and is non-empty."""

    def __init__(self, field: str) -> None:
        self.name = f"non_empty:{field}"
        self._field = field

    def check(self, context: dict[str, Any]) -> GateResult:
        value = context.get(self._field)
        if value:
            return GateResult(True)
        return GateResult(False, f"field '{self._field}' is empty")


class NumbersPreservedGate:
    """Passes when every computed number appears verbatim in the generated text.

    This is what makes the deterministic-computation pattern safe: it proves the
    LLM narrated the verified figures instead of inventing or rounding new ones.
    """

    def __init__(self, numbers_field: str, text_field: str) -> None:
        self.name = f"numbers_preserved:{numbers_field}->{text_field}"
        self._numbers_field = numbers_field
        self._text_field = text_field

    def check(self, context: dict[str, Any]) -> GateResult:
        numbers = context.get(self._numbers_field) or {}
        text = context.get(self._text_field) or ""
        values = numbers.values() if isinstance(numbers, dict) else numbers
        missing = [str(v) for v in values if str(v) not in text]
        if missing:
            return GateResult(False, f"missing numbers in text: {missing}")
        return GateResult(True)
