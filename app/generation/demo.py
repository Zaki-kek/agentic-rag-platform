"""A neutral demo pipeline showcasing the orchestration pattern.

compute (deterministic stats) -> draft (LLM narrates) -> gate (numbers preserved).
The numbers come from Python, the prose from the model, and the gate proves the
model did not alter the figures.
"""

from __future__ import annotations

from typing import Any

from app.generation.compute import compute_summary_stats
from app.generation.gates import NonEmptyGate, NumbersPreservedGate
from app.generation.models import Stage
from app.generation.validators import PlaceholderGate
from app.llm.base import LLMProvider


def build_report_pipeline(provider: LLMProvider) -> list[Stage]:
    """Build the demo report pipeline bound to an LLM provider."""

    async def compute_stage(ctx: dict[str, Any]) -> dict[str, Any]:
        data = [float(x) for x in ctx["data"]]
        return {"stats": compute_summary_stats(data)}

    async def draft_stage(ctx: dict[str, Any]) -> dict[str, Any]:
        stats = ctx["stats"]
        numbers = ", ".join(f"{k}={v}" for k, v in stats.items())
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a one-paragraph data summary. Reuse every provided number exactly; never invent figures."
                ),
            },
            {"role": "user", "content": f"Numbers: {numbers}. Write the summary."},
        ]
        return {"draft": await provider.generate(messages)}

    return [
        Stage("compute", compute_stage, [NonEmptyGate("stats")], weight=30.0),
        Stage("draft", draft_stage, [NumbersPreservedGate("stats", "draft"), PlaceholderGate("draft")], weight=70.0),
    ]
