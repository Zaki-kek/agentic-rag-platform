"""Generation job state and pipeline stage models."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["pending", "running", "done", "failed"]

# A stage transforms the job context and returns the keys it produced.
StageFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class JobState(BaseModel):
    """Serializable state of a generation job (also the checkpoint payload)."""

    job_id: str
    status: JobStatus = "pending"
    completed_stages: list[str] = Field(default_factory=list)
    current_stage: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    progress: float = 0.0


@dataclass
class GateResult:
    """Outcome of a quality gate check."""

    passed: bool
    reason: str = ""


@dataclass
class Stage:
    """A pipeline stage: an async function plus the gates its output must pass."""

    name: str
    run: StageFn
    gates: list[Any] = field(default_factory=list)  # list[QualityGate]
    weight: float = 1.0  # relative contribution to job progress (e.g. 10/40/50)
