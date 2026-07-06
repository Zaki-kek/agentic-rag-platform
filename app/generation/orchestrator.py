"""Checkpointed, resumable, gated multi-stage generation orchestrator."""

from __future__ import annotations

from typing import Any

from app.core import get_logger
from app.generation.checkpoint import CheckpointStore
from app.generation.models import JobState, Stage

logger = get_logger(__name__)


class StageGateError(RuntimeError):
    """Raised when a stage cannot pass its gates within the retry budget."""


class GenerationOrchestrator:
    """Run an ordered list of stages with gates, checkpointing each success.

    Resumability: on rerun for the same job id, completed stages are skipped and
    their context is restored from the checkpoint - a crashed job resumes instead
    of restarting. Reliability: each stage retries until its gates pass or the
    budget is exhausted, then the job is marked failed (never silently wrong).
    """

    def __init__(self, stages: list[Stage], store: CheckpointStore, max_retries: int = 2) -> None:
        self._stages = stages
        self._store = store
        self._max_retries = max_retries

    async def run(self, job_id: str, initial_context: dict[str, Any] | None = None) -> JobState:
        state = self._store.load(job_id) or JobState(job_id=job_id, context=initial_context or {})
        state.status = "running"
        self._store.save(state)

        for stage in self._stages:
            if stage.name in state.completed_stages:
                logger.info("Job %s: skipping completed stage '%s'", job_id, stage.name)
                continue
            state.current_stage = stage.name
            try:
                await self._run_stage_with_gates(stage, state)
            except Exception as exc:  # noqa: BLE001 - record and surface as failed job
                state.status = "failed"
                state.error = f"{stage.name}: {exc}"
                self._store.save(state)
                logger.error("Job %s failed at '%s': %s", job_id, stage.name, exc)
                return state
            state.completed_stages.append(stage.name)
            state.progress = self._progress(state.completed_stages)
            self._store.save(state)

        state.status = "done"
        state.current_stage = None
        state.progress = 1.0
        self._store.save(state)
        return state

    def _progress(self, completed: list[str]) -> float:
        """Fraction of total stage weight completed so far (0.0 - 1.0)."""
        total = sum(s.weight for s in self._stages) or 1.0
        done = sum(s.weight for s in self._stages if s.name in completed)
        return round(done / total, 3)

    async def _run_stage_with_gates(self, stage: Stage, state: JobState) -> None:
        last_reason = ""
        for attempt in range(1, self._max_retries + 2):
            output = await stage.run(state.context)
            candidate = {**state.context, **output}
            failures = [r.reason for g in stage.gates if not (r := g.check(candidate)).passed]
            if not failures:
                state.context = candidate
                return
            last_reason = "; ".join(failures)
            logger.warning("Job %s stage '%s' attempt %d gate fail: %s", state.job_id, stage.name, attempt, last_reason)
        raise StageGateError(last_reason or "gate check failed")
