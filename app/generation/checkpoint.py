"""Checkpoint stores for resumable generation jobs.

Mirrors a hard-won production pattern: job state is persisted after every stage
so a crash or redeploy resumes from the last completed stage instead of redoing
(and re-charging) work. The file store self-repairs a corrupt checkpoint rather
than crashing the worker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from app.core import get_logger
from app.generation.models import JobState

logger = get_logger(__name__)


class CheckpointStore(Protocol):
    """Persist and load job state by job id."""

    def save(self, state: JobState) -> None: ...

    def load(self, job_id: str) -> JobState | None: ...


class InMemoryCheckpointStore:
    """Process-local checkpoint store (tests, single-process demo)."""

    def __init__(self) -> None:
        self._states: dict[str, str] = {}

    def save(self, state: JobState) -> None:
        self._states[state.job_id] = state.model_dump_json()

    def load(self, job_id: str) -> JobState | None:
        raw = self._states.get(job_id)
        return JobState.model_validate_json(raw) if raw else None


class FileCheckpointStore:
    """JSON-file checkpoint store with atomic writes and self-repair."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.json"

    def save(self, state: JobState) -> None:
        path = self._path(state.job_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic on the same filesystem

    def load(self, job_id: str) -> JobState | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        try:
            return JobState.model_validate_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            corrupt = path.with_suffix(".json.corrupt")
            os.replace(path, corrupt)
            logger.warning("Corrupt checkpoint for %s moved to %s", job_id, corrupt.name)
            return None


def build_checkpoint_store(kind: str, directory: str | Path = ".checkpoints") -> CheckpointStore:
    """Build a checkpoint store by kind.

    Args:
        kind: ``memory`` (process-local) or ``file`` (persisted JSON).
        directory: Base directory used when ``kind == "file"``.

    Returns:
        A concrete :class:`CheckpointStore` implementation.

    Raises:
        ValueError: When ``kind`` is not a known store type.
    """
    if kind == "memory":
        return InMemoryCheckpointStore()
    if kind == "file":
        return FileCheckpointStore(directory)
    raise ValueError(f"unknown checkpoint store: {kind!r} (expected 'memory' or 'file')")
