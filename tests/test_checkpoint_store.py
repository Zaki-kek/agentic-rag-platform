"""Tests for checkpoint store persistence and corruption handling.

These focus on the *cross-instance* durability contract of
``FileCheckpointStore``: state written by one store instance must be readable
by a fresh instance pointed at the same directory (i.e. survives a process
restart), and a corrupt on-disk checkpoint must degrade to ``None`` while
quarantining the bad file instead of crashing the worker.
"""

from __future__ import annotations

import pytest

from app.generation.checkpoint import (
    FileCheckpointStore,
    InMemoryCheckpointStore,
    build_checkpoint_store,
)
from app.generation.models import JobState


def test_file_store_persists_across_instances(tmp_path) -> None:
    """save() on instance A and load() on a NEW instance B return equal state."""
    state = JobState(
        job_id="job-1",
        status="running",
        completed_stages=["outline", "draft"],
        current_stage="review",
        context={"topic": "quarterly report", "tokens": 1234},
        progress=0.6,
    )

    store_a = FileCheckpointStore(tmp_path)
    store_a.save(state)

    # Fresh instance, same directory: simulates a process restart.
    store_b = FileCheckpointStore(tmp_path)
    loaded = store_b.load("job-1")

    assert loaded is not None
    assert loaded == state


def test_file_store_load_missing_returns_none(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    assert store.load("does-not-exist") is None


def test_file_store_corrupt_file_returns_none_and_quarantines(tmp_path) -> None:
    """A corrupt checkpoint yields None and is moved aside to .corrupt."""
    store = FileCheckpointStore(tmp_path)
    store.save(JobState(job_id="job-2", completed_stages=["outline"]))

    checkpoint = tmp_path / "job-2.json"
    corrupt = tmp_path / "job-2.json.corrupt"
    checkpoint.write_text("{ this is not valid json", encoding="utf-8")

    assert store.load("job-2") is None
    assert corrupt.exists()
    assert not checkpoint.exists()


def test_build_checkpoint_store_selects_backend(tmp_path) -> None:
    assert isinstance(build_checkpoint_store("memory"), InMemoryCheckpointStore)
    assert isinstance(build_checkpoint_store("file", tmp_path), FileCheckpointStore)


def test_build_checkpoint_store_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown checkpoint store"):
        build_checkpoint_store("redis")
