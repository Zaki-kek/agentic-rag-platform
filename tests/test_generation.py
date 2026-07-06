"""Tests for the generation orchestrator, gates, compute and checkpoints."""

from __future__ import annotations

import pytest

from app.generation.checkpoint import FileCheckpointStore, InMemoryCheckpointStore
from app.generation.compute import compute_summary_stats
from app.generation.gates import NonEmptyGate, NumbersPreservedGate
from app.generation.models import JobState, Stage
from app.generation.orchestrator import GenerationOrchestrator


def test_compute_summary_stats() -> None:
    s = compute_summary_stats([10, 20, 30])
    assert s == {"count": 3.0, "mean": 20.0, "median": 20.0, "min": 10.0, "max": 30.0, "stdev": 10.0}


def test_compute_rejects_empty() -> None:
    with pytest.raises(ValueError):
        compute_summary_stats([])


def test_numbers_preserved_gate() -> None:
    gate = NumbersPreservedGate("stats", "draft")
    assert gate.check({"stats": {"mean": 20.0}, "draft": "the mean is 20.0"}).passed
    assert not gate.check({"stats": {"mean": 20.0}, "draft": "no figures here"}).passed


@pytest.mark.asyncio
async def test_orchestrator_runs_all_stages() -> None:
    executed: list[str] = []

    async def s1(_: dict) -> dict:
        executed.append("s1")
        return {"a": 1}

    async def s2(ctx: dict) -> dict:
        executed.append("s2")
        return {"b": ctx["a"] + 1}

    store = InMemoryCheckpointStore()
    state = await GenerationOrchestrator([Stage("s1", s1), Stage("s2", s2)], store).run("job1", {})

    assert state.status == "done"
    assert executed == ["s1", "s2"]
    assert state.context["b"] == 2
    assert state.completed_stages == ["s1", "s2"]
    assert state.progress == 1.0


@pytest.mark.asyncio
async def test_orchestrator_resumes_and_skips_completed() -> None:
    executed: list[str] = []

    async def s1(_: dict) -> dict:
        executed.append("s1")
        return {"a": 1}

    async def s2(_: dict) -> dict:
        executed.append("s2")
        return {"b": 2}

    store = InMemoryCheckpointStore()
    store.save(JobState(job_id="job2", completed_stages=["s1"], context={"a": 1}))

    state = await GenerationOrchestrator([Stage("s1", s1), Stage("s2", s2)], store).run("job2", {})

    assert state.status == "done"
    assert executed == ["s2"]  # the completed stage was skipped


@pytest.mark.asyncio
async def test_orchestrator_fails_on_unsatisfiable_gate() -> None:
    async def s1(_: dict) -> dict:
        return {"x": ""}

    store = InMemoryCheckpointStore()
    state = await GenerationOrchestrator([Stage("s1", s1, [NonEmptyGate("x")])], store, max_retries=1).run("job3", {})

    assert state.status == "failed"
    assert "empty" in (state.error or "")


def test_file_checkpoint_roundtrip_and_repair(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    store.save(JobState(job_id="j", completed_stages=["s1"], context={"a": 1}))

    loaded = store.load("j")
    assert loaded is not None
    assert loaded.completed_stages == ["s1"]

    (tmp_path / "j.json").write_text("{ not valid json", encoding="utf-8")
    assert store.load("j") is None
    assert (tmp_path / "j.json.corrupt").exists()
