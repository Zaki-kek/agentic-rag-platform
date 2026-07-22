"""/jobs routes: start a generation job and poll its status."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.generation.models import JobState

router = APIRouter(tags=["generation"])


class JobRequest(BaseModel):
    """Input for the demo report pipeline."""

    data: list[float] = Field(..., min_length=1, description="Numbers to summarize")
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Optional client-supplied key. Repeating a request with the same "
            "key returns the already-created job instead of starting a new one."
        ),
    )


@router.post("/jobs", response_model=JobState)
async def create_job(payload: JobRequest, request: Request) -> JobState:
    orchestrator = request.app.state.generation_orchestrator
    checkpoint_store = request.app.state.checkpoint_store
    idempotency = request.app.state.job_idempotency

    key = payload.idempotency_key
    if key is not None:
        existing_id = idempotency.get(key)
        if existing_id is not None:
            existing = checkpoint_store.load(existing_id)
            if existing is not None:
                return existing

    job_id = uuid.uuid4().hex
    state = await orchestrator.run(job_id, {"data": payload.data})
    if key is not None:
        idempotency[key] = job_id
    return state


@router.get("/jobs/{job_id}", response_model=JobState)
async def get_job(job_id: str, request: Request) -> JobState:
    state = request.app.state.checkpoint_store.load(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    return state
