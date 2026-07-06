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


@router.post("/jobs", response_model=JobState)
async def create_job(payload: JobRequest, request: Request) -> JobState:
    orchestrator = request.app.state.generation_orchestrator
    job_id = uuid.uuid4().hex
    return await orchestrator.run(job_id, {"data": payload.data})


@router.get("/jobs/{job_id}", response_model=JobState)
async def get_job(job_id: str, request: Request) -> JobState:
    state = request.app.state.checkpoint_store.load(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    return state
