"""/guardrails routes: PII redaction and citation validation."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.guardrails import redact_pii, validate_citations

router = APIRouter(tags=["guardrails"])


class RedactRequest(BaseModel):
    text: str = Field(..., min_length=1)


class RedactResponse(BaseModel):
    redacted: str


@router.post("/guardrails/redact", response_model=RedactResponse)
async def redact(payload: RedactRequest) -> RedactResponse:
    return RedactResponse(redacted=redact_pii(payload.text))


class CitationsRequest(BaseModel):
    answer: str = Field(..., min_length=1)
    num_sources: int = Field(..., ge=0)


class CitationsResponse(BaseModel):
    problems: list[str]


@router.post("/guardrails/citations", response_model=CitationsResponse)
async def citations(payload: CitationsRequest) -> CitationsResponse:
    return CitationsResponse(problems=validate_citations(payload.answer, payload.num_sources))
