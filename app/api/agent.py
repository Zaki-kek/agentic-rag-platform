"""/agent route: answer a question with the tool-using agent."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.agent import AgentResult

router = APIRouter(tags=["agent"])


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Question for the agent")


@router.post("/agent", response_model=AgentResult)
async def run_agent(payload: AgentRequest, request: Request) -> AgentResult:
    return await request.app.state.agent.run(payload.message)
