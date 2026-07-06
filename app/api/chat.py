"""/chat route: answer a question over the indexed documents."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service = request.app.state.chat_service
    return await service.answer(payload.message, payload.top_k)
