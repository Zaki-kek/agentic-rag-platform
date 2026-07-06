"""/chat/stream route: stream a grounded answer as Server-Sent Events (SSE)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest

router = APIRouter(tags=["chat"])


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Answer a question and stream it incrementally as SSE token + done events.

    The answer is computed via the chat service and streamed in word chunks;
    swap in provider-native token streaming for true incremental generation.
    """
    service = request.app.state.chat_service

    async def generate() -> AsyncIterator[str]:
        response = await service.answer(payload.message, payload.top_k)
        for index, word in enumerate(response.answer.split(" ")):
            token = word if index == 0 else f" {word}"
            yield _sse("token", {"text": token})
        yield _sse(
            "done",
            {"citations": [c.model_dump() for c in response.citations], "provider": response.provider},
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
