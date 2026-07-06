"""/telegram route: simulate a Telegram message through the assistant facade.

This exposes the transport-agnostic facade over HTTP so the Telegram delivery
path is demoable and testable without a live bot token or aiogram installed.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["telegram"])


class TelegramMessage(BaseModel):
    text: str = Field(..., min_length=1)


class TelegramReply(BaseModel):
    reply: str


@router.post("/telegram/message", response_model=TelegramReply)
async def telegram_message(payload: TelegramMessage, request: Request) -> TelegramReply:
    reply = await request.app.state.telegram_facade.handle(payload.text)
    return TelegramReply(reply=reply)
