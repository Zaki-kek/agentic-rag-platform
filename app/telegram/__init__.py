"""Telegram delivery: a pure formatting facade plus a lazy aiogram adapter."""

from app.telegram.adapter import build_dispatcher
from app.telegram.facade import AssistantFacade

__all__ = [
    "AssistantFacade",
    "build_dispatcher",
]
