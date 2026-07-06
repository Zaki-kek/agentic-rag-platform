"""HTTP API routes."""

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.guardrails import router as guardrails_router
from app.api.jobs import router as jobs_router
from app.api.payments import router as payments_router
from app.api.stream import router as stream_router
from app.api.telegram import router as telegram_router

__all__ = [
    "agent_router",
    "chat_router",
    "documents_router",
    "guardrails_router",
    "jobs_router",
    "payments_router",
    "stream_router",
    "telegram_router",
]
