"""Echo provider: deterministic, dependency-free, for offline demo and tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.llm.base import Message
from app.llm.factory import register_provider

if TYPE_CHECKING:
    from app.config import Settings


class EchoProvider:
    """Returns a grounded-looking answer built from the prompt, no network."""

    name = "echo"

    async def generate(self, messages: list[Message]) -> str:
        user_turns = [m["content"] for m in messages if m.get("role") == "user"]
        question = user_turns[-1] if user_turns else ""
        has_context = any("context" in m.get("content", "").lower() for m in messages)
        prefix = "Based on the retrieved context, " if has_context else ""
        return f"{prefix}here is an answer to: {question.strip()[:500]}"


@register_provider("echo")
def _build_echo(_: Settings) -> EchoProvider:
    return EchoProvider()
