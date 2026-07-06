"""LLM provider abstraction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# A chat message is a simple role/content dict, e.g. {"role": "user", "content": "hi"}.
Message = dict[str, str]


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal async chat-completion interface every provider implements."""

    name: str

    async def generate(self, messages: list[Message]) -> str:
        """Return the assistant reply for the given message list."""
        ...
