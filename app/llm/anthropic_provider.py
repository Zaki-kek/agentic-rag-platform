"""Anthropic (Claude) chat provider (lazy-imported so it is optional)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core import ProviderConfigError, get_logger
from app.llm.base import Message
from app.llm.factory import register_provider

if TYPE_CHECKING:
    from app.config import Settings

logger = get_logger(__name__)


class AnthropicProvider:
    """Async wrapper over the Anthropic Messages API.

    System messages are folded into the top-level `system` parameter, as the
    Messages API expects only user/assistant turns in `messages`.
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import AsyncAnthropic  # local import keeps the dep optional

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(self, messages: list[Message]) -> str:
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=turns,  # type: ignore[arg-type]
        )
        parts = [block.text for block in resp.content if block.type == "text"]
        return "".join(parts).strip()


@register_provider("anthropic")
def _build_anthropic(settings: Settings) -> AnthropicProvider:
    if not settings.anthropic_api_key:
        raise ProviderConfigError("anthropic_api_key is not set (export ANTHROPIC_API_KEY)")
    model = settings.llm_model if settings.llm_model.startswith("claude") else "claude-3-5-sonnet-latest"
    return AnthropicProvider(api_key=settings.anthropic_api_key, model=model)
