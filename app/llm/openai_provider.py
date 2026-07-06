"""OpenAI chat provider (lazy-imported so it is optional)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core import ProviderConfigError, get_logger
from app.llm.base import Message
from app.llm.factory import register_provider

if TYPE_CHECKING:
    from app.config import Settings

logger = get_logger(__name__)


class OpenAIProvider:
    """Thin async wrapper over the OpenAI Chat Completions API."""

    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI  # local import keeps the dep optional

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(self, messages: list[Message]) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        return (resp.choices[0].message.content or "").strip()


@register_provider("openai")
def _build_openai(settings: Settings) -> OpenAIProvider:
    if not settings.openai_api_key:
        raise ProviderConfigError("openai_api_key is not set (export OPENAI_API_KEY)")
    return OpenAIProvider(api_key=settings.openai_api_key, model=settings.llm_model)
