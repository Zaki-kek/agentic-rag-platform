"""GigaChat (Sber) provider stub.

Left as a clearly-marked extension point: RU employers value GigaChat /
YandexGPT support. The registry pattern means wiring a real client here needs
no changes anywhere else. Implement OAuth + the chat call, then drop the stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core import ProviderConfigError
from app.llm.base import Message
from app.llm.factory import register_provider

if TYPE_CHECKING:
    from app.config import Settings


class GigaChatProvider:
    """Placeholder GigaChat client (not wired in this build)."""

    name = "gigachat"

    def __init__(self, credentials: str) -> None:
        self._credentials = credentials

    async def generate(self, messages: list[Message]) -> str:
        raise NotImplementedError(
            "GigaChat client is a stub. Implement OAuth token exchange and the "
            "chat endpoint here, then this provider works via the same factory."
        )


@register_provider("gigachat")
def _build_gigachat(settings: Settings) -> GigaChatProvider:
    if not settings.gigachat_credentials:
        raise ProviderConfigError("gigachat_credentials is not set")
    return GigaChatProvider(credentials=settings.gigachat_credentials)
