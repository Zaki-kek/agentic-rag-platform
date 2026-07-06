"""aiogram transport adapter (lazy-imported so aiogram stays optional)."""

from __future__ import annotations

from typing import Any

from app.core import AppError, get_logger
from app.telegram.facade import AssistantFacade

logger = get_logger(__name__)

_MISSING_AIOGRAM_MSG = (
    "aiogram is required to run the Telegram bot but is not installed. Install it with: pip install aiogram"
)


def build_dispatcher(facade: AssistantFacade, token: str) -> Any:
    """Build an aiogram dispatcher wired to delegate messages to the facade.

    aiogram is imported lazily inside this function so the package and its tests
    have no hard dependency on it. A single message handler forwards each message's
    text to :meth:`AssistantFacade.handle` and replies with the formatted result.

    Args:
        facade: The transport-agnostic facade that produces formatted replies.
        token: The Telegram Bot API token used to construct the ``Bot`` instance.

    Returns:
        A configured ``aiogram.Dispatcher`` with the message handler registered.
        The associated ``aiogram.Bot`` is attached as ``dispatcher["bot"]`` so the
        caller can start polling (e.g. ``await dispatcher.start_polling(bot)``).

    Raises:
        AppError: If aiogram is not installed.
    """
    try:
        from aiogram import Bot, Dispatcher  # local import keeps the dep optional
        from aiogram.types import Message as TelegramMessage
    except ImportError as exc:
        logger.error("aiogram import failed: %s", exc)
        raise AppError(_MISSING_AIOGRAM_MSG, status_code=500) from exc

    bot = Bot(token=token)
    dispatcher = Dispatcher()

    @dispatcher.message()
    async def _on_message(message: TelegramMessage) -> None:
        """Delegate an incoming Telegram message to the facade and reply."""
        reply = await facade.handle(message.text or "")
        await message.answer(reply)

    dispatcher["bot"] = bot
    logger.info("Telegram dispatcher built and message handler registered")
    return dispatcher
