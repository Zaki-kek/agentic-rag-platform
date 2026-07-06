"""Transport-agnostic Telegram assistant facade (no aiogram dependency)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.core import get_logger

logger = get_logger(__name__)

# An answer function maps user text to (reply_text, source_snippets).
AnswerFn = Callable[[str], Awaitable[tuple[str, list[str]]]]

_EMPTY_INPUT_REPLY = "Please send a question and I'll do my best to answer it."
_SOURCES_HEADER = "Sources:"


class AssistantFacade:
    """Format assistant replies for Telegram, independent of any bot library.

    The facade owns presentation only: it delegates the actual answering to an
    injected async ``answer_fn`` and turns its ``(text, sources)`` result into a
    single Telegram-ready string. Keeping it free of any transport import makes it
    trivial to unit-test and reusable across bot frameworks.
    """

    def __init__(self, answer_fn: AnswerFn) -> None:
        """Initialize the facade.

        Args:
            answer_fn: Async callable returning a ``(reply_text, sources)`` tuple
                for a given user message. ``sources`` is a list of citation or
                source snippets that will be appended under a "Sources:" header.
        """
        self._answer_fn = answer_fn

    async def handle(self, text: str) -> str:
        """Answer a user message and format it as a Telegram-ready reply.

        Empty or whitespace-only input is handled gracefully by returning a short
        prompt asking the user for a question, without calling ``answer_fn``.

        Args:
            text: The raw incoming message text from the user.

        Returns:
            A single string: the answer, optionally followed by a "Sources:" list
            when the answer function returned any source snippets.
        """
        if not text or not text.strip():
            logger.debug("Received empty Telegram message; returning prompt-for-input reply")
            return _EMPTY_INPUT_REPLY

        answer, sources = await self._answer_fn(text)
        return self._format(answer, sources)

    @staticmethod
    def _format(answer: str, sources: list[str]) -> str:
        """Combine an answer and its source snippets into one reply string."""
        body = answer.strip()
        snippets = [s.strip() for s in sources if s and s.strip()]
        if not snippets:
            return body
        lines = [body, "", _SOURCES_HEADER]
        lines.extend(f"{i}. {snippet}" for i, snippet in enumerate(snippets, start=1))
        return "\n".join(lines)
