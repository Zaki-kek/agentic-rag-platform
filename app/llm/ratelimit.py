"""Rate-limit-aware, retrying LLM provider wrapper."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.core import get_logger
from app.llm.base import Message

if TYPE_CHECKING:
    from app.llm.base import LLMProvider

logger = get_logger(__name__)


class LLMTimeoutError(TimeoutError):
    """Raised when a single ``generate`` call exceeds its per-call timeout.

    Subclasses the builtin ``TimeoutError`` (which ``asyncio.wait_for`` raises)
    so callers can catch either type, while carrying a message that names the
    provider and the budget that was exceeded rather than failing silently.
    """


class RateLimitedProvider:
    """Wrap an ``LLMProvider`` with bounded concurrency, retries and timeouts.

    This is a neutral production pattern for living within a provider's
    request limits: a semaphore caps how many ``generate`` calls run
    concurrently, each call is bounded by a per-call timeout, and transient
    failures are retried with exponential backoff. The wrapper itself satisfies
    the ``LLMProvider`` protocol, so it can be dropped in anywhere an
    ``LLMProvider`` is expected.

    Attributes:
        name: Mirrors the inner provider's name so the wrapper is transparent.
    """

    def __init__(
        self,
        inner: LLMProvider,
        max_concurrency: int = 4,
        max_retries: int = 2,
        backoff_base: float = 0.0,
        timeout_seconds: float | None = 60.0,
    ) -> None:
        """Initialize the wrapper.

        Args:
            inner: The provider whose ``generate`` calls are guarded.
            max_concurrency: Maximum number of concurrent ``generate`` calls.
                Must be at least 1.
            max_retries: Number of additional attempts after the first failure.
                ``0`` means a single attempt with no retries.
            backoff_base: Base delay in seconds for exponential backoff; the
                sleep before retry ``attempt`` is ``backoff_base * 2 ** attempt``.
                A value of ``0.0`` disables real sleeping (useful in tests).
            timeout_seconds: Per-call wall-clock budget for a single inner
                ``generate`` attempt. A slow call is cancelled and raises
                ``LLMTimeoutError`` instead of hanging. ``None`` disables the
                timeout. Must be positive when set.

        Raises:
            ValueError: If ``max_concurrency`` is less than 1, ``max_retries``
                is negative, ``backoff_base`` is negative, or ``timeout_seconds``
                is set to a non-positive value.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_base < 0:
            raise ValueError("backoff_base must be >= 0")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self._inner = inner
        self.name = inner.name
        self._max_concurrency = max_concurrency
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._timeout_seconds = timeout_seconds
        # Created lazily on first use so it binds to the running event loop.
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return the concurrency semaphore, creating it on first use.

        Creating the semaphore lazily binds it to whichever event loop is
        running when ``generate`` is first awaited, avoiding cross-loop issues.

        Returns:
            The shared ``asyncio.Semaphore`` for this wrapper.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._semaphore

    async def generate(self, messages: list[Message]) -> str:
        """Generate a reply, bounding concurrency and retrying on failure.

        Args:
            messages: The chat messages to forward to the inner provider.

        Returns:
            The inner provider's reply on the first successful attempt.

        Raises:
            Exception: Re-raises the last exception from the inner provider if
                every attempt fails.
        """
        semaphore = self._get_semaphore()
        async with semaphore:
            last_exc: BaseException | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    return await self._call_inner(messages)
                except Exception as exc:  # noqa: BLE001 - retry any provider error
                    last_exc = exc
                    if attempt >= self._max_retries:
                        break
                    delay = self._backoff_base * 2**attempt
                    logger.warning(
                        "provider %s failed (attempt %d/%d): %s; retrying in %.3fs",
                        self.name,
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                        delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
            assert last_exc is not None  # loop ran at least once
            raise last_exc

    async def _call_inner(self, messages: list[Message]) -> str:
        """Run one inner ``generate`` attempt, bounded by the per-call timeout.

        Args:
            messages: The chat messages to forward to the inner provider.

        Returns:
            The inner provider's reply.

        Raises:
            LLMTimeoutError: If the attempt exceeds ``timeout_seconds``.
        """
        if self._timeout_seconds is None:
            return await self._inner.generate(messages)
        try:
            return await asyncio.wait_for(
                self._inner.generate(messages), timeout=self._timeout_seconds
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:  # noqa: UP041 - be explicit
            raise LLMTimeoutError(
                f"provider {self.name} timed out after {self._timeout_seconds:g}s"
            ) from exc
