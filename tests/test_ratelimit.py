"""Tests for the rate-limit-aware retrying provider wrapper."""

from __future__ import annotations

import asyncio

import pytest

from app.llm.base import Message
from app.llm.ratelimit import LLMTimeoutError, RateLimitedProvider


class SlowProvider:
    """Sleeps longer than the caller's timeout budget before replying."""

    name = "slow"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    async def generate(self, messages: list[Message]) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return "too late"


class FlakyProvider:
    """Fails the first ``fail_times`` calls, then returns a fixed reply."""

    name = "flaky"

    def __init__(self, fail_times: int, reply: str = "ok") -> None:
        self.fail_times = fail_times
        self.reply = reply
        self.calls = 0

    async def generate(self, messages: list[Message]) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"transient failure {self.calls}")
        return self.reply


class AlwaysFailsProvider:
    """Always raises, counting how many times it was invoked."""

    name = "always-fails"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages: list[Message]) -> str:
        self.calls += 1
        raise RuntimeError("permanent failure")


class ConcurrencyProbeProvider:
    """Records the maximum number of concurrent in-flight ``generate`` calls."""

    name = "probe"

    def __init__(self, hold: float = 0.01) -> None:
        self.hold = hold
        self.current = 0
        self.max_observed = 0
        self._lock = asyncio.Lock()

    async def generate(self, messages: list[Message]) -> str:
        async with self._lock:
            self.current += 1
            self.max_observed = max(self.max_observed, self.current)
        try:
            await asyncio.sleep(self.hold)
            return "done"
        finally:
            async with self._lock:
                self.current -= 1


async def test_name_mirrors_inner() -> None:
    inner = FlakyProvider(fail_times=0)
    wrapped = RateLimitedProvider(inner)
    assert wrapped.name == inner.name == "flaky"


async def test_retries_then_succeeds() -> None:
    inner = FlakyProvider(fail_times=2, reply="recovered")
    wrapped = RateLimitedProvider(inner, max_retries=2, backoff_base=0.0)

    result = await wrapped.generate([{"role": "user", "content": "hi"}])

    assert result == "recovered"
    assert inner.calls == 3  # two failures + one success


async def test_exhausts_retries_and_propagates() -> None:
    inner = AlwaysFailsProvider()
    wrapped = RateLimitedProvider(inner, max_retries=2, backoff_base=0.0)

    with pytest.raises(RuntimeError, match="permanent failure"):
        await wrapped.generate([{"role": "user", "content": "hi"}])

    assert inner.calls == 3  # initial attempt + 2 retries


async def test_no_retries_single_attempt() -> None:
    inner = AlwaysFailsProvider()
    wrapped = RateLimitedProvider(inner, max_retries=0, backoff_base=0.0)

    with pytest.raises(RuntimeError):
        await wrapped.generate([{"role": "user", "content": "hi"}])

    assert inner.calls == 1


async def test_concurrency_is_capped() -> None:
    inner = ConcurrencyProbeProvider(hold=0.01)
    max_concurrency = 3
    wrapped = RateLimitedProvider(inner, max_concurrency=max_concurrency, backoff_base=0.0)

    messages: list[Message] = [{"role": "user", "content": "x"}]
    results = await asyncio.gather(*(wrapped.generate(messages) for _ in range(20)))

    assert results == ["done"] * 20
    assert inner.max_observed <= max_concurrency
    assert inner.max_observed >= 1


async def test_concurrency_one_serializes() -> None:
    inner = ConcurrencyProbeProvider(hold=0.005)
    wrapped = RateLimitedProvider(inner, max_concurrency=1, backoff_base=0.0)

    messages: list[Message] = [{"role": "user", "content": "x"}]
    await asyncio.gather(*(wrapped.generate(messages) for _ in range(10)))

    assert inner.max_observed == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_concurrency": 0}, "max_concurrency"),
        ({"max_retries": -1}, "max_retries"),
        ({"backoff_base": -1.0}, "backoff_base"),
    ],
)
async def test_invalid_arguments_rejected(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RateLimitedProvider(FlakyProvider(fail_times=0), **kwargs)


async def test_slow_call_times_out() -> None:
    inner = SlowProvider(delay=1.0)
    # No retries so we only wait one short budget, not several.
    wrapped = RateLimitedProvider(
        inner, max_retries=0, backoff_base=0.0, timeout_seconds=0.02
    )

    with pytest.raises(LLMTimeoutError, match="timed out after"):
        await wrapped.generate([{"role": "user", "content": "hi"}])

    assert inner.calls == 1


async def test_timeout_error_is_a_timeouterror() -> None:
    inner = SlowProvider(delay=1.0)
    wrapped = RateLimitedProvider(
        inner, max_retries=0, backoff_base=0.0, timeout_seconds=0.02
    )

    # LLMTimeoutError subclasses builtin TimeoutError, so generic callers catch it.
    with pytest.raises(TimeoutError):
        await wrapped.generate([{"role": "user", "content": "hi"}])


async def test_fast_call_within_timeout_succeeds() -> None:
    inner = FlakyProvider(fail_times=0, reply="on time")
    wrapped = RateLimitedProvider(inner, timeout_seconds=5.0)

    result = await wrapped.generate([{"role": "user", "content": "hi"}])

    assert result == "on time"


async def test_none_timeout_disables_budget() -> None:
    inner = SlowProvider(delay=0.01)
    wrapped = RateLimitedProvider(inner, timeout_seconds=None)

    result = await wrapped.generate([{"role": "user", "content": "hi"}])

    assert result == "too late"


async def test_non_positive_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        RateLimitedProvider(FlakyProvider(fail_times=0), timeout_seconds=0)


async def test_satisfies_llm_provider_protocol() -> None:
    from app.llm.base import LLMProvider

    wrapped = RateLimitedProvider(FlakyProvider(fail_times=0))
    assert isinstance(wrapped, LLMProvider)
