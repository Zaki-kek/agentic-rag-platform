"""Tests for batched embedding (:func:`app.rag.embed.embed_in_batches`)."""

from __future__ import annotations

import math

import pytest

from app.rag.embed import HashEmbedder, embed_in_batches


class CountingHashEmbedder:
    """A :class:`HashEmbedder` that also counts ``embed`` (provider) calls."""

    def __init__(self, dim: int = 32) -> None:
        self._inner = HashEmbedder(dim=dim)
        self.dim = dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return await self._inner.embed(texts)


async def test_hundred_texts_batch_size_32_makes_exactly_four_calls() -> None:
    embedder = CountingHashEmbedder(dim=32)
    texts = [f"document number {i}" for i in range(100)]

    result = await embed_in_batches(embedder, texts, batch_size=32)

    # ceil(100 / 32) == 4 batches -> exactly 4 provider calls.
    assert math.ceil(100 / 32) == 4
    assert embedder.call_count == 4
    assert len(result) == 100


async def test_batched_equals_non_batched_byte_for_byte() -> None:
    texts = [f"document number {i}" for i in range(100)]

    batched = await embed_in_batches(CountingHashEmbedder(dim=32), texts, batch_size=32)
    non_batched = await HashEmbedder(dim=32).embed(texts)

    assert batched == non_batched


async def test_empty_input_returns_empty_without_calling_provider() -> None:
    embedder = CountingHashEmbedder(dim=8)
    assert await embed_in_batches(embedder, [], batch_size=32) == []
    assert embedder.call_count == 0


async def test_batch_larger_than_input_is_single_call() -> None:
    embedder = CountingHashEmbedder(dim=8)
    texts = ["one", "two", "three"]

    result = await embed_in_batches(embedder, texts, batch_size=32)

    assert embedder.call_count == 1
    assert len(result) == 3


async def test_non_positive_batch_size_raises() -> None:
    embedder = CountingHashEmbedder(dim=8)
    with pytest.raises(ValueError):
        await embed_in_batches(embedder, ["a"], batch_size=0)
