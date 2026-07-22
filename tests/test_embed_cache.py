"""Tests for the in-process embedding cache (:mod:`app.rag.cache`)."""

from __future__ import annotations

from app.rag.cache import EmbeddingCache


class CountingEmbedder:
    """Deterministic mock embedder that counts how many texts it embedded.

    ``call_count`` tracks the number of ``embed`` invocations (provider calls);
    ``embedded_texts`` accumulates the texts it was actually asked to embed, so
    tests can assert that cached texts never reach the provider.
    """

    dim = 4

    def __init__(self) -> None:
        self.call_count = 0
        self.embedded_texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.embedded_texts.extend(texts)
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


async def test_repeated_text_hits_provider_once() -> None:
    cache = EmbeddingCache()
    embedder = CountingEmbedder()

    first = await cache.get_or_compute(embedder, ["hello world"], model_id="m")
    second = await cache.get_or_compute(embedder, ["hello world"], model_id="m")

    assert first == second
    # The second call is served entirely from the cache: provider untouched.
    assert embedder.call_count == 1
    assert embedder.embedded_texts == ["hello world"]


async def test_distinct_texts_increase_provider_work() -> None:
    cache = EmbeddingCache()
    embedder = CountingEmbedder()

    await cache.get_or_compute(embedder, ["alpha"], model_id="m")
    await cache.get_or_compute(embedder, ["beta"], model_id="m")
    await cache.get_or_compute(embedder, ["gamma"], model_id="m")

    assert embedder.call_count == 3
    assert embedder.embedded_texts == ["alpha", "beta", "gamma"]


async def test_only_misses_are_computed_and_order_preserved() -> None:
    cache = EmbeddingCache()
    embedder = CountingEmbedder()

    await cache.get_or_compute(embedder, ["a", "b"], model_id="m")
    # "a" and "b" are cached; only "c" is a miss on this call.
    result = await cache.get_or_compute(embedder, ["a", "c", "b"], model_id="m")

    assert embedder.embedded_texts == ["a", "b", "c"]
    expected = [
        [1.0, 0.0, 0.0, 0.0],  # "a"
        [1.0, 0.0, 0.0, 0.0],  # "c"
        [1.0, 0.0, 0.0, 0.0],  # "b"
    ]
    assert result == expected


async def test_same_text_different_model_does_not_collide() -> None:
    cache = EmbeddingCache()
    embedder = CountingEmbedder()

    await cache.get_or_compute(embedder, ["shared"], model_id="model-a")
    await cache.get_or_compute(embedder, ["shared"], model_id="model-b")

    # Different model ids are distinct cache keys -> two provider calls.
    assert embedder.call_count == 2
    assert len(cache) == 2


async def test_duplicate_within_single_call_computed_once() -> None:
    cache = EmbeddingCache()
    embedder = CountingEmbedder()

    result = await cache.get_or_compute(embedder, ["dup", "dup", "dup"], model_id="m")

    # The provider sees "dup" a single time even though it appears three times.
    assert embedder.embedded_texts == ["dup"]
    assert result == [[3.0, 0.0, 0.0, 0.0]] * 3
