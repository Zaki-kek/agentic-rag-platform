"""In-process embedding cache to avoid re-embedding repeated text.

Retrieval workloads re-embed the same strings constantly: a document may
contain a paragraph twice, or the same question may be asked again. Calling a
paid embeddings endpoint each time is wasteful, so :class:`EmbeddingCache`
memoises embeddings keyed by text and model. It is deliberately minimal -
zero external dependencies, a plain in-memory dict, no eviction - suited to a
single process. Cache keys include the model id so vectors from different
models never collide.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.embed import Embedder


def _cache_key(text: str, model_id: str) -> str:
    """Build the cache key for a piece of text under a given model.

    Args:
        text: The text whose embedding is being cached.
        model_id: Identifier of the embedding model/provider. Keeps vectors
            from different models from colliding.

    Returns:
        A string of the form ``"<sha1(text)>:<model_id>"``.
    """
    return f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}:{model_id}"


class EmbeddingCache:
    """Memoises text embeddings per model in an in-process dictionary."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def __len__(self) -> int:
        """Return the number of cached (text, model) embeddings."""
        return len(self._store)

    async def get_or_compute(
        self, embedder: Embedder, texts: list[str], model_id: str
    ) -> list[list[float]]:
        """Return embeddings for ``texts``, computing only the cache misses.

        Cached texts are served from memory; the remaining (miss) texts are
        embedded in a single ``embedder.embed`` call and stored. Results are
        returned in the same order as ``texts``, including for duplicate texts
        within the input (each position gets the shared cached vector).

        Args:
            embedder: The embedding provider to call for cache misses. Only
                invoked when at least one text is uncached, and never more than
                once per call.
            texts: The texts to embed, in order. Not mutated.
            model_id: Identifier of the embedding model, folded into the cache
                key so different models keep separate vectors.

        Returns:
            One embedding vector per input text, aligned with ``texts``.
        """
        keys = [_cache_key(text, model_id) for text in texts]

        missing_texts: list[str] = []
        missing_keys: list[str] = []
        seen_missing: set[str] = set()
        for text, key in zip(texts, keys, strict=True):
            if key in self._store or key in seen_missing:
                continue
            seen_missing.add(key)
            missing_texts.append(text)
            missing_keys.append(key)

        if missing_texts:
            computed = await embedder.embed(missing_texts)
            for key, vector in zip(missing_keys, computed, strict=True):
                self._store[key] = vector

        return [self._store[key] for key in keys]
