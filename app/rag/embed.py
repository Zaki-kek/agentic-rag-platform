"""Embedding providers (factory-selected).

`hash` is a deterministic, offline embedder (bag-of-tokens hashed into a fixed
dimension) so the whole stack runs with no API key. `openai` calls the real
embeddings endpoint. Both expose the same async interface and a `dim`.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from app.core import ProviderConfigError

if TYPE_CHECKING:
    from app.config import Settings


@runtime_checkable
class Embedder(Protocol):
    """Async text embedder with a fixed output dimension."""

    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic offline embedder: hashed bag-of-tokens, L2-normalised."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec: np.ndarray = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


class OpenAIEmbedder:
    """Real embeddings via the OpenAI API (lazy-imported)."""

    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self.dim = self._DIMS.get(model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the embedder selected in settings.embedder."""
    if settings.embedder == "hash":
        return HashEmbedder(dim=settings.hash_embedding_dim)
    if settings.embedder == "openai":
        if not settings.openai_api_key:
            raise ProviderConfigError("openai_api_key is required for the openai embedder")
        return OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.embedding_model)
    raise ValueError(f"Unknown embedder '{settings.embedder}' (use: hash, openai)")
