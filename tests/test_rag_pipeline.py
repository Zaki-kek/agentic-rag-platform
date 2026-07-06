"""Integration test for the RAG pipeline (offline: hash embedder + memory store)."""

from __future__ import annotations

import pytest

from app.rag.embed import HashEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.store import InMemoryVectorStore

pytestmark = pytest.mark.asyncio

_DOC = (
    "The capybara is the largest living rodent native to South America. "
    "Photosynthesis is the process by which green plants convert sunlight into energy. "
    "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France."
)


async def test_ingest_and_retrieve_relevant_chunk() -> None:
    pipeline = RagPipeline(HashEmbedder(dim=256), InMemoryVectorStore(), chunk_size=80, chunk_overlap=20)
    chunks = await pipeline.ingest("facts.txt", _DOC.encode("utf-8"))
    assert chunks >= 1

    hits = await pipeline.retrieve("Where is the Eiffel Tower located?", k=1)
    assert hits
    assert "Eiffel Tower" in hits[0].text


async def test_retrieve_on_empty_store_returns_nothing() -> None:
    pipeline = RagPipeline(HashEmbedder(dim=64), InMemoryVectorStore())
    assert await pipeline.retrieve("anything", k=3) == []
