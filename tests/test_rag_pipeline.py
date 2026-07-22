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

# A document that repeats a whole paragraph so dedup has something to drop.
# Small chunks (chunk_size=40) split each sentence into its own chunk, and the
# duplicated sentence produces byte-identical chunks that dedup collapses.
_DOC_WITH_DUPES = (
    "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
    "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
    "Photosynthesis converts sunlight into chemical energy inside green plants."
)


class _CountingEmbedder:
    """Wraps HashEmbedder and counts how many times ``embed`` is invoked."""

    def __init__(self, dim: int = 256) -> None:
        self._inner = HashEmbedder(dim=dim)
        self.dim = dim
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return await self._inner.embed(texts)


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


async def test_skip_duplicates_avoids_reembedding_same_document() -> None:
    embedder = _CountingEmbedder(dim=128)
    pipeline = RagPipeline(embedder, InMemoryVectorStore(), chunk_size=80, chunk_overlap=20)
    data = _DOC.encode("utf-8")

    first = await pipeline.ingest("facts.txt", data, skip_duplicates=True)
    calls_after_first = embedder.calls
    assert first >= 1
    assert calls_after_first >= 1

    second = await pipeline.ingest("facts.txt", data, skip_duplicates=True)
    assert second == 0
    # No further embed calls: the duplicate was skipped before embedding.
    assert embedder.calls == calls_after_first


async def test_default_ingest_reembeds_and_returns_full_chunk_count() -> None:
    # Regression guard: with no flags, re-ingesting the same document still
    # embeds it and returns the full chunk count (contract unchanged 1:1).
    embedder = _CountingEmbedder(dim=128)
    pipeline = RagPipeline(embedder, InMemoryVectorStore(), chunk_size=80, chunk_overlap=20)
    data = _DOC.encode("utf-8")

    first = await pipeline.ingest("facts.txt", data)
    calls_after_first = embedder.calls
    second = await pipeline.ingest("facts.txt", data)

    assert first >= 1
    assert second == first  # same document -> same chunk count, no skipping
    assert embedder.calls > calls_after_first  # embedded again


async def test_dedup_reduces_chunk_count_for_repeated_text() -> None:
    data = _DOC_WITH_DUPES.encode("utf-8")

    baseline = RagPipeline(HashEmbedder(dim=128), InMemoryVectorStore(), chunk_size=40, chunk_overlap=0)
    without_dedup = await baseline.ingest("dupes.txt", data)

    deduped = RagPipeline(HashEmbedder(dim=128), InMemoryVectorStore(), chunk_size=40, chunk_overlap=0)
    with_dedup = await deduped.ingest("dupes.txt", data, dedup=True)

    assert with_dedup < without_dedup
    assert with_dedup >= 1


async def test_batched_embeddings_match_direct_embeddings() -> None:
    # embed_in_batches (used inside ingest) must be byte-for-byte identical to a
    # direct embed call, so batching never changes stored vectors.
    from app.rag.chunk import chunk_text
    from app.rag.embed import embed_in_batches

    embedder = HashEmbedder(dim=64)
    text = _DOC
    chunks = chunk_text(text, 40, 0)

    batched = await embed_in_batches(embedder, chunks)
    direct = await embedder.embed(chunks)
    assert batched == direct


async def test_ingest_records_document_meta_in_memory_store() -> None:
    store = InMemoryVectorStore()
    pipeline = RagPipeline(HashEmbedder(dim=128), store, chunk_size=80, chunk_overlap=20)
    data = _DOC.encode("utf-8")
    count = await pipeline.ingest("facts.txt", data)

    meta = store._meta["facts.txt"]  # noqa: SLF001 - white-box metadata check
    assert meta.source == "facts.txt"
    assert meta.size_bytes == len(data)
    assert meta.chunk_count == count


async def test_repeated_retrieve_of_same_query_embeds_once() -> None:
    # The query embedding is cached, so asking the identical question twice
    # calls the embedder only once (matters for paid embedding endpoints).
    embedder = _CountingEmbedder(dim=128)
    pipeline = RagPipeline(embedder, InMemoryVectorStore(), chunk_size=80, chunk_overlap=20)
    await pipeline.ingest("facts.txt", _DOC.encode("utf-8"))
    calls_after_ingest = embedder.calls

    first = await pipeline.retrieve("Where is the Eiffel Tower located?", k=1)
    assert embedder.calls == calls_after_ingest + 1

    second = await pipeline.retrieve("Where is the Eiffel Tower located?", k=1)
    # No extra embed call: the query vector was served from cache.
    assert embedder.calls == calls_after_ingest + 1
    # Same vector -> identical hits.
    assert [(h.document, h.chunk_id) for h in first] == [(h.document, h.chunk_id) for h in second]


async def test_distinct_queries_each_embed() -> None:
    # Guard against over-caching: a different question must still embed.
    embedder = _CountingEmbedder(dim=128)
    pipeline = RagPipeline(embedder, InMemoryVectorStore(), chunk_size=80, chunk_overlap=20)
    await pipeline.ingest("facts.txt", _DOC.encode("utf-8"))
    calls_after_ingest = embedder.calls

    await pipeline.retrieve("Where is the Eiffel Tower located?", k=1)
    await pipeline.retrieve("What is photosynthesis?", k=1)
    assert embedder.calls == calls_after_ingest + 2
