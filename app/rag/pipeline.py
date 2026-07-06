"""RAG pipeline: orchestrates extraction, chunking, embedding and retrieval."""

from __future__ import annotations

from app.core import get_logger
from app.rag.chunk import chunk_text
from app.rag.embed import Embedder
from app.rag.extract import extract_text
from app.rag.store import Hit, VectorStore

logger = get_logger(__name__)


class RagPipeline:
    """Ingest documents into a vector store and retrieve relevant chunks."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest(self, filename: str, data: bytes) -> int:
        """Extract, chunk, embed and store a document. Returns chunk count."""
        text = extract_text(filename, data)
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            logger.warning("No text extracted from %s", filename)
            return 0
        embeddings = await self._embedder.embed(chunks)
        await self._store.add(filename, chunks, embeddings)
        logger.info("Ingested %s (%d chunks)", filename, len(chunks))
        return len(chunks)

    async def retrieve(self, query: str, k: int) -> list[Hit]:
        """Return the top-k chunks most relevant to the query."""
        embedding = (await self._embedder.embed([query]))[0]
        return await self._store.search(embedding, k)
