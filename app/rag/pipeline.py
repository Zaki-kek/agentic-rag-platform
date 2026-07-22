"""RAG pipeline: orchestrates extraction, chunking, embedding and retrieval."""

from __future__ import annotations

from app.core import get_logger
from app.rag.cache import EmbeddingCache
from app.rag.chunk import chunk_text
from app.rag.dedup import dedup_chunks
from app.rag.embed import Embedder, embed_in_batches
from app.rag.extract import extract_text
from app.rag.metadata import DocumentMeta, compute_content_hash
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
        # Content hashes of documents already ingested by this pipeline
        # instance. Kept here - not in the store - so incremental ingest works
        # against any VectorStore (the Protocol has no `has_document`, and the
        # pgvector store is untouched). Only consulted when skip_duplicates=True.
        self._seen_hashes: set[str] = set()
        # In-process cache for query embeddings: asking the same question twice
        # re-uses the first vector instead of calling the (possibly paid)
        # embedder again. The result is identical to embedding the query
        # directly; only the number of provider calls changes.
        self._query_cache = EmbeddingCache()
        self._query_model_id = f"{type(embedder).__name__}:{getattr(embedder, 'dim', 0)}"

    async def ingest(
        self,
        filename: str,
        data: bytes,
        *,
        dedup: bool = False,
        skip_duplicates: bool = False,
    ) -> int:
        """Extract, chunk, embed and store a document. Returns chunk count.

        With both flags at their defaults this behaves exactly as before:
        every chunk is embedded and stored, and the chunk count is returned.

        Args:
            filename: Name of the document (drives type detection and provenance).
            data: Raw document bytes.
            dedup: When ``True``, collapse chunks that are identical up to
                whitespace and case (see :func:`app.rag.dedup.dedup_chunks`)
                before embedding, so repeated text is embedded once.
            skip_duplicates: When ``True``, skip the document entirely (no
                embedding, no store write) if a byte-identical document was
                already ingested by this pipeline instance; returns ``0``.

        Returns:
            The number of chunks embedded and stored (``0`` if nothing was
            extracted, or if the document was skipped as a duplicate).
        """
        content_hash = compute_content_hash(data)
        if skip_duplicates and content_hash in self._seen_hashes:
            logger.info("Skipping duplicate %s (hash %s)", filename, content_hash[:12])
            return 0

        text = extract_text(filename, data)
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            logger.warning("No text extracted from %s", filename)
            return 0
        if dedup:
            chunks, dropped = dedup_chunks(chunks)
            if dropped:
                logger.info("Deduplicated %s: dropped %d chunk(s)", filename, len(dropped))

        embeddings = await embed_in_batches(self._embedder, chunks)
        meta = DocumentMeta.create(source=filename, data=data, chunk_count=len(chunks))
        await self._store.add(filename, chunks, embeddings, meta=meta)
        self._seen_hashes.add(content_hash)
        logger.info("Ingested %s (%d chunks)", filename, len(chunks))
        return len(chunks)

    async def retrieve(self, query: str, k: int) -> list[Hit]:
        """Return the top-k chunks most relevant to the query.

        The query embedding is served from an in-process cache, so repeating the
        same question does not re-embed it. The vector - and therefore the
        returned hits - are identical to embedding the query directly.
        """
        embedding = (await self._query_cache.get_or_compute(self._embedder, [query], self._query_model_id))[0]
        return await self._store.search(embedding, k)
