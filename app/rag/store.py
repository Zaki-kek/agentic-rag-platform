"""Vector stores (factory-selected): in-memory for tests, pgvector for prod."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from app.config import Settings
    from app.rag.metadata import DocumentMeta


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its similarity score (higher is closer)."""

    document: str
    chunk_id: int
    score: float
    text: str


class VectorStore(Protocol):
    """Async vector store interface."""

    async def init(self) -> None: ...

    async def add(
        self,
        document: str,
        chunks: list[str],
        embeddings: list[list[float]],
        meta: DocumentMeta | None = None,
    ) -> None: ...

    async def search(self, embedding: list[float], k: int) -> list[Hit]: ...

    async def close(self) -> None: ...


class InMemoryVectorStore:
    """Process-local store using cosine similarity over numpy arrays."""

    def __init__(self) -> None:
        self._docs: list[str] = []
        self._chunk_ids: list[int] = []
        self._texts: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._meta: dict[str, DocumentMeta] = {}

    async def init(self) -> None:
        return None

    async def add(
        self,
        document: str,
        chunks: list[str],
        embeddings: list[list[float]],
        meta: DocumentMeta | None = None,
    ) -> None:
        if meta is not None:
            self._meta[document] = meta
        for i, (text, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            self._docs.append(document)
            self._chunk_ids.append(i)
            self._texts.append(text)
            self._vectors.append(_unit(np.asarray(emb, dtype=np.float32)))

    async def search(self, embedding: list[float], k: int) -> list[Hit]:
        if not self._vectors:
            return []
        query = _unit(np.asarray(embedding, dtype=np.float32))
        matrix = np.vstack(self._vectors)
        scores = matrix @ query
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self._docs[i], self._chunk_ids[i], float(scores[i]), self._texts[i]) for i in top]

    async def close(self) -> None:
        return None


class PgVectorStore:
    """PostgreSQL + pgvector store using cosine distance (`<=>`)."""

    def __init__(self, dsn: str, dim: int, pool_min: int = 1, pool_max: int = 10) -> None:
        if pool_min > pool_max:
            raise ValueError(f"pool_min ({pool_min}) must not exceed pool_max ({pool_max})")
        self._dsn = dsn
        self._dim = dim
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: Any = None  # asyncpg.Pool, set in init()

    async def init(self) -> None:
        import asyncpg  # lazy-imported

        self._pool = await asyncpg.create_pool(self._dsn, min_size=self._pool_min, max_size=self._pool_max)
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document TEXT NOT NULL,
                    chunk_id INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self._dim}) NOT NULL
                )
                """
            )
            # ANN index so search() uses an Index Scan instead of a full Seq Scan.
            # HNSW + vector_cosine_ops matches search()'s `<=>` (cosine distance)
            # operator; pgvector:pg16 ships HNSW support. Verified by the
            # docker-backed test tests/integration/test_pgvector_index.py.
            await conn.execute(  # pragma: no cover - requires live pgvector, offline-unreachable
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )

    async def add(
        self,
        document: str,
        chunks: list[str],
        embeddings: list[list[float]],
        meta: DocumentMeta | None = None,
    ) -> None:
        rows = [
            (document, i, text, _to_pgvector(emb)) for i, (text, emb) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO chunks (document, chunk_id, content, embedding) VALUES ($1, $2, $3, $4::vector)",
                rows,
            )
            if meta is not None:  # pragma: no cover - requires live pgvector, offline-unreachable
                await self._record_meta(conn, meta)

    async def _record_meta(self, conn: Any, meta: DocumentMeta) -> None:  # pragma: no cover - live DB only
        """Persist optional per-document provenance in a nullable side table.

        The core ``chunks`` insert is untouched; document metadata lives in a
        separate ``documents`` table created on demand so older schemas keep
        working. Reached only against a live pgvector database.
        """
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source TEXT PRIMARY KEY,
                content_hash TEXT,
                ingested_at TEXT,
                size_bytes BIGINT,
                chunk_count INT
            )
            """
        )
        await conn.execute(
            "INSERT INTO documents (source, content_hash, ingested_at, size_bytes, chunk_count) "
            "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (source) DO UPDATE SET "
            "content_hash = EXCLUDED.content_hash, ingested_at = EXCLUDED.ingested_at, "
            "size_bytes = EXCLUDED.size_bytes, chunk_count = EXCLUDED.chunk_count",
            meta.source,
            meta.content_hash,
            meta.ingested_at,
            meta.size_bytes,
            meta.chunk_count,
        )

    async def search(self, embedding: list[float], k: int) -> list[Hit]:
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT document, chunk_id, content, 1 - (embedding <=> $1::vector) AS score "
                "FROM chunks ORDER BY embedding <=> $1::vector LIMIT $2",
                _to_pgvector(embedding),
                k,
            )
        return [Hit(r["document"], r["chunk_id"], float(r["score"]), r["content"]) for r in records]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def build_store(settings: Settings, dim: int) -> VectorStore:
    """Instantiate the vector store selected in settings.vector_store."""
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    if settings.vector_store == "pgvector":
        return PgVectorStore(
            dsn=settings.db_dsn,
            dim=dim,
            pool_min=settings.db_pool_min,
            pool_max=settings.db_pool_max,
        )
    raise ValueError(f"Unknown vector_store '{settings.vector_store}' (use: memory, pgvector)")
