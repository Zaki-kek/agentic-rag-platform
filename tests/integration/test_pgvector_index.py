"""Docker-backed integration test: pgvector's ANN index is actually used.

Runs the real PgVectorStore lifecycle (init -> add -> search) against a live
PostgreSQL + pgvector instance (see docker-compose.yml, pgvector/pgvector:pg16),
then asserts the planner picks an Index Scan on the HNSW index created in
init() rather than a full Seq Scan.

Skipped automatically when no database is reachable, so the offline unit suite
stays green. To run it locally:

    docker compose up -d db
    DB_DSN=postgresql://postgres:postgres@localhost:5432/assistant \\
        python3 -m pytest tests/integration/test_pgvector_index.py
"""

from __future__ import annotations

import os

import pytest

from app.rag.store import PgVectorStore

DB_DSN = os.environ.get("DB_DSN")

# asyncpg is only needed on the path that actually talks to a database.
asyncpg = pytest.importorskip("asyncpg")

pytestmark = pytest.mark.skipif(
    not DB_DSN,
    reason="DB_DSN unset - set it to a live pgvector DSN to run this docker-backed test",
)

DIM = 16


def _vec(seed: int) -> list[float]:
    """A deterministic, well-spread unit-ish vector for the given seed."""
    return [float((seed * (i + 1)) % 7) for i in range(DIM)]


async def _reachable(dsn: str) -> bool:
    try:
        conn = await asyncpg.connect(dsn)
    except (OSError, asyncpg.PostgresError):
        return False
    await conn.close()
    return True


async def test_search_uses_index_scan_not_seq_scan() -> None:
    assert DB_DSN is not None  # narrowed for type-checkers; guarded by skipif
    if not await _reachable(DB_DSN):
        pytest.skip(f"no database reachable at {DB_DSN}")

    store = PgVectorStore(dsn=DB_DSN, dim=DIM)
    await store.init()
    try:
        # Enough rows that the planner prefers the ANN index over a Seq Scan.
        docs = 50
        await store.add(
            "kb.txt",
            [f"chunk {i}" for i in range(docs)],
            [_vec(i + 1) for i in range(docs)],
        )

        # Sanity: search returns results and respects k.
        hits = await store.search(_vec(1), k=5)
        assert len(hits) == 5

        pool = store._pool  # noqa: SLF001 - intentional white-box index-plan check
        async with pool.acquire() as conn:
            # Force the planner to consider the ANN index (it may fall back to a
            # Seq Scan on tiny/cold tables otherwise).
            await conn.execute("SET LOCAL enable_seqscan = off")
            plan_rows = await conn.fetch(
                "EXPLAIN (ANALYZE, FORMAT TEXT) "
                "SELECT document, chunk_id, content "
                "FROM chunks ORDER BY embedding <=> $1::vector LIMIT 5",
                "[" + ",".join(f"{x:.8f}" for x in _vec(1)) + "]",
            )
        plan = "\n".join(row[0] for row in plan_rows)

        assert "Index Scan" in plan, f"expected an Index Scan, got plan:\n{plan}"
        assert "Seq Scan" not in plan, f"unexpected Seq Scan in plan:\n{plan}"
    finally:
        # Keep the test hermetic across repeated runs.
        pool = store._pool  # noqa: SLF001
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS chunks")
        await store.close()
