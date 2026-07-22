"""Unit tests for PgVectorStore connection-pool configuration (no live DB).

These exercise only the constructor: pool bounds are stored for asyncpg to use
later, and an inverted range is rejected eagerly. Opening the pool itself needs
a real Postgres and is covered by the docker-backed integration test.
"""

from __future__ import annotations

import pytest

from app.rag.store import PgVectorStore

_DSN = "postgresql://postgres:postgres@localhost:5432/assistant"


def test_pool_bounds_are_stored() -> None:
    store = PgVectorStore(_DSN, dim=256, pool_min=2, pool_max=5)
    assert store._pool_min == 2  # noqa: SLF001 - white-box config check
    assert store._pool_max == 5  # noqa: SLF001 - white-box config check
    # The pool itself is only opened in init() against a live database.
    assert store._pool is None  # noqa: SLF001 - white-box config check


def test_pool_defaults_are_valid() -> None:
    store = PgVectorStore(_DSN, dim=256)
    assert store._pool_min == 1  # noqa: SLF001 - white-box config check
    assert store._pool_max == 10  # noqa: SLF001 - white-box config check


def test_pool_min_above_max_raises_value_error() -> None:
    with pytest.raises(ValueError, match="pool_min"):
        PgVectorStore(_DSN, dim=256, pool_min=5, pool_max=2)


def test_pool_min_equal_max_is_allowed() -> None:
    store = PgVectorStore(_DSN, dim=256, pool_min=4, pool_max=4)
    assert store._pool_min == store._pool_max == 4  # noqa: SLF001 - white-box config check
