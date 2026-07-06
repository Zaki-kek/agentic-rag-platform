"""Unit tests for text chunking."""

from __future__ import annotations

import pytest

from app.rag.chunk import chunk_text


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_single_chunk() -> None:
    chunks = chunk_text("hello world", size=800)
    assert chunks == ["hello world"]


def test_long_text_is_split() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        # Allow a small slack for the trailing word that crosses the boundary.
        assert len(chunk) <= 200 + 16


def test_chunks_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, size=120, overlap=40)
    first_tail = chunks[0].split()[-1]
    assert first_tail in chunks[1].split()


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", size=10, overlap=10)
