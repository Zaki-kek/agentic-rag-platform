"""Tests for document metadata and content hashing (:mod:`app.rag.metadata`)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from app.rag.metadata import DocumentMeta, compute_content_hash


def test_content_hash_is_deterministic() -> None:
    data = b"the eiffel tower is 330 metres tall"
    assert compute_content_hash(data) == compute_content_hash(data)


def test_content_hash_differs_for_different_bytes() -> None:
    assert compute_content_hash(b"alpha") != compute_content_hash(b"beta")


def test_content_hash_is_sha256_hex() -> None:
    digest = compute_content_hash(b"anything")
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_ingested_at_parses_as_iso_datetime() -> None:
    meta = DocumentMeta.create("doc.txt", b"hello world", chunk_count=2)
    parsed = datetime.fromisoformat(meta.ingested_at)
    assert isinstance(parsed, datetime)


def test_create_populates_fields_from_bytes() -> None:
    data = b"photosynthesis converts light into energy"
    meta = DocumentMeta.create("bio.txt", data, chunk_count=3)
    assert meta.source == "bio.txt"
    assert meta.content_hash == compute_content_hash(data)
    assert meta.size_bytes == len(data)
    assert meta.chunk_count == 3


def test_document_meta_is_frozen() -> None:
    meta = DocumentMeta.create("doc.txt", b"data", chunk_count=1)
    with pytest.raises(FrozenInstanceError):
        meta.source = "other.txt"  # type: ignore[misc]
