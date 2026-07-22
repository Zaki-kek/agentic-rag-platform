"""Document-level metadata and content hashing for the ingest pipeline.

A :class:`DocumentMeta` records the provenance of an ingested document: where
it came from, a content-addressable hash of its raw bytes, when it was
ingested, its size and how many chunks it produced. The content hash lets the
pipeline recognise byte-identical documents (deduplication, incremental
ingest) without re-embedding them. Everything here is pure and offline: no
network, no disk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime


def compute_content_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw document bytes.

    The digest is content-addressable: byte-identical inputs yield the same
    string, and any change to the bytes yields a different one. This is used as
    a stable document identity for deduplication and incremental ingest.

    Args:
        data: The raw bytes of a document (as read from disk or an upload).

    Returns:
        The 64-character lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class DocumentMeta:
    """Immutable provenance record for a single ingested document.

    Instances are frozen so that a document's recorded metadata cannot drift
    after ingestion; construct a new record instead of mutating one.

    Attributes:
        source: Identifier of the document's origin (filename or path).
        content_hash: SHA-256 hex digest of the raw bytes, as produced by
            :func:`compute_content_hash`.
        ingested_at: ISO-8601 timestamp of ingestion, parseable by
            :func:`datetime.datetime.fromisoformat`.
        size_bytes: Size of the raw document in bytes.
        chunk_count: Number of chunks the document was split into.
    """

    source: str
    content_hash: str
    ingested_at: str
    size_bytes: int
    chunk_count: int

    @classmethod
    def create(cls, source: str, data: bytes, chunk_count: int) -> DocumentMeta:
        """Build a metadata record from a document's bytes at the current time.

        The content hash and size are derived from ``data``; ``ingested_at`` is
        stamped with the current UTC time in ISO-8601 form.

        Args:
            source: Identifier of the document's origin (filename or path).
            data: The raw bytes of the document.
            chunk_count: Number of chunks the document was split into.

        Returns:
            A new frozen :class:`DocumentMeta` describing the document.
        """
        return cls(
            source=source,
            content_hash=compute_content_hash(data),
            ingested_at=datetime.now(UTC).isoformat(),
            size_bytes=len(data),
            chunk_count=chunk_count,
        )
