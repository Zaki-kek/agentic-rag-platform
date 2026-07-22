"""Chunk deduplication for the ingest pipeline.

Documents frequently repeat text - boilerplate headers, copy-pasted
paragraphs, near-identical list items - and embedding every copy wastes both
compute and vector-store space. :func:`dedup_chunks` collapses chunks that are
identical up to whitespace and case, keeping the first occurrence of each and
reporting which positions were dropped. It is a pure function: the input list
is never mutated. Everything here is offline - no network, no disk.
"""

from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Canonicalise a chunk so trivial formatting differences collapse.

    Leading/trailing whitespace is stripped, internal runs of whitespace are
    collapsed to a single space, and the result is lowercased. This makes
    ``"  Hello   World "`` and ``"hello world"`` compare equal.

    Args:
        text: A raw chunk of document text.

    Returns:
        The normalised form used as the deduplication key.
    """
    return _WHITESPACE_RE.sub(" ", text.strip()).lower()


def _chunk_key(text: str) -> str:
    """Return a stable SHA-1 hex key for a chunk's normalised content.

    Args:
        text: A raw chunk of document text.

    Returns:
        The 40-character hexadecimal SHA-1 digest of the normalised chunk.
    """
    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()


def dedup_chunks(chunks: list[str]) -> tuple[list[str], list[int]]:
    """Drop chunks that duplicate earlier ones, keeping first occurrences.

    Two chunks are considered duplicates when they are identical after
    normalisation (:func:`_normalize`: strip, collapse whitespace, lowercase).
    The first time a given normalised form is seen its original chunk is kept;
    every later chunk with the same form is dropped, and its index in the
    original list is recorded.

    Args:
        chunks: The chunks to deduplicate, in order. Not mutated.

    Returns:
        A ``(unique, duplicate_indices)`` pair where ``unique`` holds the kept
        chunks (verbatim first-occurrence text) in original order, and
        ``duplicate_indices`` lists the positions in ``chunks`` that were
        dropped as duplicates, in ascending order.
    """
    seen: set[str] = set()
    unique: list[str] = []
    duplicate_indices: list[int] = []
    for index, chunk in enumerate(chunks):
        key = _chunk_key(chunk)
        if key in seen:
            duplicate_indices.append(index)
            continue
        seen.add(key)
        unique.append(chunk)
    return unique, duplicate_indices
