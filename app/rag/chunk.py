"""Text chunking with overlap (word-boundary aware)."""

from __future__ import annotations


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks of roughly `size` characters.

    Splits on whitespace so words are not cut mid-token. `overlap` characters
    of context are repeated between consecutive chunks to preserve continuity.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        # +1 accounts for the joining space.
        if length + len(word) + 1 > size and current:
            chunks.append(" ".join(current))
            current, length = _carry_overlap(current, overlap)
        current.append(word)
        length += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def _carry_overlap(words: list[str], overlap: int) -> tuple[list[str], int]:
    """Return the tail of `words` whose joined length is <= overlap."""
    carried: list[str] = []
    length = 0
    for word in reversed(words):
        if length + len(word) + 1 > overlap:
            break
        carried.insert(0, word)
        length += len(word) + 1
    return carried, length
