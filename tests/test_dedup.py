"""Tests for chunk deduplication (:mod:`app.rag.dedup`)."""

from __future__ import annotations

from app.rag.dedup import dedup_chunks


def test_dedup_drops_repeated_paragraph_and_keeps_order() -> None:
    # A document whose second paragraph is repeated verbatim later on.
    chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Water covers most of the Earth's surface.",
        "Photosynthesis converts light into chemical energy.",
        "Water covers most of the Earth's surface.",
    ]
    unique, duplicate_indices = dedup_chunks(chunks)

    assert len(unique) < len(chunks)
    assert unique == [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Water covers most of the Earth's surface.",
        "Photosynthesis converts light into chemical energy.",
    ]
    assert duplicate_indices == [3]


def test_dedup_normalizes_whitespace_and_case() -> None:
    # Same content, differing only by surrounding/interior whitespace and case.
    chunks = ["Hello   World", "  hello world  ", "goodbye"]
    unique, duplicate_indices = dedup_chunks(chunks)

    assert unique == ["Hello   World", "goodbye"]
    assert duplicate_indices == [1]


def test_dedup_all_unique_returns_input_unchanged() -> None:
    chunks = ["alpha", "beta", "gamma"]
    unique, duplicate_indices = dedup_chunks(chunks)

    assert unique == chunks
    assert duplicate_indices == []


def test_dedup_does_not_mutate_input() -> None:
    chunks = ["a", "a", "b"]
    original = list(chunks)
    dedup_chunks(chunks)
    assert chunks == original


def test_dedup_empty_input() -> None:
    assert dedup_chunks([]) == ([], [])
