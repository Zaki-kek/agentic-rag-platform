"""Tests for the transport-agnostic Telegram assistant facade."""

from __future__ import annotations

from app.telegram import AssistantFacade


def _make_answer_fn(answer: str, sources: list[str]):
    """Build a fake async answer_fn returning canned answer and sources."""

    async def _answer_fn(text: str) -> tuple[str, list[str]]:
        assert text  # the facade should only call us with non-empty input
        return answer, list(sources)

    return _answer_fn


async def test_handle_includes_answer_and_sources() -> None:
    facade = AssistantFacade(
        _make_answer_fn("The capital of France is Paris.", ["doc-a: geography notes", "doc-b: atlas"])
    )

    reply = await facade.handle("What is the capital of France?")

    assert "The capital of France is Paris." in reply
    assert "Sources:" in reply
    assert "doc-a: geography notes" in reply
    assert "doc-b: atlas" in reply
    # sources are enumerated after the answer
    assert reply.index("Sources:") > reply.index("The capital of France is Paris.")
    assert "1. doc-a: geography notes" in reply
    assert "2. doc-b: atlas" in reply


async def test_handle_without_sources_omits_header() -> None:
    facade = AssistantFacade(_make_answer_fn("A short answer.", []))

    reply = await facade.handle("Anything?")

    assert reply == "A short answer."
    assert "Sources:" not in reply


async def test_handle_empty_input_returns_prompt() -> None:
    called = False

    async def _answer_fn(text: str) -> tuple[str, list[str]]:
        nonlocal called
        called = True
        return "should not happen", []

    facade = AssistantFacade(_answer_fn)

    reply = await facade.handle("")

    assert not called  # answer_fn must not be invoked on empty input
    assert reply
    assert "Sources:" not in reply
    assert "question" in reply.lower()


async def test_handle_whitespace_only_input_returns_prompt() -> None:
    facade = AssistantFacade(_make_answer_fn("unused", ["unused"]))

    reply = await facade.handle("   \n\t  ")

    assert "Sources:" not in reply
    assert "question" in reply.lower()


async def test_handle_ignores_blank_source_snippets() -> None:
    facade = AssistantFacade(_make_answer_fn("Answer.", ["  ", "", "real source"]))

    reply = await facade.handle("Q?")

    assert "Sources:" in reply
    assert "1. real source" in reply
    assert "2." not in reply  # blank snippets were filtered out
