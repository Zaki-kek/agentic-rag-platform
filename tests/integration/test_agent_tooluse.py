"""Integration: the agent drives real tools via a scripted provider."""

from __future__ import annotations

import pytest

from app.agent import Agent, CalculatorTool, RetrievalTool, ToolRegistry
from app.llm.base import Message
from app.rag import RagPipeline
from app.rag.embed import HashEmbedder
from app.rag.store import InMemoryVectorStore

pytestmark = pytest.mark.asyncio


class ScriptedProvider:
    """Returns canned replies in sequence to drive the agent deterministically."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._index = 0

    async def generate(self, messages: list[Message]) -> str:
        reply = self._replies[self._index]
        self._index += 1
        return reply


async def test_agent_calls_calculator_then_finalizes() -> None:
    provider = ScriptedProvider(['ACTION: calculator {"expression": "2+2*3"}', "FINAL: the result is 8"])
    registry = ToolRegistry()
    registry.add(CalculatorTool())

    result = await Agent(provider, registry).run("what is 2+2*3?")

    assert result.finished
    assert result.steps[0]["tool"] == "calculator"
    assert result.steps[0]["observation"] == "8"
    assert "8" in result.answer


async def test_agent_calls_retrieval_over_real_pipeline() -> None:
    pipeline = RagPipeline(HashEmbedder(128), InMemoryVectorStore())
    await pipeline.ingest("kb.txt", b"The capital of France is Paris.")

    provider = ScriptedProvider(['ACTION: retrieval {"query": "capital of France"}', "FINAL: Paris"])
    registry = ToolRegistry()
    registry.add(RetrievalTool(pipeline.retrieve))

    result = await Agent(provider, registry).run("capital of France?")

    assert result.finished
    assert "Paris" in result.steps[0]["observation"]


async def test_agent_recovers_from_unknown_tool() -> None:
    provider = ScriptedProvider(["ACTION: nope {}", "FINAL: done"])
    registry = ToolRegistry()
    registry.add(CalculatorTool())

    result = await Agent(provider, registry).run("go")

    assert result.finished
    assert result.steps[0]["observation"].startswith("ERROR")
