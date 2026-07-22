"""Tests for the ReAct agent loop, registry and built-in tools (fully offline)."""

from __future__ import annotations

import pytest

from app.agent.base import ToolError
from app.agent.loop import Agent, AgentResult, estimate_tokens
from app.agent.registry import ToolRegistry
from app.agent.tools import CalculatorTool, RetrievalTool, safe_eval
from app.observability.tracer import InMemoryTracer
from app.rag.store import Hit


class FakeProvider:
    """Provider returning scripted replies in sequence, recording inputs."""

    name = "fake"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages: list[dict[str, str]]) -> str:
        # Snapshot the conversation so tests can assert on observations.
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            return "FINAL: (out of scripted replies)"
        return self._replies.pop(0)


def _registry_with_calculator() -> ToolRegistry:
    return ToolRegistry().add(CalculatorTool())


async def test_action_then_final_runs_calculator() -> None:
    provider = FakeProvider(
        [
            'ACTION: calculator {"expression": "2+2*3"}',
            "FINAL: the answer is 8",
        ]
    )
    agent = Agent(provider, _registry_with_calculator())

    result = await agent.run("what is 2+2*3?")

    assert isinstance(result, AgentResult)
    assert result.finished is True
    assert result.answer == "the answer is 8"
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step["tool"] == "calculator"
    assert step["args"] == {"expression": "2+2*3"}
    assert step["observation"] == "8"

    # The observation must have been fed back into the conversation.
    last_conversation = provider.calls[-1]
    assert any(m["content"] == "OBSERVATION: 8" for m in last_conversation)


async def test_reply_without_prefix_is_treated_as_final() -> None:
    provider = FakeProvider(["just a plain answer with no prefix"])
    agent = Agent(provider, _registry_with_calculator())

    result = await agent.run("hello?")

    assert result.finished is True
    assert result.answer == "just a plain answer with no prefix"
    assert result.steps == []


async def test_final_prefix_is_trimmed() -> None:
    provider = FakeProvider(["FINAL:   spaced answer  "])
    result = await Agent(provider, _registry_with_calculator()).run("q")
    assert result.answer == "spaced answer"


async def test_calculator_rejects_unsafe_input_in_tool() -> None:
    with pytest.raises(ToolError):
        await CalculatorTool().run({"expression": "__import__('os').system('echo hi')"})


def test_safe_eval_rejects_names_and_calls() -> None:
    with pytest.raises(ToolError):
        safe_eval("os.getcwd()")
    with pytest.raises(ToolError):
        safe_eval("1 ** 8")  # power is not whitelisted
    with pytest.raises(ToolError):
        safe_eval("")


def test_safe_eval_basic_arithmetic() -> None:
    assert safe_eval("2+2*3") == 8.0
    assert safe_eval("(1+2)/4") == 0.75
    assert safe_eval("-3 + 5") == 2.0


async def test_calculator_division_by_zero_raises_toolerror() -> None:
    with pytest.raises(ToolError):
        await CalculatorTool().run({"expression": "1/0"})


async def test_max_steps_is_respected() -> None:
    # Every reply is an ACTION, so the loop can never finish naturally.
    provider = FakeProvider(['ACTION: calculator {"expression": "1+1"}'] * 10)
    agent = Agent(provider, _registry_with_calculator(), max_steps=3)

    result = await agent.run("loop forever?")

    assert result.finished is False
    assert len(result.steps) == 3
    assert len(provider.calls) == 3
    assert result.answer == "2"  # falls back to the last observation


def test_estimate_tokens_is_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


async def test_agent_stops_on_token_budget() -> None:
    # Every reply is an ACTION, so only the budget can stop the loop early.
    provider = FakeProvider(['ACTION: calculator {"expression": "1+1"}'] * 20)
    # Tiny budget: the very first prompt (system + question) already overruns it.
    agent = Agent(provider, _registry_with_calculator(), max_steps=20, max_tokens=1)

    result = await agent.run("what is 1+1?")

    assert result.finished is False
    assert result.stop_reason == "budget"
    assert result.steps == []  # stopped before issuing any LLM call
    assert provider.calls == []  # no generate() call was made
    assert "budget" in result.answer.lower()


async def test_agent_budget_allows_completion_within_limit() -> None:
    # A generous budget must not interfere with a normal finish.
    provider = FakeProvider(
        [
            'ACTION: calculator {"expression": "2+2"}',
            "FINAL: four",
        ]
    )
    agent = Agent(provider, _registry_with_calculator(), max_tokens=10_000)

    result = await agent.run("2+2?")

    assert result.finished is True
    assert result.stop_reason == "final"
    assert result.answer == "four"
    assert result.tokens_used > 0


async def test_agent_emits_a_span_per_step() -> None:
    tracer = InMemoryTracer()
    provider = FakeProvider(
        [
            'ACTION: calculator {"expression": "2+2"}',
            "FINAL: four",
        ]
    )
    agent = Agent(provider, _registry_with_calculator(), tracer=tracer)

    result = await agent.run("2+2?")

    assert result.finished is True
    step_spans = [s for s in tracer.spans if s.name == "agent.step"]
    # One span for the ACTION step and one for the FINAL step.
    assert len(step_spans) == 2
    assert step_spans[0].attributes["step"] == 0
    assert step_spans[0].attributes["tool"] == "calculator"
    assert step_spans[-1].attributes["stop_reason"] == "final"
    assert all(s.duration_ms is not None for s in step_spans)


async def test_budget_span_records_stop_reason() -> None:
    tracer = InMemoryTracer()
    provider = FakeProvider(['ACTION: calculator {"expression": "1+1"}'] * 5)
    agent = Agent(provider, _registry_with_calculator(), max_tokens=1, tracer=tracer)

    result = await agent.run("hi?")

    assert result.stop_reason == "budget"
    budget_spans = [s for s in tracer.spans if s.attributes.get("stop_reason") == "budget"]
    assert len(budget_spans) == 1


async def test_unknown_tool_becomes_recoverable_observation() -> None:
    provider = FakeProvider(
        [
            'ACTION: nosuchtool {"x": 1}',
            "FINAL: recovered",
        ]
    )
    result = await Agent(provider, _registry_with_calculator()).run("q")

    assert result.finished is True
    assert result.answer == "recovered"
    assert result.steps[0]["observation"].startswith("ERROR:")


async def test_retrieval_tool_concatenates_chunks() -> None:
    async def fake_retrieve(query: str, k: int) -> list[Hit]:
        assert query == "eiffel tower"
        return [
            Hit(document="facts.txt", chunk_id=0, score=0.9, text="The Eiffel Tower is in Paris."),
            Hit(document="facts.txt", chunk_id=1, score=0.8, text="It is made of wrought iron."),
        ][:k]

    tool = RetrievalTool(fake_retrieve)
    observation = await tool.run({"query": "eiffel tower", "k": 2})

    assert "Paris" in observation
    assert "wrought iron" in observation


async def test_retrieval_tool_handles_no_hits() -> None:
    async def empty_retrieve(query: str, k: int) -> list[Hit]:
        return []

    out = await RetrievalTool(empty_retrieve).run({"query": "anything"})
    assert out == "No relevant context found."


async def test_retrieval_tool_requires_query() -> None:
    async def never(query: str, k: int) -> list[Hit]:  # pragma: no cover - never called
        return []

    with pytest.raises(ToolError):
        await RetrievalTool(never).run({"k": 3})


def test_registry_get_unknown_and_duplicate() -> None:
    registry = ToolRegistry().add(CalculatorTool())
    with pytest.raises(ToolError):
        registry.get("missing")
    with pytest.raises(ToolError):
        registry.add(CalculatorTool())  # duplicate name
    assert [t.name for t in registry.list_tools()] == ["calculator"]


async def test_agent_end_to_end_with_retrieval_then_calculator() -> None:
    async def fake_retrieve(query: str, k: int) -> list[Hit]:
        return [Hit(document="d", chunk_id=0, score=1.0, text="The tower has 3 levels.")]

    registry = ToolRegistry().add(CalculatorTool()).add(RetrievalTool(fake_retrieve))
    provider = FakeProvider(
        [
            'ACTION: retrieval {"query": "levels", "k": 1}',
            'ACTION: calculator {"expression": "3*2"}',
            "FINAL: six",
        ]
    )

    result = await Agent(provider, registry, max_steps=5).run("how many levels times two?")

    assert result.finished is True
    assert result.answer == "six"
    assert [s["tool"] for s in result.steps] == ["retrieval", "calculator"]
    assert result.steps[0]["observation"] == "The tower has 3 levels."
    assert result.steps[1]["observation"] == "6"
