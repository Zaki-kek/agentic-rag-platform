"""Tests for the dependency-free LangGraph-style state-graph runner (fully offline)."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.graph import END, CompiledGraph, StateGraph
from app.core import AppError


async def test_branching_graph_routes_even_and_odd() -> None:
    """A start node sets a number; a router sends even/odd to different sinks."""

    async def start(state: dict[str, Any]) -> dict[str, Any]:
        return {"n": state["n"] * 2 + state.get("offset", 0)}

    async def even_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"label": "even", "result": state["n"] // 2}

    async def odd_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"label": "odd", "result": state["n"] * 3}

    def route(state: dict[str, Any]) -> str:
        return "even" if state["n"] % 2 == 0 else "odd"

    graph = (
        StateGraph()
        .add_node("start", start)
        .add_node("even", even_node)
        .add_node("odd", odd_node)
        .add_conditional_edges("start", route)
        .add_edge("even", END)
        .add_edge("odd", END)
        .set_entry("start")
    )
    compiled = graph.compile()
    assert isinstance(compiled, CompiledGraph)

    # n = 3*2 + 0 = 6 -> even branch -> result = 6 // 2 = 3
    even_state = await compiled.run({"n": 3})
    assert even_state["label"] == "even"
    assert even_state["result"] == 3

    # n = 3*2 + 1 = 7 -> odd branch -> result = 7 * 3 = 21
    odd_state = await compiled.run({"n": 3, "offset": 1})
    assert odd_state["label"] == "odd"
    assert odd_state["result"] == 21


async def test_run_mutates_and_returns_same_state_object() -> None:
    async def bump(state: dict[str, Any]) -> dict[str, Any]:
        return {"x": state.get("x", 0) + 1}

    compiled = StateGraph().add_node("bump", bump).add_edge("bump", END).set_entry("bump").compile()

    original = {"x": 41}
    returned = await compiled.run(original)

    assert returned is original
    assert returned["x"] == 42


async def test_unconditional_edge_chain() -> None:
    async def a(_: dict[str, Any]) -> dict[str, Any]:
        return {"trace": ["a"]}

    async def b(state: dict[str, Any]) -> dict[str, Any]:
        return {"trace": [*state["trace"], "b"]}

    compiled = (
        StateGraph().add_node("a", a).add_node("b", b).add_edge("a", "b").add_edge("b", END).set_entry("a").compile()
    )

    state = await compiled.run({})
    assert state["trace"] == ["a", "b"]


def test_compile_without_entry_raises() -> None:
    async def noop(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    graph = StateGraph().add_node("only", noop).add_edge("only", END)
    with pytest.raises(AppError, match="no entry node"):
        graph.compile()


def test_compile_with_unknown_entry_raises() -> None:
    async def noop(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    graph = StateGraph().add_node("real", noop).add_edge("real", END).set_entry("ghost")
    with pytest.raises(AppError, match="Entry node 'ghost'"):
        graph.compile()


def test_compile_with_dangling_edge_raises() -> None:
    async def noop(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    graph = StateGraph().add_node("a", noop).add_edge("a", "nowhere").set_entry("a")
    with pytest.raises(AppError, match="Dangling edge"):
        graph.compile()


async def test_cyclic_graph_hits_max_steps() -> None:
    async def loop(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    compiled = StateGraph().add_node("loop", loop).add_edge("loop", "loop").set_entry("loop").compile()

    with pytest.raises(AppError, match="max_steps"):
        await compiled.run({}, max_steps=5)


async def test_node_without_transition_raises_at_run() -> None:
    """A node that is neither END nor has an edge/router fails when reached."""

    async def dead_end(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    # No edge or router registered for 'dead_end'; compile() permits this
    # (routers may target it dynamically), but running it has nowhere to go.
    compiled = StateGraph().add_node("dead_end", dead_end).set_entry("dead_end").compile()
    with pytest.raises(AppError, match="no outgoing edge"):
        await compiled.run({})


async def test_router_returning_unknown_node_raises() -> None:
    async def start(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    def bad_router(_: dict[str, Any]) -> str:
        return "missing"

    compiled = (
        StateGraph().add_node("start", start).add_conditional_edges("start", bad_router).set_entry("start").compile()
    )
    with pytest.raises(AppError, match="unknown node 'missing'"):
        await compiled.run({})


def test_reserved_end_name_rejected() -> None:
    async def noop(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    with pytest.raises(AppError, match="reserved"):
        StateGraph().add_node(END, noop)


def test_duplicate_node_rejected() -> None:
    async def noop(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    graph = StateGraph().add_node("a", noop)
    with pytest.raises(AppError, match="already registered"):
        graph.add_node("a", noop)


def test_duplicate_edge_and_router_rejected() -> None:
    async def noop(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    def router(_: dict[str, Any]) -> str:
        return END

    graph = StateGraph().add_node("a", noop).add_edge("a", END)
    with pytest.raises(AppError, match="already has an unconditional edge"):
        graph.add_edge("a", END)

    graph2 = StateGraph().add_node("b", noop).add_conditional_edges("b", router)
    with pytest.raises(AppError, match="already has a conditional router"):
        graph2.add_conditional_edges("b", router)


async def test_conditional_edge_priority_over_unconditional() -> None:
    """When both a router and an edge exist, the router wins."""

    async def start(_: dict[str, Any]) -> dict[str, Any]:
        return {"via": "router"}

    async def picked(state: dict[str, Any]) -> dict[str, Any]:
        return {"reached": state["via"]}

    def router(_: dict[str, Any]) -> str:
        return "picked"

    # Build via the builder for the edge, then attach a router targeting a
    # different node; the router must take priority at run time.
    graph = StateGraph()
    graph.add_node("start", start)
    graph.add_node("picked", picked)
    graph.add_node("skipped", start)
    graph.add_edge("start", "skipped")
    graph.add_conditional_edges("start", router)
    graph.add_edge("picked", END)
    graph.set_entry("start")

    state = await graph.compile().run({})
    assert state["reached"] == "router"
