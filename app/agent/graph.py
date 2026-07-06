"""Dependency-free, LangGraph-style state-graph runner for agent orchestration.

A graph is a set of named async *nodes* connected by edges. Each node is an
async callable ``(state) -> dict`` whose returned dict is shallow-merged into a
shared mutable ``state``. Control flow follows either an unconditional edge or a
*router* that inspects the current state and returns the next node name. The
special sink node :data:`END` terminates the walk.

This is a small, dependency-free stand-in that mirrors the ergonomics of
`LangGraph <https://github.com/langchain-ai/langgraph>`_ (``StateGraph`` /
``add_node`` / ``add_edge`` / ``add_conditional_edges`` / ``compile``). For
production use, swap this module for LangGraph; the call sites built around the
:class:`StateGraph` / :class:`CompiledGraph` shapes here map onto its API
directly. The runner is deliberately LLM-agnostic: nodes may do anything, and
nothing here knows about providers, tools or messages.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.core import AppError, get_logger

logger = get_logger(__name__)

# A node consumes the current state and returns the keys it produced, which are
# merged into the state. Keep it generic: no LLM specifics belong here.
NodeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# A router inspects the current state and returns the name of the next node.
RouterFn = Callable[[dict[str, Any]], str]

#: The terminal sink node name. Reaching it ends the walk.
END: str = "END"


class StateGraph:
    """A mutable builder for a directed graph of async nodes.

    Build the graph by registering nodes and edges, choose an entry node, then
    call :meth:`compile` to validate it and obtain a runnable
    :class:`CompiledGraph`.

    A node may have at most one outgoing transition: either an unconditional
    edge (:meth:`add_edge`) or a conditional router (:meth:`add_conditional_edges`).
    The conditional router takes priority if both are somehow present.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._routers: dict[str, RouterFn] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        """Register an async node under ``name``.

        Args:
            name: The unique node name. Must not be the reserved sink :data:`END`.
            fn: The async callable ``(state) -> dict`` to run for this node.

        Returns:
            This graph, so calls can be chained.

        Raises:
            AppError: If ``name`` is the reserved :data:`END` sink or is already
                registered.
        """
        if name == END:
            raise AppError(f"Node name '{END}' is reserved for the sink")
        if name in self._nodes:
            raise AppError(f"Node '{name}' is already registered")
        self._nodes[name] = fn
        logger.debug("Added node: %s", name)
        return self

    def add_edge(self, from_name: str, to_name: str) -> StateGraph:
        """Add an unconditional edge from ``from_name`` to ``to_name``.

        Args:
            from_name: The source node name.
            to_name: The destination node name, or :data:`END` to terminate.

        Returns:
            This graph, so calls can be chained.

        Raises:
            AppError: If ``from_name`` already has an unconditional edge.
        """
        if from_name in self._edges:
            raise AppError(f"Node '{from_name}' already has an unconditional edge")
        self._edges[from_name] = to_name
        logger.debug("Added edge: %s -> %s", from_name, to_name)
        return self

    def add_conditional_edges(self, from_name: str, router: RouterFn) -> StateGraph:
        """Add a state-driven router that picks the next node for ``from_name``.

        Args:
            from_name: The source node name.
            router: A callable ``(state) -> str`` returning the next node name
                (or :data:`END`) based on the current state.

        Returns:
            This graph, so calls can be chained.

        Raises:
            AppError: If ``from_name`` already has a conditional router.
        """
        if from_name in self._routers:
            raise AppError(f"Node '{from_name}' already has a conditional router")
        self._routers[from_name] = router
        logger.debug("Added conditional edges from: %s", from_name)
        return self

    def set_entry(self, name: str) -> StateGraph:
        """Set the entry node where a run begins.

        Args:
            name: The node name to start from.

        Returns:
            This graph, so calls can be chained.
        """
        self._entry = name
        logger.debug("Set entry: %s", name)
        return self

    def compile(self) -> CompiledGraph:
        """Validate the graph and return an immutable runnable.

        Validation checks that an entry node is set and exists, and that every
        edge source and every unconditional-edge destination references a known
        node (or :data:`END` for destinations). Conditional routers are not
        statically validated because their targets are computed at runtime; an
        unknown runtime target is caught during :meth:`CompiledGraph.run`.

        Returns:
            A :class:`CompiledGraph` snapshot of the current builder state.

        Raises:
            AppError: If the entry node is unset/unknown, or if any edge points
                to or from a node that does not exist (a dangling edge).
        """
        if self._entry is None:
            raise AppError("Graph has no entry node; call set_entry() before compile()")
        if self._entry not in self._nodes:
            raise AppError(f"Entry node '{self._entry}' is not a registered node")

        for from_name, to_name in self._edges.items():
            if from_name not in self._nodes:
                raise AppError(f"Edge source '{from_name}' is not a registered node")
            if to_name != END and to_name not in self._nodes:
                raise AppError(f"Dangling edge: '{from_name}' -> '{to_name}' (unknown node)")

        for from_name in self._routers:
            if from_name not in self._nodes:
                raise AppError(f"Conditional edge source '{from_name}' is not a registered node")

        logger.info("Compiled graph with %d node(s), entry='%s'", len(self._nodes), self._entry)
        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            routers=dict(self._routers),
            entry=self._entry,
        )


class CompiledGraph:
    """An immutable, runnable graph produced by :meth:`StateGraph.compile`.

    Instances are normally created via :meth:`StateGraph.compile` rather than
    directly. Running walks from the entry node, executes each node, merges its
    output into the state, and follows the node's transition until :data:`END`
    is reached.
    """

    def __init__(
        self,
        nodes: dict[str, NodeFn],
        edges: dict[str, str],
        routers: dict[str, RouterFn],
        entry: str,
    ) -> None:
        """Store the validated graph snapshot.

        Args:
            nodes: Mapping of node name to its async callable.
            edges: Mapping of source node name to unconditional destination.
            routers: Mapping of source node name to its state-driven router.
            entry: The validated entry node name.
        """
        self._nodes = nodes
        self._edges = edges
        self._routers = routers
        self._entry = entry

    async def run(self, state: dict[str, Any], max_steps: int = 25) -> dict[str, Any]:
        """Execute the graph from the entry node until it reaches :data:`END`.

        The supplied ``state`` is mutated in place (and also returned). At each
        step the current node runs, its returned dict is merged into the state,
        and the next node is chosen by the node's conditional router (if any) or
        its unconditional edge.

        Args:
            state: The mutable state dict shared across nodes.
            max_steps: Maximum number of node executions before aborting. Guards
                against cycles that never reach :data:`END`.

        Returns:
            The final state dict after the walk terminates at :data:`END`.

        Raises:
            AppError: If a node is missing a transition, a router returns an
                unknown node, or ``max_steps`` is exceeded (a likely cycle).
        """
        current = self._entry
        for step in range(max_steps):
            if current == END:
                logger.info("Graph reached END after %d step(s)", step)
                return state

            node = self._nodes.get(current)
            if node is None:  # pragma: no cover - guarded by compile()
                raise AppError(f"Reached unknown node '{current}' at step {step}")

            produced = await node(state)
            state.update(produced)

            current = self._next_node(current, state)

        raise AppError(f"Graph exceeded max_steps={max_steps} without reaching END (possible cycle)")

    def _next_node(self, current: str, state: dict[str, Any]) -> str:
        """Resolve the transition out of ``current`` given the current state.

        A conditional router takes priority over an unconditional edge. The
        chosen target must be a known node or :data:`END`.

        Args:
            current: The node that just ran.
            state: The current state, passed to any router.

        Returns:
            The name of the next node (or :data:`END`).

        Raises:
            AppError: If ``current`` has no outgoing transition, or a router
                returns a node name that does not exist.
        """
        router = self._routers.get(current)
        if router is not None:
            target = router(state)
            if target != END and target not in self._nodes:
                raise AppError(f"Router from '{current}' returned unknown node '{target}'")
            return target

        edge_target = self._edges.get(current)
        if edge_target is None:
            raise AppError(f"Node '{current}' has no outgoing edge and is not END")
        return edge_target


__all__ = [
    "END",
    "NodeFn",
    "RouterFn",
    "StateGraph",
    "CompiledGraph",
]
