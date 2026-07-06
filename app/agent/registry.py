"""Runtime registry of tool instances available to the agent."""

from __future__ import annotations

from app.agent.base import Tool, ToolError
from app.core import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """A name-indexed collection of tool instances.

    Unlike the decorator-based provider registry, tools are injected as live
    instances because some carry runtime dependencies (e.g. a retrieval
    callable). Build one at request time and pass it to :class:`~app.agent.loop.Agent`.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> ToolRegistry:
        """Register a tool instance, returning self for chaining.

        Args:
            tool: The tool to register; its ``name`` is the lookup key.

        Returns:
            This registry, so ``add`` calls can be chained.

        Raises:
            ToolError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ToolError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)
        return self

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name``.

        Args:
            name: The tool name to look up.

        Returns:
            The matching tool instance.

        Raises:
            ToolError: If no tool is registered under ``name``.
        """
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "(none)"
            raise ToolError(f"Unknown tool '{name}'. Known: {known}")
        return tool

    def list_tools(self) -> list[Tool]:
        """Return all registered tools, ordered by name."""
        return [self._tools[name] for name in sorted(self._tools)]

    def describe(self) -> str:
        """Return a newline-joined ``name: description`` catalogue for prompts."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self.list_tools())
