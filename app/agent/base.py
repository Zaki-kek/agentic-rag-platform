"""Agent tool abstraction and errors."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core import AppError


class ToolError(AppError):
    """Raised when a tool receives invalid arguments or fails to run."""

    status_code = 422


@runtime_checkable
class Tool(Protocol):
    """A capability the agent can invoke by name with JSON arguments.

    Implementations are plain instances (often dependency-injected) registered
    in a :class:`~app.agent.registry.ToolRegistry`. The agent serializes the
    tool's ``run`` result back into the conversation as an ``OBSERVATION``.
    """

    name: str
    description: str

    async def run(self, args: dict[str, Any]) -> str:
        """Execute the tool.

        Args:
            args: Parsed JSON arguments supplied by the model.

        Returns:
            A short string observation appended to the conversation.

        Raises:
            ToolError: If the arguments are invalid or execution fails.
        """
        ...
