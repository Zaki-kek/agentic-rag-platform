"""ReAct-style tool-using agent loop driven by an LLM provider.

The agent prompts the model to reply with exactly one line, either::

    ACTION: <tool_name> <json-args>

to invoke a tool, or::

    FINAL: <answer text>

to finish. The loop parses each reply, runs the requested tool, feeds the
result back as an ``OBSERVATION`` user turn, and repeats until the model
emits ``FINAL`` or ``max_steps`` is reached. This is a dependency-free stand-in
for a production graph runner such as LangGraph.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.agent.base import ToolError
from app.agent.registry import ToolRegistry
from app.core import get_logger
from app.llm.base import LLMProvider, Message

logger = get_logger(__name__)

_ACTION_PREFIX = "ACTION:"
_FINAL_PREFIX = "FINAL:"

_SYSTEM_TEMPLATE = (
    "You are a tool-using assistant. On each turn reply with EXACTLY ONE line.\n"
    "To use a tool, reply:\n"
    "ACTION: <tool_name> <json-args>\n"
    'For example: ACTION: calculator {{"expression": "2+2"}}\n'
    "When you can answer, reply:\n"
    "FINAL: <answer>\n"
    "Use only these tools:\n"
    "{tools}\n"
    "Never invent tools. Keep JSON arguments on the same line as ACTION."
)


class AgentResult(BaseModel):
    """The outcome of an agent run.

    Attributes:
        answer: The final answer text returned to the caller.
        steps: Per-action trace; each item has ``tool``, ``args`` and ``observation``.
        finished: ``True`` if the model emitted ``FINAL``; ``False`` if the loop
            stopped because ``max_steps`` was exhausted.
    """

    answer: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    finished: bool


class _ParsedReply(BaseModel):
    """Internal structured form of a single model reply."""

    is_action: bool
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    final_text: str = ""


def _parse_reply(reply: str) -> _ParsedReply:
    """Parse one model reply into an action or a final answer.

    Robustness rules: surrounding whitespace is trimmed; if the text neither
    starts with ``ACTION:`` nor ``FINAL:`` it is treated as a ``FINAL`` answer.

    Args:
        reply: The raw text returned by the provider.

    Returns:
        A :class:`_ParsedReply` describing the action or final answer.

    Raises:
        ToolError: If an ``ACTION`` line is missing a tool name or has
            non-object / invalid JSON arguments.
    """
    text = reply.strip()

    if text.startswith(_FINAL_PREFIX):
        return _ParsedReply(is_action=False, final_text=text[len(_FINAL_PREFIX) :].strip())

    if not text.startswith(_ACTION_PREFIX):
        # No recognizable prefix: treat the whole reply as the final answer.
        return _ParsedReply(is_action=False, final_text=text)

    body = text[len(_ACTION_PREFIX) :].strip()
    tool_name, _, json_part = body.partition(" ")
    tool_name = tool_name.strip()
    if not tool_name:
        raise ToolError("ACTION line is missing a tool name")

    json_part = json_part.strip()
    if not json_part:
        args: dict[str, Any] = {}
    else:
        try:
            parsed = json.loads(json_part)
        except json.JSONDecodeError as exc:
            raise ToolError(f"Invalid JSON args for tool '{tool_name}': {json_part!r}") from exc
        if not isinstance(parsed, dict):
            raise ToolError(f"Tool args must be a JSON object, got: {json_part!r}")
        args = parsed

    return _ParsedReply(is_action=True, tool=tool_name, args=args)


class Agent:
    """A minimal ReAct agent that loops over LLM-selected tool calls."""

    def __init__(self, provider: LLMProvider, registry: ToolRegistry, max_steps: int = 5) -> None:
        """Initialise the agent.

        Args:
            provider: The chat-completion provider that drives reasoning.
            registry: The tools the agent may call.
            max_steps: Maximum number of ACTION iterations before forcing a stop.
        """
        self._provider = provider
        self._registry = registry
        self._max_steps = max(1, max_steps)

    def _system_prompt(self) -> str:
        """Render the system prompt embedding the available tool catalogue."""
        return _SYSTEM_TEMPLATE.format(tools=self._registry.describe() or "(no tools available)")

    async def run(self, question: str) -> AgentResult:
        """Run the agent until it produces a final answer or runs out of steps.

        Args:
            question: The user's question.

        Returns:
            An :class:`AgentResult` with the answer, the action trace and a
            ``finished`` flag.
        """
        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": question},
        ]
        steps: list[dict[str, Any]] = []

        for step in range(self._max_steps):
            reply = await self._provider.generate(messages)
            messages.append({"role": "assistant", "content": reply})
            parsed = _parse_reply(reply)

            if not parsed.is_action:
                logger.info("Agent finished after %d step(s)", step)
                return AgentResult(answer=parsed.final_text, steps=steps, finished=True)

            observation = await self._run_tool(parsed.tool, parsed.args)
            steps.append({"tool": parsed.tool, "args": parsed.args, "observation": observation})
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

        logger.warning("Agent hit max_steps=%d without a FINAL answer", self._max_steps)
        last_observation = steps[-1]["observation"] if steps else ""
        return AgentResult(answer=str(last_observation), steps=steps, finished=False)

    async def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        """Look up and execute a tool, returning its observation string.

        A :class:`ToolError` is caught and turned into an observation so the
        model can recover on the next turn instead of crashing the loop.

        Args:
            name: The tool name from the ACTION line.
            args: The parsed JSON arguments.

        Returns:
            The tool's result, or an error notice the model can react to.
        """
        try:
            tool = self._registry.get(name)
            return await tool.run(args)
        except ToolError as exc:
            logger.warning("Tool '%s' failed: %s", name, exc.message)
            return f"ERROR: {exc.message}"
