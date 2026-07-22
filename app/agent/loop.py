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
from app.observability.tracer import NoOpTracer, Tracer

logger = get_logger(__name__)

_ACTION_PREFIX = "ACTION:"
_FINAL_PREFIX = "FINAL:"

# Rough token estimate: ~4 characters per token. Deliberately cheap and
# provider-agnostic; good enough to enforce a run budget without a tokenizer.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Roughly estimate the token count of ``text`` as ``len(text) // 4``.

    This is a deterministic, dependency-free heuristic used only for budgeting;
    it is not meant to match any specific provider's tokenizer.

    Args:
        text: The text to estimate.

    Returns:
        A non-negative integer token estimate.
    """
    return len(text) // _CHARS_PER_TOKEN

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
            stopped early (``max_steps`` exhausted or token budget exceeded).
        tokens_used: Cumulative rough token estimate spent on LLM calls.
        stop_reason: Why the loop ended: ``"final"``, ``"max_steps"`` or
            ``"budget"``.
    """

    answer: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    finished: bool
    tokens_used: int = 0
    stop_reason: str = "final"


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

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_steps: int = 5,
        *,
        max_tokens: int = 0,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            provider: The chat-completion provider that drives reasoning.
            registry: The tools the agent may call.
            max_steps: Maximum number of ACTION iterations before forcing a stop.
            max_tokens: Rough token budget for the whole run (prompt plus reply,
                estimated as characters / 4 per LLM call). ``0`` disables the
                budget; the run is still capped by ``max_steps``.
            tracer: Optional tracer; one span is opened per agent step. Defaults
                to a no-op tracer.
        """
        self._provider = provider
        self._registry = registry
        self._max_steps = max(1, max_steps)
        self._max_tokens = max(0, max_tokens)
        self._tracer = tracer or NoOpTracer()

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
        tokens_used = 0

        for step in range(self._max_steps):
            with self._tracer.span("agent.step", step=step) as handle:
                # Estimate the prompt cost before the call and stop if a budget
                # is set and already exhausted - no point issuing another call.
                prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
                if self._over_budget(tokens_used + prompt_tokens):
                    handle.set(stop_reason="budget", tokens_used=tokens_used)
                    logger.warning(
                        "Agent stopped at step %d: token budget %d exceeded (used ~%d)",
                        step,
                        self._max_tokens,
                        tokens_used + prompt_tokens,
                    )
                    return self._budget_stop(steps, tokens_used)

                reply = await self._provider.generate(messages)
                tokens_used += prompt_tokens + estimate_tokens(reply)
                messages.append({"role": "assistant", "content": reply})
                parsed = _parse_reply(reply)
                handle.set(tokens_used=tokens_used, is_action=parsed.is_action)

                if not parsed.is_action:
                    handle.set(stop_reason="final")
                    logger.info("Agent finished after %d step(s)", step)
                    return AgentResult(
                        answer=parsed.final_text,
                        steps=steps,
                        finished=True,
                        tokens_used=tokens_used,
                        stop_reason="final",
                    )

                observation = await self._run_tool(parsed.tool, parsed.args)
                steps.append(
                    {"tool": parsed.tool, "args": parsed.args, "observation": observation}
                )
                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
                handle.set(tool=parsed.tool)

        logger.warning("Agent hit max_steps=%d without a FINAL answer", self._max_steps)
        last_observation = steps[-1]["observation"] if steps else ""
        return AgentResult(
            answer=str(last_observation),
            steps=steps,
            finished=False,
            tokens_used=tokens_used,
            stop_reason="max_steps",
        )

    def _over_budget(self, tokens: int) -> bool:
        """Return ``True`` if a positive budget is set and ``tokens`` exceeds it."""
        return self._max_tokens > 0 and tokens > self._max_tokens

    def _budget_stop(self, steps: list[dict[str, Any]], tokens_used: int) -> AgentResult:
        """Build the :class:`AgentResult` returned when the token budget is hit.

        Args:
            steps: The action trace accumulated so far.
            tokens_used: Rough token estimate spent before stopping.

        Returns:
            An unfinished :class:`AgentResult` with ``stop_reason="budget"`` and
            an answer explaining the stop (falling back to the last observation).
        """
        last_observation = str(steps[-1]["observation"]) if steps else ""
        reason = f"Stopped: token budget of {self._max_tokens} exceeded."
        return AgentResult(
            answer=last_observation or reason,
            steps=steps,
            finished=False,
            tokens_used=tokens_used,
            stop_reason="budget",
        )

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
