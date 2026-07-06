"""Agent package: a dependency-free ReAct tool-using loop over an LLMProvider."""

from app.agent.base import Tool, ToolError
from app.agent.loop import Agent, AgentResult
from app.agent.registry import ToolRegistry
from app.agent.tools import CalculatorTool, RetrievalTool, safe_eval

__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "CalculatorTool",
    "RetrievalTool",
    "safe_eval",
    "Agent",
    "AgentResult",
]
