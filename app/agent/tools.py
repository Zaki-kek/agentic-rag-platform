"""Built-in agent tools: safe arithmetic and RAG retrieval."""

from __future__ import annotations

import ast
import operator
from collections.abc import Awaitable, Callable
from typing import Any

from app.agent.base import ToolError
from app.core import get_logger

logger = get_logger(__name__)

# An async callable matching RagPipeline.retrieve: (query, k) -> list[Hit].
RetrieveFn = Callable[[str, int], Awaitable[list[Any]]]

# Whitelisted binary and unary operators for the arithmetic evaluator.
_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node.

    Args:
        node: A node from a parsed expression tree.

    Returns:
        The numeric value of the sub-expression.

    Raises:
        ToolError: If the node uses a construct outside the safe whitelist.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"Only numeric literals are allowed, got: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise ToolError(f"Unary operator not allowed: {type(node.op).__name__}")
        return unary(_eval_node(node.operand))
    raise ToolError(f"Unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression over +, -, *, /, parentheses and numbers.

    This never calls :func:`eval`; it parses the expression with :mod:`ast` and
    walks a strict node whitelist, so names, calls, attributes and other code
    cannot execute.

    Args:
        expression: The arithmetic expression to evaluate.

    Returns:
        The computed value as a float.

    Raises:
        ToolError: If the expression is empty, malformed, or uses unsafe syntax.
    """
    if not expression or not expression.strip():
        raise ToolError("Empty expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Could not parse expression: {expression!r}") from exc
    try:
        return _eval_node(tree)
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero") from exc


class CalculatorTool:
    """Evaluate arithmetic expressions safely (no arbitrary code execution)."""

    name = "calculator"
    description = 'Evaluate arithmetic. Args: {"expression": "2+2*3"}. Supports + - * / and parentheses.'

    async def run(self, args: dict[str, Any]) -> str:
        """Compute the expression in ``args['expression']``.

        Args:
            args: Must contain a string ``expression``.

        Returns:
            The result rendered without a trailing ``.0`` for whole numbers.

        Raises:
            ToolError: If ``expression`` is missing or evaluates unsafely.
        """
        expression = args.get("expression")
        if not isinstance(expression, str):
            raise ToolError("calculator requires a string 'expression' argument")
        value = safe_eval(expression)
        if value.is_integer():
            return str(int(value))
        return repr(value)


class RetrievalTool:
    """Retrieve the most relevant document chunks for a query via RAG."""

    name = "retrieval"
    description = 'Search the knowledge base. Args: {"query": "...", "k": 3}. Returns top chunk texts.'

    def __init__(self, retrieve: RetrieveFn, default_k: int = 3, max_chars: int = 1500) -> None:
        """Initialise with a retrieval callable.

        Args:
            retrieve: An async ``(query, k) -> list[Hit]`` callable, typically
                ``RagPipeline.retrieve``.
            default_k: Number of chunks to fetch when ``k`` is not supplied.
            max_chars: Cap on the concatenated observation length.
        """
        self._retrieve = retrieve
        self._default_k = default_k
        self._max_chars = max_chars

    async def run(self, args: dict[str, Any]) -> str:
        """Retrieve and concatenate the top chunk texts for ``args['query']``.

        Args:
            args: Must contain a string ``query``; may contain an int ``k``.

        Returns:
            The joined chunk texts, or a no-results notice.

        Raises:
            ToolError: If ``query`` is missing or not a string.
        """
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("retrieval requires a non-empty string 'query' argument")
        raw_k = args.get("k", self._default_k)
        try:
            k = int(raw_k)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"retrieval 'k' must be an integer, got: {raw_k!r}") from exc
        k = max(1, k)

        hits = await self._retrieve(query, k)
        if not hits:
            return "No relevant context found."
        joined = "\n\n".join(hit.text for hit in hits)
        return joined[: self._max_chars]
