"""Tools the agent can call.

Each tool is a simple callable-ish object exposing a `name`, a `description`
(shown to the "LLM" / used for routing), and a `run(query)` method that
returns a string observation.

- `CalculatorTool` safely evaluates arithmetic expressions using Python's
  `ast` module instead of raw `eval`, so it can't execute arbitrary code.
- `KnowledgeLookupTool` is a mock/local stand-in for a real search or
  vector-lookup tool: a small in-memory dict of facts/Q&A pairs. In a real
  system this would call a search API, a vector database, or similar.
"""
from __future__ import annotations

import ast
import operator
from typing import Callable


class ToolError(Exception):
    """Raised when a tool cannot produce a result for the given input."""


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------

# Only these operators/node types are allowed in expressions. Anything else
# (names, function calls, attribute access, subscripts, comprehensions, ...)
# is rejected, so arbitrary code execution isn't possible.
_ALLOWED_BINOPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type, Callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ToolError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_BINOPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ToolError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_safe_eval(node.operand))
    raise ToolError(f"Unsupported expression: {type(node).__name__}")


class CalculatorTool:
    """Safely evaluates arithmetic expressions like '2 + 3 * (4 - 1)'.

    Uses Python's `ast` module to parse the expression into a syntax tree
    and only walks a small allow-list of numeric/operator node types
    (Constant, BinOp, UnaryOp with +,-,*,/,//,%,**). No `eval`/`exec` is
    used, so names, calls, imports, and other arbitrary code cannot run.
    """

    name = "calculator"
    description = "Evaluates arithmetic expressions, e.g. '2 + 2' or '(3 + 4) * 5'."

    def run(self, query: str) -> str:
        try:
            tree = ast.parse(query, mode="eval")
            result = _safe_eval(tree)
        except ZeroDivisionError as exc:
            raise ToolError("Division by zero") from exc
        except (SyntaxError, ToolError) as exc:
            raise ToolError(f"Could not safely evaluate {query!r}: {exc}") from exc
        return str(result)


# ---------------------------------------------------------------------------
# Knowledge lookup tool (mock/local stand-in for a real search / vector DB)
# ---------------------------------------------------------------------------

# NOTE: This is a deliberately tiny, hard-coded in-memory dictionary standing
# in for a real knowledge source (a search API, a vector database over a
# document corpus, etc.). It exists purely so the agent demo has a second,
# non-numeric tool to route to without any external dependency or network
# call.
_FACTS: dict[str, str] = {
    "capital of france": "The capital of France is Paris.",
    "capital of japan": "The capital of Japan is Tokyo.",
    "largest planet": "The largest planet in the solar system is Jupiter.",
    "speed of light": "The speed of light in a vacuum is about 299,792,458 meters per second.",
    "author of hamlet": "Hamlet was written by William Shakespeare.",
    "python creator": "Python was created by Guido van Rossum.",
    "tallest mountain": "The tallest mountain above sea level is Mount Everest.",
}


class KnowledgeLookupTool:
    """Mock/local knowledge-lookup tool.

    Stands in for a real search or vector-lookup tool. Looks up a small
    in-memory dict of facts by matching keywords from the query against the
    dict's keys, returning the best (most-overlapping) match, or a clear
    "not found" message if nothing matches well enough.
    """

    name = "knowledge_lookup"
    description = "Looks up simple known facts, e.g. 'What is the capital of France?'."

    def run(self, query: str) -> str:
        normalized = query.lower().strip().rstrip("?.!")
        query_words = set(normalized.split())

        best_key = None
        best_overlap = 0
        for key in _FACTS:
            key_words = set(key.split())
            overlap = len(key_words & query_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key

        if best_key is not None and best_overlap >= 2:
            return _FACTS[best_key]

        raise ToolError(f"No known fact matches the query {query!r}.")
