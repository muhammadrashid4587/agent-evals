"""A tiny calculator agent, used as an example agent under test.

It finds the first arithmetic expression in the prompt (integers/floats,
``+ - * /`` and parentheses) and evaluates it with a safe AST walk — no
``eval()``. It returns the result as a plain string, or ``"I don't know"``
when the prompt has no arithmetic in it.
"""

from __future__ import annotations

import ast
import re

_EXPR_RE = re.compile(r"\(?-?\d[\d\s+\-*/().]*\d\)?")

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculate(prompt: str) -> str:
    """Evaluate the arithmetic in ``prompt``; ``"I don't know"`` if none."""
    match = _EXPR_RE.search(prompt)
    if match is None:
        return "I don't know"
    expr = match.group(0).strip()
    if not any(ch.isdigit() for ch in expr):
        return "I don't know"
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval(tree)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return "I don't know"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def agent(prompt: str) -> str:
    """The agent callable the eval runner calls."""
    return calculate(prompt)
