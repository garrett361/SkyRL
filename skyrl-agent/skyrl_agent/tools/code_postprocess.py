"""Shared code post-processing utilities for code interpreter tools."""

from __future__ import annotations

import ast
import re


def strip_markdown_fences(code: str) -> str:
    """Remove markdown code block delimiters (```python ... ```)."""
    code = re.sub(r"^```\w*\s*\n?", "", code.strip(), flags=re.MULTILINE)
    code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)
    code = code.replace("```", "")
    return code.strip()


def wrap_last_expr_in_print(code: str) -> str:
    """If the last statement is a bare expression, wrap it in print().

    Uses AST parsing to correctly handle trailing comments, multiline
    expressions, and other edge cases that naive string manipulation
    gets wrong (e.g. ``x  # comment`` -> ``print(x  # comment)`` breaks).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    if not tree.body:
        return code

    last_stmt = tree.body[-1]
    if not isinstance(last_stmt, ast.Expr):
        return code

    # Already a print() call
    if (
        isinstance(last_stmt.value, ast.Call)
        and isinstance(last_stmt.value.func, ast.Name)
        and last_stmt.value.func.id == "print"
    ):
        return code

    # Use ast.unparse to get the expression without comments, then wrap.
    # This avoids the bug where trailing comments swallow the closing paren.
    expr_source = ast.unparse(last_stmt.value)
    wrapped = f"print({expr_source})"

    lines = code.split("\n")
    start = last_stmt.lineno - 1
    end = last_stmt.end_lineno  # end_lineno is 1-indexed inclusive
    lines[start:end] = [wrapped]
    return "\n".join(lines)


def post_process_code(code: str) -> str:
    """Clean markdown fences and ensure the last expression is printed."""
    if not code:
        return code
    code = strip_markdown_fences(code)
    code = wrap_last_expr_in_print(code)
    return code
