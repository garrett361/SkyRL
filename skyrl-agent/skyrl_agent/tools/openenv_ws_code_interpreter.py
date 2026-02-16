from __future__ import annotations

import ast
import asyncio
import logging
import re
from typing import TYPE_CHECKING, Optional, Union

from skyrl_agent.tools.base import BaseTool, register_tool

if TYPE_CHECKING:
    from coding_env.client import CodingEnv

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60.0


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code block delimiters (```python ... ```)."""
    code = re.sub(r"^```\w*\s*\n?", "", code.strip(), flags=re.MULTILINE)
    code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)
    code = code.replace("```", "")
    return code.strip()


def _wrap_last_expr_in_print(code: str) -> str:
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

    # Already a print() call — nothing to do
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


def _post_process_code(code: str) -> str:
    """Clean markdown fences and ensure the last expression is printed."""
    if not code:
        return code
    code = _strip_markdown_fences(code)
    code = _wrap_last_expr_in_print(code)
    return code


@register_tool("openenv_ws_code_interpreter")
class OpenEnvWSCodeInterpreter(BaseTool):
    name = "openenv_ws_code_interpreter"
    description = (
        "Executes Python code in a sandboxed environment. "
        "IMPORTANT: Each invocation runs in a fresh namespace — variables, "
        "functions, and imports from previous calls are NOT available. You "
        "must re-import modules and redefine any functions in every call. "
        "The last expression in the code is automatically printed (like a "
        "Python REPL), so you do not need to wrap it in print()."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute. Each call starts with an empty "
                    "namespace, so always include all necessary imports and "
                    "definitions."
                ),
            },
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self._session: Optional[CodingEnv] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_ws_session(
        self, session: CodingEnv, event_loop: asyncio.AbstractEventLoop
    ) -> None:
        self._session = session
        self._event_loop = event_loop

    def close_ws_session(self) -> None:
        self._session = None
        self._event_loop = None

    def _execute_code(self, code: str) -> str:
        if self._session is None or self._event_loop is None:
            raise RuntimeError(
                "WebSocket session not initialized. "
                "Call set_ws_session() before executing code."
            )

        from coding_env.models import CodeAction

        action = CodeAction(code=code)
        future = asyncio.run_coroutine_threadsafe(
            self._session.step(action), self._event_loop
        )
        result = future.result(timeout=DEFAULT_TIMEOUT_S)
        return result.observation.stdout + result.observation.stderr

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, dict]:
        try:
            params = self._verify_json_format_args(params)
        except (ValueError, Exception) as exc:
            return {"error": f"Invalid parameters: {exc}"}

        code = params.get("code", "")
        code = _post_process_code(code)

        if not code:
            return {"error": "Code parameter is required."}

        try:
            result = self._execute_code(code)
        except RuntimeError as exc:
            return {"error": str(exc)}

        return result
