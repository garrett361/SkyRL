"""Tests for OpenEnvWSCodeInterpreter tool."""

import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl_agent.tools.base import TOOL_REGISTRY
from skyrl_agent.tools.code_postprocess import (
    post_process_code as _post_process_code,
    strip_markdown_fences as _strip_markdown_fences,
    wrap_last_expr_in_print as _wrap_last_expr_in_print,
)
from skyrl_agent.tools.openenv_ws_code_interpreter import OpenEnvWSCodeInterpreter


@pytest.fixture
def tool():
    return OpenEnvWSCodeInterpreter()


def _make_step_result(stdout="", stderr=""):
    obs = MagicMock()
    obs.stdout = stdout
    obs.stderr = stderr
    result = MagicMock()
    result.observation = obs
    return result


def _make_session_and_loop(step_result):
    """Create a mock CodingEnv session and a real event loop in a background thread."""
    session = MagicMock()
    session.step = AsyncMock(return_value=step_result)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return session, loop, thread


class TestRegistrationAndSchema:
    def test_registered_in_tool_registry(self):
        assert "openenv_ws_code_interpreter" in TOOL_REGISTRY

    def test_tool_name(self):
        assert OpenEnvWSCodeInterpreter.name == "openenv_ws_code_interpreter"

    def test_required_parameters(self):
        assert "code" in OpenEnvWSCodeInterpreter.parameters["required"]


class TestSessionManagement:
    def test_call_without_session_returns_error(self, tool):
        result = tool.call({"code": "print(1)"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "session not initialized" in result["error"].lower()

    def test_set_and_close_session(self, tool):
        mock_session = MagicMock()
        mock_loop = MagicMock()

        tool.set_ws_session(mock_session, mock_loop)
        assert tool._session is mock_session
        assert tool._event_loop is mock_loop

        tool.close_ws_session()
        assert tool._session is None
        assert tool._event_loop is None


class TestCodeExecution:
    def test_call_with_mocked_session(self, tool):
        step_result = _make_step_result(stdout="42\n")
        session, loop, thread = _make_session_and_loop(step_result)
        try:
            tool.set_ws_session(session, loop)
            result = tool.call({"code": "print(42)"})
            assert result == "42\n"
            session.step.assert_called_once()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    def test_code_post_processing(self, tool):
        step_result = _make_step_result(stdout="42\n")
        session, loop, thread = _make_session_and_loop(step_result)
        try:
            tool.set_ws_session(session, loop)
            tool.call({"code": "```python\nprint(42)\n```"})
            action_arg = session.step.call_args[0][0]
            assert "```" not in action_arg.code
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    def test_empty_code_returns_error(self, tool):
        result = tool.call({"code": ""})
        assert isinstance(result, dict)
        assert "error" in result

    def test_stderr_concatenated(self, tool):
        step_result = _make_step_result(
            stdout="partial\n", stderr="NameError: name 'x' is not defined\n"
        )
        session, loop, thread = _make_session_and_loop(step_result)
        try:
            tool.set_ws_session(session, loop)
            result = tool.call({"code": "print('partial'); x"})
            assert "partial\n" in result
            assert "NameError" in result
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    def test_params_as_json_string(self, tool):
        step_result = _make_step_result(stdout="42\n")
        session, loop, thread = _make_session_and_loop(step_result)
        try:
            tool.set_ws_session(session, loop)
            result = tool.call(json.dumps({"code": "print(42)"}))
            assert result == "42\n"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()


class TestStripMarkdownFences:
    def test_python_fence(self):
        assert _strip_markdown_fences("```python\nprint(1)\n```") == "print(1)"

    def test_py_fence(self):
        assert _strip_markdown_fences("```py\nx = 1\n```") == "x = 1"

    def test_bare_fence(self):
        assert _strip_markdown_fences("```\nfoo\n```") == "foo"

    def test_no_fence(self):
        assert _strip_markdown_fences("x = 1") == "x = 1"

    def test_empty(self):
        assert _strip_markdown_fences("") == ""

    def test_middle_backticks_removed(self):
        result = _strip_markdown_fences("```py\nx = 1\n```\n```py\ny = 2\n```")
        assert "```" not in result


class TestWrapLastExprInPrint:
    """Tests for the AST-based last-expression print wrapping."""

    def test_trailing_comment_does_not_break(self):
        """The original bug: trailing # comment caused SyntaxError."""
        code = "results[2]  # third such n"
        result = _wrap_last_expr_in_print(code)
        assert result == "print(results[2])"
        compile(result, "<test>", "exec")

    def test_comment_with_parens(self):
        """Comment containing parens must not confuse the wrapper."""
        code = "x  # the final answer for part (a)"
        result = _wrap_last_expr_in_print(code)
        assert result == "print(x)"
        compile(result, "<test>", "exec")

    def test_bare_expression_wrapped(self):
        assert _wrap_last_expr_in_print("results[2]") == "print(results[2])"

    def test_already_print_not_double_wrapped(self):
        code = "print(results[2])"
        assert _wrap_last_expr_in_print(code) == code

    def test_assignment_not_wrapped(self):
        code = "x = 42"
        assert _wrap_last_expr_in_print(code) == code

    def test_for_loop_not_wrapped(self):
        code = "for i in range(3):\n    x = i"
        assert _wrap_last_expr_in_print(code) == code

    def test_function_def_not_wrapped(self):
        code = "def foo():\n    return 42"
        assert _wrap_last_expr_in_print(code) == code

    def test_function_then_call_wrapped(self):
        code = "def foo():\n    return 42\n\nfoo()"
        result = _wrap_last_expr_in_print(code)
        assert "print(foo())" in result
        compile(result, "<test>", "exec")

    def test_multi_statement_only_last_wrapped(self):
        code = "import math\nx = math.sqrt(2)\nx"
        result = _wrap_last_expr_in_print(code)
        assert result == "import math\nx = math.sqrt(2)\nprint(x)"

    def test_syntax_error_passthrough(self):
        code = "if True\n  pass"
        assert _wrap_last_expr_in_print(code) == code

    def test_empty_code(self):
        assert _wrap_last_expr_in_print("") == ""

    def test_multi_line_expression(self):
        code = "x = 1\n(a +\n b +\n c)"
        result = _wrap_last_expr_in_print(code)
        compile(result, "<test>", "exec")
        assert result.startswith("x = 1\nprint(")

    def test_string_containing_hash(self):
        """Hash inside a string is not a comment."""
        code = 'len("hello # world")'
        result = _wrap_last_expr_in_print(code)
        compile(result, "<test>", "exec")
        assert "print(" in result

    def test_variable_starting_with_print(self):
        """Old regex-based code would skip 'print_count' due to startswith."""
        code = "print_count"
        result = _wrap_last_expr_in_print(code)
        assert result == "print(print_count)"

    def test_method_call_wrapped(self):
        code = "df.head()"
        result = _wrap_last_expr_in_print(code)
        assert result == "print(df.head())"

    def test_return_outside_function_passthrough(self):
        """'return 42' is a SyntaxError at module level, should pass through."""
        code = "return 42"
        assert _wrap_last_expr_in_print(code) == code


class TestPostProcessCode:
    """Integration tests for the full pipeline."""

    def test_markdown_and_trailing_comment(self):
        code = "```py\nx = [1, 2, 3]\nsum(x)  # total\n```"
        result = _post_process_code(code)
        compile(result, "<test>", "exec")
        assert "print(sum(x))" in result
        assert "#" not in result

    def test_markdown_with_print(self):
        code = "```python\nprint(42)\n```"
        result = _post_process_code(code)
        assert result == "print(42)"

    def test_empty(self):
        assert _post_process_code("") == ""

    def test_plain_code_no_markdown(self):
        code = "x = 1\nx + 2"
        result = _post_process_code(code)
        assert result == "x = 1\nprint(x + 2)"
