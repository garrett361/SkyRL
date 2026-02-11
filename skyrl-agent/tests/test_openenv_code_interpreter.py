"""Tests for OpenEnvCodeInterpreter tool."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter
from skyrl_agent.tools.base import TOOL_REGISTRY

# Patch some of the modules imported in openenv_code_interpreter
MOCK_MODULE = "skyrl_agent.tools.openenv_code_interpreter"


def _ok_response(stdout="", stderr="", exit_code=0):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "observation": {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
    }
    return resp


@pytest.fixture
def tool():
    with patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"}):
        yield OpenEnvCodeInterpreter()


class TestRegistrationAndSchema:
    def test_registered_in_tool_registry(self):
        assert "openenv_code_interpreter" in TOOL_REGISTRY
        assert "code_interpreter" in TOOL_REGISTRY

    def test_name(self):
        assert OpenEnvCodeInterpreter.name == "openenv_code_interpreter"

    def test_code_is_required(self):
        assert "code" in OpenEnvCodeInterpreter.parameters["required"]


class TestHttpRequest:
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_correct_url_and_body(self, mock_post, tool):
        mock_post.return_value = _ok_response(stdout="42\n")

        tool.call({"code": "print(42)"})

        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        body = mock_post.call_args[1]["json"]
        assert url == "http://localhost:8000/step"
        assert "code" in body["action"]
        assert body["timeout_s"] == 30.0

    @patch(f"{MOCK_MODULE}.requests.post")
    def test_code_goes_through_post_process(self, mock_post, tool):
        mock_post.return_value = _ok_response(stdout="42\n")

        tool.call({"code": "```python\nx = 42\n```"})

        sent_code = mock_post.call_args[1]["json"]["action"]["code"]
        assert "```" not in sent_code


class TestResponseParsing:
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_stdout_only(self, mock_post, tool):
        mock_post.return_value = _ok_response(stdout="hello\n")
        assert tool.call({"code": "print('hello')"}) == "hello\n"

    @patch(f"{MOCK_MODULE}.requests.post")
    def test_stdout_plus_stderr(self, mock_post, tool):
        mock_post.return_value = _ok_response(
            stdout="partial output\n",
            stderr="NameError: name 'x' is not defined\n",
            exit_code=1,
        )
        result = tool.call({"code": "print('partial output'); x"})
        assert result == "partial output\nNameError: name 'x' is not defined\n"

    @patch(f"{MOCK_MODULE}.requests.post")
    def test_nonzero_exit_code_still_returns_output(self, mock_post, tool):
        """HTTP 200 with non-zero exit_code returns stdout+stderr (not an error dict)."""
        mock_post.return_value = _ok_response(
            stderr="Traceback (most recent call last):\n  ZeroDivisionError\n",
            exit_code=1,
        )
        result = tool.call({"code": "1/0"})
        assert isinstance(result, str)
        assert "ZeroDivisionError" in result


class TestRetryLogic:
    @patch(f"{MOCK_MODULE}.time.sleep")
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep, tool):
        mock_post.side_effect = requests.ConnectionError("refused")

        result = tool.call({"code": "print(1)"})

        assert mock_post.call_count == 5
        assert isinstance(result, dict) and "error" in result

    @patch(f"{MOCK_MODULE}.time.sleep")
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_retries_on_timeout(self, mock_post, mock_sleep, tool):
        mock_post.side_effect = requests.Timeout("timed out")

        result = tool.call({"code": "print(1)"})

        assert mock_post.call_count == 5
        assert isinstance(result, dict) and "error" in result

    @patch(f"{MOCK_MODULE}.time.sleep")
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_succeeds_after_transient_failures(self, mock_post, mock_sleep, tool):
        mock_post.side_effect = [
            requests.ConnectionError("fail 1"),
            requests.ConnectionError("fail 2"),
            _ok_response(stdout="42\n"),
        ]

        result = tool.call({"code": "print(42)"})

        assert mock_post.call_count == 3
        assert result == "42\n"

    @patch(f"{MOCK_MODULE}.time.sleep")
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_retries_on_5xx(self, mock_post, mock_sleep, tool):
        bad = MagicMock(status_code=502, text="Bad Gateway")
        mock_post.return_value = bad

        result = tool.call({"code": "print(1)"})

        assert mock_post.call_count == 5
        assert isinstance(result, dict) and "error" in result


class TestErrorHandling:
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_empty_code(self, mock_post, tool):
        result = tool.call({"code": ""})
        assert isinstance(result, dict) and "error" in result
        mock_post.assert_not_called()

    def test_missing_code_param(self, tool):
        result = tool.call({"not_code": "print(1)"})
        assert isinstance(result, dict) and "error" in result

    @patch(f"{MOCK_MODULE}.requests.post")
    def test_params_as_json_string(self, mock_post, tool):
        mock_post.return_value = _ok_response(stdout="42\n")
        assert tool.call(json.dumps({"code": "print(42)"})) == "42\n"
