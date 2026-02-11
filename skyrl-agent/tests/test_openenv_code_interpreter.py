"""Tests for OpenEnvCodeInterpreter tool (written first, TDD RED phase)."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import requests


class TestModuleImport(unittest.TestCase):
    """Module and class are importable."""

    def test_import_module(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        self.assertIsNotNone(OpenEnvCodeInterpreter)

    def test_import_from_package(self):
        from skyrl_agent.tools import OpenEnvCodeInterpreter

        self.assertIsNotNone(OpenEnvCodeInterpreter)


class TestToolRegistration(unittest.TestCase):
    """Registered as 'openenv_code_interpreter' in TOOL_REGISTRY, coexists with 'code_interpreter'."""

    def test_registered_in_tool_registry(self):
        from skyrl_agent.tools.base import TOOL_REGISTRY

        # Force import so decorators fire
        import skyrl_agent.tools.openenv_code_interpreter  # noqa: F401

        self.assertIn("openenv_code_interpreter", TOOL_REGISTRY)

    def test_coexists_with_code_interpreter(self):
        from skyrl_agent.tools.base import TOOL_REGISTRY

        import skyrl_agent.tools.sandbox_fusion  # noqa: F401
        import skyrl_agent.tools.openenv_code_interpreter  # noqa: F401

        self.assertIn("code_interpreter", TOOL_REGISTRY)
        self.assertIn("openenv_code_interpreter", TOOL_REGISTRY)

    def test_registry_value_is_class(self):
        from skyrl_agent.tools.base import TOOL_REGISTRY
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        self.assertIs(TOOL_REGISTRY["openenv_code_interpreter"], OpenEnvCodeInterpreter)


class TestPostProcessCodeReuse(unittest.TestCase):
    """_post_process_code is the exact same function object as CodeInterpreter's."""

    def test_identity_check(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter
        from skyrl_agent.tools.sandbox_fusion import CodeInterpreter

        self.assertIs(
            OpenEnvCodeInterpreter._post_process_code,
            CodeInterpreter._post_process_code,
        )


class TestToolSchema(unittest.TestCase):
    """name, description, parameters schema has only 'code' (no 'language'), tool is instantiable."""

    def test_name(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        self.assertEqual(OpenEnvCodeInterpreter.name, "openenv_code_interpreter")

    def test_description_nonempty(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        self.assertTrue(len(OpenEnvCodeInterpreter.description) > 0)

    def test_parameters_has_code(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        props = OpenEnvCodeInterpreter.parameters["properties"]
        self.assertIn("code", props)

    def test_parameters_no_language(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        props = OpenEnvCodeInterpreter.parameters["properties"]
        self.assertNotIn("language", props)

    def test_code_is_required(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        self.assertIn("code", OpenEnvCodeInterpreter.parameters["required"])

    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_instantiable(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        tool = OpenEnvCodeInterpreter()
        self.assertIsNotNone(tool)


class TestHttpRequest(unittest.TestCase):
    """Correct URL, JSON body, _post_process_code applied before sending, trailing slash handled."""

    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_correct_url_and_body(self, mock_post):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}
        }
        mock_post.return_value = mock_resp

        tool = OpenEnvCodeInterpreter()
        tool.call({"code": "print(42)"})

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "http://localhost:8000/step")
        body = call_args[1]["json"]
        self.assertIn("action", body)
        self.assertIn("code", body["action"])
        self.assertEqual(body["timeout_s"], 30.0)

    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_code_goes_through_post_process(self, mock_post):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}
        }
        mock_post.return_value = mock_resp

        tool = OpenEnvCodeInterpreter()
        # Markdown-wrapped code should be stripped by _post_process_code
        tool.call({"code": "```python\nx = 42\n```"})

        call_args = mock_post.call_args
        sent_code = call_args[1]["json"]["action"]["code"]
        # Markdown delimiters should be removed
        self.assertNotIn("```", sent_code)

    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000/"})
    def test_trailing_slash_handled(self, mock_post):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "observation": {"stdout": "ok\n", "stderr": "", "exit_code": 0}
        }
        mock_post.return_value = mock_resp

        tool = OpenEnvCodeInterpreter()
        tool.call({"code": "print('ok')"})

        call_args = mock_post.call_args
        url = call_args[0][0]
        # Should NOT have double slash: http://localhost:8000//step
        self.assertNotIn("//step", url)
        self.assertTrue(url.endswith("/step"))


class TestResponseParsing(unittest.TestCase):
    """stdout-only, stdout+stderr concatenated, non-zero exit_code still returns output."""

    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_stdout_only(self, mock_post):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "observation": {"stdout": "hello\n", "stderr": "", "exit_code": 0}
        }
        mock_post.return_value = mock_resp

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": "print('hello')"})
        self.assertEqual(result, "hello\n")

    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_stdout_plus_stderr(self, mock_post):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "observation": {
                "stdout": "partial output\n",
                "stderr": "NameError: name 'x' is not defined\n",
                "exit_code": 1,
            }
        }
        mock_post.return_value = mock_resp

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": "print('partial output'); x"})
        self.assertEqual(
            result, "partial output\nNameError: name 'x' is not defined\n"
        )

    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_nonzero_exit_code_still_returns_output(self, mock_post):
        """Matches sandbox_fusion behavior: 'Finished' status returns stdout+stderr regardless of exit code."""
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "observation": {
                "stdout": "",
                "stderr": "Traceback (most recent call last):\n  ZeroDivisionError\n",
                "exit_code": 1,
            }
        }
        mock_post.return_value = mock_resp

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": "1/0"})
        # Should return the traceback as a string, NOT an error dict
        self.assertIsInstance(result, str)
        self.assertIn("ZeroDivisionError", result)


class TestRetryLogic(unittest.TestCase):
    """Retries on ConnectionError, Timeout, 5xx; succeeds after transient failures."""

    @patch("skyrl_agent.tools.openenv_code_interpreter.time.sleep")
    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_post.side_effect = requests.ConnectionError("Connection refused")

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": "print(1)"})

        self.assertEqual(mock_post.call_count, 5)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    @patch("skyrl_agent.tools.openenv_code_interpreter.time.sleep")
    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_retries_on_timeout(self, mock_post, mock_sleep):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_post.side_effect = requests.Timeout("Request timed out")

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": "print(1)"})

        self.assertEqual(mock_post.call_count, 5)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    @patch("skyrl_agent.tools.openenv_code_interpreter.time.sleep")
    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_succeeds_after_transient_failures(self, mock_post, mock_sleep):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        good_resp = MagicMock()
        good_resp.status_code = 200
        good_resp.json.return_value = {
            "observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}
        }

        # Fail twice, then succeed
        mock_post.side_effect = [
            requests.ConnectionError("fail 1"),
            requests.ConnectionError("fail 2"),
            good_resp,
        ]

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": "print(42)"})

        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(result, "42\n")

    @patch("skyrl_agent.tools.openenv_code_interpreter.time.sleep")
    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_retries_on_5xx(self, mock_post, mock_sleep):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        bad_resp = MagicMock()
        bad_resp.status_code = 502
        bad_resp.text = "Bad Gateway"
        mock_post.return_value = bad_resp

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": "print(1)"})

        self.assertEqual(mock_post.call_count, 5)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)


class TestErrorHandling(unittest.TestCase):
    """Missing OPENENV_URL, empty code, missing code param, params-as-JSON-string, 4xx no retry."""

    def test_missing_openenv_url(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENENV_URL if set
            os.environ.pop("OPENENV_URL", None)
            tool = OpenEnvCodeInterpreter()
            result = tool.call({"code": "print(1)"})
            self.assertIsInstance(result, dict)
            self.assertIn("error", result)

    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_empty_code(self, mock_post):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": ""})
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        # Should NOT have made any HTTP requests
        mock_post.assert_not_called()

    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_missing_code_param(self):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"not_code": "print(1)"})
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_params_as_json_string(self, mock_post):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}
        }
        mock_post.return_value = mock_resp

        tool = OpenEnvCodeInterpreter()
        result = tool.call(json.dumps({"code": "print(42)"}))

        self.assertEqual(result, "42\n")

    @patch("skyrl_agent.tools.openenv_code_interpreter.time.sleep")
    @patch("skyrl_agent.tools.openenv_code_interpreter.requests.post")
    @patch.dict(os.environ, {"OPENENV_URL": "http://localhost:8000"})
    def test_4xx_does_not_retry(self, mock_post, mock_sleep):
        from skyrl_agent.tools.openenv_code_interpreter import OpenEnvCodeInterpreter

        bad_resp = MagicMock()
        bad_resp.status_code = 400
        bad_resp.text = "Bad Request"
        mock_post.return_value = bad_resp

        tool = OpenEnvCodeInterpreter()
        result = tool.call({"code": "print(1)"})

        # 4xx should NOT retry
        self.assertEqual(mock_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
