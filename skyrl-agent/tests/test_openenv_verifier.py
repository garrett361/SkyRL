"""Tests for the OpenEnv verifier (call_openenv_api, _process_single_case, check_correctness)."""

from unittest.mock import MagicMock, patch

import requests
from skyrl_agent.tasks.verifiers.openenv.utils import (
    MAX_RETRIES,
    TIMEOUT_ERROR_PREFIX,
    _process_single_case,
    call_openenv_api,
    check_correctness,
)

MOCK_MODULE = "skyrl_agent.tasks.verifiers.openenv.utils"
OPENENV_URL = "http://openenv:8000"


def _ok_response(stdout="", stderr="", exit_code=0):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "observation": {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
    }
    return resp


def _error_response(status_code, text="error"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status.side_effect = requests.HTTPError(
        response=resp, request=MagicMock()
    )
    return resp


class TestCallOpenenvApi:
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_correct_url_and_json_body(self, mock_post):
        mock_post.return_value = _ok_response(stdout="42\n")

        call_openenv_api(OPENENV_URL, "print(42)", timeout_s=15.0)

        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        body = mock_post.call_args[1]["json"]
        assert url == f"{OPENENV_URL}/step"
        assert body == {"action": {"code": "print(42)"}, "timeout_s": 15.0}

    @patch(f"{MOCK_MODULE}.requests.post")
    def test_success_returns_response_dict_and_none_error(self, mock_post):
        mock_post.return_value = _ok_response(stdout="hello\n")

        resp, err = call_openenv_api(OPENENV_URL, "print('hello')")

        assert err is None
        assert resp == {
            "observation": {"stdout": "hello\n", "stderr": "", "exit_code": 0}
        }

    @patch(f"{MOCK_MODULE}.time.sleep")
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.ConnectionError("refused")

        resp, err = call_openenv_api(OPENENV_URL, "print(1)")

        assert mock_post.call_count == MAX_RETRIES
        assert resp is None
        assert err is not None

    @patch(f"{MOCK_MODULE}.time.sleep")
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_retries_on_timeout(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.Timeout("timed out")

        resp, err = call_openenv_api(OPENENV_URL, "print(1)")

        assert mock_post.call_count == MAX_RETRIES
        assert resp is None
        assert err is not None
        assert err.startswith(TIMEOUT_ERROR_PREFIX)

    @patch(f"{MOCK_MODULE}.time.sleep")
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_retries_on_5xx(self, mock_post, mock_sleep):
        mock_post.return_value = _error_response(502, "Bad Gateway")

        resp, err = call_openenv_api(OPENENV_URL, "print(1)")

        assert mock_post.call_count == MAX_RETRIES
        assert resp is None
        assert err is not None

    @patch(f"{MOCK_MODULE}.requests.post")
    def test_no_retry_on_4xx(self, mock_post):
        mock_post.return_value = _error_response(400, "Bad Request")

        resp, err = call_openenv_api(OPENENV_URL, "print(1)")

        assert mock_post.call_count == 1
        assert resp is None
        assert err is not None

    @patch(f"{MOCK_MODULE}.time.sleep")
    @patch(f"{MOCK_MODULE}.requests.post")
    def test_succeeds_after_transient_failures(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            requests.ConnectionError("fail 1"),
            _ok_response(stdout="ok\n"),
        ]

        resp, err = call_openenv_api(OPENENV_URL, "print('ok')")

        assert mock_post.call_count == 2
        assert err is None
        assert resp["observation"]["stdout"] == "ok\n"

    @patch(f"{MOCK_MODULE}.requests.post")
    def test_trailing_slash_handled(self, mock_post):
        mock_post.return_value = _ok_response()

        call_openenv_api(f"{OPENENV_URL}/", "print(1)")

        url = mock_post.call_args[0][0]
        assert url == f"{OPENENV_URL}/step"
        assert "//" not in url.split("://", 1)[1]

    @patch(f"{MOCK_MODULE}.requests.post")
    def test_http_timeout_includes_buffer(self, mock_post):
        mock_post.return_value = _ok_response()

        call_openenv_api(OPENENV_URL, "print(1)", timeout_s=20.0)

        assert mock_post.call_args[1]["timeout"] == 30.0  # 20 + 10


class TestProcessSingleCase:
    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_correct_answer(self, mock_api):
        mock_api.return_value = (
            {"observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}},
            None,
        )

        result, meta = _process_single_case(
            case_index=0,
            stdin_data="5",
            expected_output="42",
            openenv_url=OPENENV_URL,
            generation="print(42)",
            timeout=10,
        )

        assert result is True
        assert meta["status"] == "success"
        assert meta["stdout"] == "42\n"
        assert meta["case_index"] == 0

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_wrong_answer(self, mock_api):
        mock_api.return_value = (
            {"observation": {"stdout": "99\n", "stderr": "", "exit_code": 0}},
            None,
        )

        result, meta = _process_single_case(
            case_index=0,
            stdin_data="5",
            expected_output="42",
            openenv_url=OPENENV_URL,
            generation="print(99)",
            timeout=10,
        )

        assert result is False
        assert meta["status"] == "wrong_answer"

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_runtime_error(self, mock_api):
        mock_api.return_value = (
            {
                "observation": {
                    "stdout": "",
                    "stderr": "ZeroDivisionError",
                    "exit_code": 1,
                }
            },
            None,
        )

        result, meta = _process_single_case(
            case_index=0,
            stdin_data="",
            expected_output="42",
            openenv_url=OPENENV_URL,
            generation="1/0",
            timeout=10,
        )

        assert result == -2
        assert meta["status"] == "runtime_error"

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_api_error(self, mock_api):
        mock_api.return_value = (None, "Connection refused")

        result, meta = _process_single_case(
            case_index=0,
            stdin_data="",
            expected_output="42",
            openenv_url=OPENENV_URL,
            generation="print(42)",
            timeout=10,
        )

        assert result == -1
        assert meta["status"] == "api_error"
        assert meta["api_request_error"] == "Connection refused"

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_timeout_error(self, mock_api):
        mock_api.return_value = (None, f"{TIMEOUT_ERROR_PREFIX}timed out after 10s")

        result, meta = _process_single_case(
            case_index=0,
            stdin_data="",
            expected_output="42",
            openenv_url=OPENENV_URL,
            generation="import time; time.sleep(999)",
            timeout=10,
        )

        assert result == -3
        assert meta["status"] == "timeout"

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_fn_name_wraps_code_with_harness(self, mock_api):
        mock_api.return_value = (
            {"observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}},
            None,
        )

        _process_single_case(
            case_index=0,
            stdin_data="[1, 2]\n39",
            expected_output="42",
            openenv_url=OPENENV_URL,
            generation="def add(a, b): return a + b",
            timeout=10,
            fn_name="add",
        )

        sent_code = mock_api.call_args[0][1]  # second positional arg is code
        assert "add(*_args)" in sent_code
        assert "def add(a, b): return a + b" in sent_code

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_fn_name_embeds_stdin_as_literal(self, mock_api):
        mock_api.return_value = (
            {"observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}},
            None,
        )

        _process_single_case(
            case_index=0,
            stdin_data="[1, 2]\n39",
            expected_output="42",
            openenv_url=OPENENV_URL,
            generation="def add(a, b): return a + b",
            timeout=10,
            fn_name="add",
        )

        sent_code = mock_api.call_args[0][1]
        # stdin should be embedded as a string literal, NOT read from sys.stdin
        assert "sys.stdin.read()" not in sent_code
        assert "_raw_input_str" in sent_code

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_output_comparison_strips_trailing_newlines(self, mock_api):
        mock_api.return_value = (
            {"observation": {"stdout": "42\n\n", "stderr": "", "exit_code": 0}},
            None,
        )

        result, meta = _process_single_case(
            case_index=0,
            stdin_data="",
            expected_output="42\n",
            openenv_url=OPENENV_URL,
            generation="print(42)",
            timeout=10,
        )

        assert result is True
        assert meta["status"] == "success"

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_metadata_has_expected_keys(self, mock_api):
        mock_api.return_value = (
            {"observation": {"stdout": "ok\n", "stderr": "", "exit_code": 0}},
            None,
        )

        _, meta = _process_single_case(
            case_index=3,
            stdin_data="input_val",
            expected_output="ok",
            openenv_url=OPENENV_URL,
            generation="print('ok')",
            timeout=10,
        )

        expected_keys = {
            "case_index",
            "input",
            "expected_output",
            "stdout",
            "stderr",
            "exit_code",
            "status",
            "api_request_error",
        }
        assert expected_keys.issubset(meta.keys())
        assert meta["case_index"] == 3
        assert meta["expected_output"] == "ok"
        assert meta["exit_code"] == 0


class TestCheckCorrectness:
    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_all_cases_pass(self, mock_api):
        mock_api.return_value = (
            {"observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}},
            None,
        )
        in_outs = {"inputs": ["1", "2"], "outputs": ["42", "42"]}

        results, metadata = check_correctness(
            OPENENV_URL, in_outs, "print(42)", timeout=10
        )

        assert results == [True, True]
        assert len(metadata) == 2

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_mixed_results_preserved_in_order(self, mock_api):
        def side_effect(url, code, timeout_s=30.0):
            # All calls get same code, but we control output to get mixed results
            return (
                {"observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}},
                None,
            )

        mock_api.side_effect = side_effect
        in_outs = {"inputs": ["1", "2"], "outputs": ["42", "99"]}

        results, metadata = check_correctness(
            OPENENV_URL, in_outs, "print(42)", timeout=10
        )

        assert results[0] is True
        assert results[1] is False

    def test_none_in_outs_returns_error(self):
        results, metadata = check_correctness(OPENENV_URL, None, "print(42)")

        assert results == [-1]
        assert len(metadata) == 1

    def test_missing_keys_returns_error(self):
        results, metadata = check_correctness(
            OPENENV_URL, {"inputs": ["1"]}, "print(42)"
        )

        assert results == [-1]

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_empty_inputs(self, mock_api):
        in_outs = {"inputs": [], "outputs": []}

        results, metadata = check_correctness(OPENENV_URL, in_outs, "print(42)")

        assert results == []
        assert metadata == []
        mock_api.assert_not_called()

    @patch(f"{MOCK_MODULE}.call_openenv_api")
    def test_fn_name_passed_through(self, mock_api):
        mock_api.return_value = (
            {"observation": {"stdout": "42\n", "stderr": "", "exit_code": 0}},
            None,
        )
        in_outs = {"inputs": ["5"], "outputs": ["42"], "fn_name": "solve"}

        check_correctness(OPENENV_URL, in_outs, "def solve(x): return 42", timeout=10)

        sent_code = mock_api.call_args[0][1]
        assert "solve(*_args)" in sent_code
        assert "def solve(x): return 42" in sent_code
