"""Integration tests for the OpenEnv verifier.

Spins up an in-process OpenEnv coding_env HTTP server (uvicorn in a background
thread), runs the verifier functions against it, and tears down after.

    python -m pytest tests/test_openenv_verifier_integration.py -v
"""

import socket
import threading
import time

import pytest
import requests
import uvicorn

from coding_env.models import CodeAction, CodeObservation
from coding_env.server.python_codeact_env import PythonCodeActEnv
from coding_env.server.python_executor import PyExecutor
from coding_env.server.transforms import create_safe_coding_transform
from openenv.core.env_server import create_fastapi_app

from skyrl_agent.tasks.verifiers.openenv.utils import (
    call_openenv_api,
    check_correctness,
    _process_single_case,
)

pytestmark = pytest.mark.integration

TEST_TIMEOUT = 15

# Imports the fn_name wrapper needs beyond smolagents' defaults.
_WRAPPER_IMPORTS = ["json", "sys", "traceback"]


class _VerifierCompatibleEnv(PythonCodeActEnv):
    """PythonCodeActEnv with extra authorized imports for the fn_name wrapper.

    Only needed for tests that exercise the fn_name code path.  The
    ``_build_fn_wrapper`` helper emits ``import json / sys / traceback``,
    which are outside smolagents' default allow-list.  In production TORL
    the wrapper is never used (fn_name=None, agent code is sent directly),
    so the standard PythonCodeActEnv works unmodified.
    """

    def __init__(self):
        self.transform = create_safe_coding_transform()
        self._executor = PyExecutor(additional_imports=_WRAPPER_IMPORTS)
        from coding_env.models import CodeState

        self._state = CodeState()

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._executor = PyExecutor(additional_imports=_WRAPPER_IMPORTS)
        return obs


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def openenv_server():
    """Start an OpenEnv coding_env HTTP server on a random port for the module."""
    port = _free_port()
    app = create_fastapi_app(_VerifierCompatibleEnv, CodeAction, CodeObservation)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError(f"OpenEnv server failed to start on port {port}")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


class TestCallOpenenvApiIntegration:
    def test_simple_print(self, openenv_server):
        resp, err = call_openenv_api(openenv_server, "print('hello')", timeout_s=TEST_TIMEOUT)

        assert err is None, f"Unexpected error: {err}"
        obs = resp["observation"]
        assert obs["stdout"].strip() == "hello"
        assert obs["exit_code"] == 0

    def test_arithmetic(self, openenv_server):
        resp, err = call_openenv_api(openenv_server, "print(2 + 3)", timeout_s=TEST_TIMEOUT)

        assert err is None, f"Unexpected error: {err}"
        assert resp["observation"]["stdout"].strip() == "5"

    def test_syntax_error_returns_nonzero_exit(self, openenv_server):
        resp, err = call_openenv_api(openenv_server, "def f(\n", timeout_s=TEST_TIMEOUT)

        assert err is None, f"Unexpected error: {err}"
        obs = resp["observation"]
        assert obs["exit_code"] != 0
        assert "SyntaxError" in obs["stderr"]

    def test_runtime_error_returns_nonzero_exit(self, openenv_server):
        resp, err = call_openenv_api(openenv_server, "1 / 0", timeout_s=TEST_TIMEOUT)

        assert err is None, f"Unexpected error: {err}"
        obs = resp["observation"]
        assert obs["exit_code"] != 0
        assert "ZeroDivisionError" in obs["stderr"]

    def test_multiline_output(self, openenv_server):
        code = "for i in range(3): print(i)"
        resp, err = call_openenv_api(openenv_server, code, timeout_s=TEST_TIMEOUT)

        assert err is None, f"Unexpected error: {err}"
        assert resp["observation"]["stdout"].strip() == "0\n1\n2"

    def test_import_and_compute(self, openenv_server):
        code = "import math; print(math.factorial(5))"
        resp, err = call_openenv_api(openenv_server, code, timeout_s=TEST_TIMEOUT)

        assert err is None, f"Unexpected error: {err}"
        assert resp["observation"]["stdout"].strip() == "120"


class TestProcessSingleCaseIntegration:
    def test_correct_answer(self, openenv_server):
        result, meta = _process_single_case(
            case_index=0,
            stdin_data="",
            expected_output="42",
            openenv_url=openenv_server,
            generation="print(42)",
            timeout=TEST_TIMEOUT,
        )

        assert result is True
        assert meta["status"] == "success"
        assert meta["exit_code"] == 0

    def test_wrong_answer(self, openenv_server):
        result, meta = _process_single_case(
            case_index=0,
            stdin_data="",
            expected_output="99",
            openenv_url=openenv_server,
            generation="print(42)",
            timeout=TEST_TIMEOUT,
        )

        assert result is False
        assert meta["status"] == "wrong_answer"

    def test_runtime_error(self, openenv_server):
        result, meta = _process_single_case(
            case_index=0,
            stdin_data="",
            expected_output="42",
            openenv_url=openenv_server,
            generation="1 / 0",
            timeout=TEST_TIMEOUT,
        )

        assert result == -2
        assert meta["status"] == "runtime_error"
        assert "ZeroDivisionError" in meta["stderr"]

    def test_fn_name_with_global_function(self, openenv_server):
        generation = "def add(a, b): return a + b"
        result, meta = _process_single_case(
            case_index=0,
            stdin_data="3\n4",
            expected_output="7",
            openenv_url=openenv_server,
            generation=generation,
            timeout=TEST_TIMEOUT,
            fn_name="add",
        )

        assert result is True, f"Expected True, got {result}. meta={meta}"
        assert meta["status"] == "success"

    def test_fn_name_with_scalar_output(self, openenv_server):
        generation = "def double(x): return x * 2"
        result, meta = _process_single_case(
            case_index=0,
            stdin_data="5",
            expected_output="10",
            openenv_url=openenv_server,
            generation=generation,
            timeout=TEST_TIMEOUT,
            fn_name="double",
        )

        assert result is True, f"Expected True, got {result}. meta={meta}"


class TestCheckCorrectnessIntegration:
    def test_all_pass(self, openenv_server):
        in_outs = {
            "inputs": ["", ""],
            "outputs": ["42", "42"],
        }

        results, metadata = check_correctness(
            openenv_server, in_outs, "print(42)", timeout=TEST_TIMEOUT
        )

        assert results == [True, True]
        assert len(metadata) == 2
        assert all(m["status"] == "success" for m in metadata)

    def test_mixed_pass_and_fail(self, openenv_server):
        in_outs = {
            "inputs": ["", ""],
            "outputs": ["42", "99"],
        }

        results, metadata = check_correctness(
            openenv_server, in_outs, "print(42)", timeout=TEST_TIMEOUT
        )

        assert results[0] is True
        assert results[1] is False

    def test_fn_name_multiple_cases(self, openenv_server):
        in_outs = {
            "inputs": ["1\n2", "10\n20", "0\n0"],
            "outputs": ["3", "30", "0"],
            "fn_name": "add",
        }

        results, metadata = check_correctness(
            openenv_server,
            in_outs,
            "def add(a, b): return a + b",
            timeout=TEST_TIMEOUT,
        )

        assert results == [True, True, True], f"results={results}, metadata={metadata}"

    def test_runtime_error_across_cases(self, openenv_server):
        in_outs = {
            "inputs": ["", ""],
            "outputs": ["42", "42"],
        }

        results, metadata = check_correctness(
            openenv_server, in_outs, "1 / 0", timeout=TEST_TIMEOUT
        )

        assert all(r == -2 for r in results)

    def test_single_case(self, openenv_server):
        in_outs = {
            "inputs": [""],
            "outputs": ["hello world"],
        }

        results, metadata = check_correctness(
            openenv_server, in_outs, "print('hello world')", timeout=TEST_TIMEOUT
        )

        assert results == [True]
        assert metadata[0]["status"] == "success"
