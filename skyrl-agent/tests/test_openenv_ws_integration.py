"""Integration tests for the OpenEnv WebSocket tool and session management.

Spins up an in-process coding_env server (uvicorn in a background thread),
exercises the WS tool and CodingEnv client, and tears down after.

Run with:
    cd SkyRL/skyrl-agent
    python -m pytest tests/test_openenv_ws_integration.py -v
"""

import asyncio
import socket
import threading
import time

import pytest
import requests
import uvicorn

from coding_env.client import CodingEnv
from coding_env.models import CodeAction, CodeObservation
from coding_env.server.python_codeact_env import PythonCodeActEnv
from openenv.core.env_server import create_fastapi_app

from skyrl_agent.tools.openenv_ws_code_interpreter import OpenEnvWSCodeInterpreter

PythonCodeActEnv.SUPPORTS_CONCURRENT_SESSIONS = True

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(env_cls, max_concurrent_envs=8):
    """Start a uvicorn server for *env_cls* and return (base_url, server, thread)."""
    port = _free_port()
    app = create_fastapi_app(
        env_cls, CodeAction, CodeObservation, max_concurrent_envs=max_concurrent_envs
    )
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
                return base_url, server, thread
        except requests.ConnectionError:
            pass
        time.sleep(0.1)

    raise RuntimeError(f"OpenEnv server failed to start on port {port}")


@pytest.fixture(scope="module")
def coding_env_server():
    """Start a coding_env server with concurrent session support."""
    base_url, server, thread = _start_server(PythonCodeActEnv)
    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


class TestWSToolExecutesCode:
    def test_ws_tool_executes_code(self, coding_env_server):
        """Tool executes code through a real coding_env WS session."""
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        try:
            env = CodingEnv(base_url=coding_env_server)
            asyncio.run_coroutine_threadsafe(env.connect(), loop).result(timeout=10)
            asyncio.run_coroutine_threadsafe(env.reset(), loop).result(timeout=10)

            tool = OpenEnvWSCodeInterpreter()
            tool.set_ws_session(env, loop)

            result = tool.call({"code": "print(2 + 2)"})
            assert "4" in result

            tool.close_ws_session()
            asyncio.run_coroutine_threadsafe(env.close(), loop).result(timeout=10)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)


class TestWSSessionIsolation:
    def test_ws_sessions_have_independent_state(self, coding_env_server):
        """Variables set in one session are invisible to another.

        PythonCodeActEnv uses an in-process PyExecutor that persists state
        across steps within a session, so x=42 in session A is visible to
        later steps in A but not in B.
        """

        async def _run():
            env_a = CodingEnv(base_url=coding_env_server)
            env_b = CodingEnv(base_url=coding_env_server)

            await env_a.connect()
            await env_b.connect()
            try:
                await env_a.reset()
                await env_b.reset()

                result_a = await env_a.step(CodeAction(code="x = 42"))
                assert result_a.observation.exit_code == 0

                result_a2 = await env_a.step(CodeAction(code="print(x)"))
                assert result_a2.observation.exit_code == 0
                assert "42" in result_a2.observation.stdout

                result_b = await env_b.step(CodeAction(code="print(x)"))
                assert result_b.observation.exit_code == 1
                assert "not defined" in result_b.observation.stderr
            finally:
                await env_a.close()
                await env_b.close()

        asyncio.run(_run())


class TestWSSessionCleanup:
    def test_ws_session_cleanup(self, coding_env_server):
        """close() tears down WS; reconnecting yields a fresh session.

        EnvClient._ensure_connected() auto-reconnects if _ws is None, so
        rather than asserting that step raises, we verify the old session's
        state is gone after close + reconnect (proving a new server-side
        session was created).
        """

        async def _run():
            env = CodingEnv(base_url=coding_env_server)
            await env.connect()
            await env.reset()

            await env.step(CodeAction(code="cleanup_var = 'exists'"))
            result = await env.step(CodeAction(code="print(cleanup_var)"))
            assert "exists" in result.observation.stdout

            await env.close()
            assert env._ws is None

            await env.connect()
            await env.reset()
            result = await env.step(CodeAction(code="print(cleanup_var)"))
            assert result.observation.exit_code == 1
            assert "not defined" in result.observation.stderr
            await env.close()

        asyncio.run(_run())


class TestMultipleConcurrentSessions:
    @staticmethod
    async def _assert_concurrent(server_url):
        async def _session_lifecycle(base_url: str, value: int) -> str:
            env = CodingEnv(base_url=base_url)
            await env.connect()
            try:
                await env.reset()
                result = await env.step(CodeAction(code=f"print({value})"))
                assert result.observation.exit_code == 0
                assert str(value) in result.observation.stdout
                return result.observation.stdout.strip()
            finally:
                await env.close()

        results = await asyncio.gather(
            _session_lifecycle(server_url, 10),
            _session_lifecycle(server_url, 20),
            _session_lifecycle(server_url, 30),
            _session_lifecycle(server_url, 40),
        )
        assert set(results) == {"10", "20", "30", "40"}

    def test_multiple_concurrent_sessions(self, coding_env_server):
        """Four concurrent sessions all succeed."""
        asyncio.run(self._assert_concurrent(coding_env_server))
