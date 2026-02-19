"""Tests for MemlimitedPythonCodeActEnv.

Covers: basic execution, subprocess isolation (no state leaks), memory limits,
transform/reward pipeline, and reset behaviour.
"""

from __future__ import annotations

import os

import pytest

from coding_env.models import CodeAction
from skyrl_agent.servers.memlimited_coding_env import MemlimitedPythonCodeActEnv


@pytest.fixture
def env():
    old_mem = os.environ.get("OPENENV_MEMORY_LIMIT_MB")
    old_timeout = os.environ.get("OPENENV_TIMEOUT_S")
    os.environ["OPENENV_MEMORY_LIMIT_MB"] = "512"
    os.environ["OPENENV_TIMEOUT_S"] = "10"
    try:
        e = MemlimitedPythonCodeActEnv()
        e.reset()
        yield e
    finally:
        if old_mem is None:
            os.environ.pop("OPENENV_MEMORY_LIMIT_MB", None)
        else:
            os.environ["OPENENV_MEMORY_LIMIT_MB"] = old_mem
        if old_timeout is None:
            os.environ.pop("OPENENV_TIMEOUT_S", None)
        else:
            os.environ["OPENENV_TIMEOUT_S"] = old_timeout


class TestBasicExecution:
    def test_hello_world(self, env: MemlimitedPythonCodeActEnv):
        obs = env.step(CodeAction(code="print('hello world')"))
        assert obs.exit_code == 0
        assert "hello" in obs.stdout.lower()

    def test_arithmetic(self, env: MemlimitedPythonCodeActEnv):
        obs = env.step(CodeAction(code="x = 2 + 3\nprint(x)"))
        assert obs.exit_code == 0
        assert "5" in obs.stdout

    def test_syntax_error(self, env: MemlimitedPythonCodeActEnv):
        obs = env.step(CodeAction(code="def f(\n"))
        assert obs.exit_code != 0
        assert obs.stderr

    def test_runtime_error(self, env: MemlimitedPythonCodeActEnv):
        obs = env.step(CodeAction(code="1 / 0"))
        assert obs.exit_code != 0
        assert obs.stderr


class TestIsolation:
    def test_no_state_leaks_between_steps(self, env: MemlimitedPythonCodeActEnv):
        """Variables from one step must not be visible in the next."""
        obs1 = env.step(CodeAction(code="leaked_var = 42"))
        assert obs1.exit_code == 0

        obs2 = env.step(CodeAction(code="print(leaked_var)"))
        assert obs2.exit_code != 0, "Variable from prior step should not exist"

    def test_function_not_persisted(self, env: MemlimitedPythonCodeActEnv):
        env.step(CodeAction(code="def greet():\n    return 'hi'\n"))
        obs = env.step(CodeAction(code="print(greet())"))
        assert obs.exit_code != 0

    def test_import_not_persisted(self, env: MemlimitedPythonCodeActEnv):
        env.step(CodeAction(code="import math as m"))
        obs = env.step(CodeAction(code="print(m.pi)"))
        assert obs.exit_code != 0


class TestMemoryLimit:
    def test_large_allocation_is_stopped(self):
        os.environ["OPENENV_MEMORY_LIMIT_MB"] = "512"
        os.environ["OPENENV_TIMEOUT_S"] = "10"
        try:
            e = MemlimitedPythonCodeActEnv()
            e.reset()
            obs = e.step(CodeAction(code="x = [0] * (500_000_000)"))
            assert obs.exit_code != 0, (
                "Allocation should have been killed by RLIMIT_AS"
            )
            stderr_lower = obs.stderr.lower()
            assert (
                "memoryerror" in stderr_lower
                or "memory" in stderr_lower
                or "killed" in stderr_lower
                or "signal" in stderr_lower
            )
        finally:
            os.environ.pop("OPENENV_MEMORY_LIMIT_MB", None)
            os.environ.pop("OPENENV_TIMEOUT_S", None)


class TestTimeout:
    def test_infinite_loop_is_killed(self):
        os.environ["OPENENV_TIMEOUT_S"] = "2"
        try:
            e = MemlimitedPythonCodeActEnv()
            e.reset()
            obs = e.step(CodeAction(code="while True: pass"))
            assert obs.exit_code != 0
            assert "TIMEOUT" in obs.stderr or "limit" in obs.stderr.lower()
        finally:
            os.environ.pop("OPENENV_TIMEOUT_S", None)


class TestTransformRewards:
    def test_safe_code_gets_reward(self, env: MemlimitedPythonCodeActEnv):
        obs = env.step(CodeAction(code="x = 5"))
        assert obs.reward == pytest.approx(0.1, rel=1e-9)

    def test_dangerous_code_penalized(self, env: MemlimitedPythonCodeActEnv):
        obs = env.step(CodeAction(code="import os"))
        assert obs.reward == pytest.approx(-0.9, rel=1e-9)
        assert "safety_violation" in obs.metadata

    def test_metadata_contains_last_code(self, env: MemlimitedPythonCodeActEnv):
        code = "print('test')"
        obs = env.step(CodeAction(code=code))
        assert obs.metadata["last_code"] == code


class TestReset:
    def test_reset_resets_step_count(self, env: MemlimitedPythonCodeActEnv):
        for _ in range(3):
            env.step(CodeAction(code="x = 1"))
        assert env.state.step_count == 3

        env.reset()
        assert env.state.step_count == 0

    def test_reset_changes_episode_id(self, env: MemlimitedPythonCodeActEnv):
        ep1 = env.state.episode_id
        env.step(CodeAction(code="x = 1"))
        env.reset()
        assert env.state.episode_id != ep1

    def test_reward_works_after_reset(self, env: MemlimitedPythonCodeActEnv):
        env.step(CodeAction(code="x = 1"))
        env.reset()
        obs = env.step(CodeAction(code="y = 2"))
        assert obs.reward == pytest.approx(0.1, rel=1e-9)
