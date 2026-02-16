"""Tests for MemlimitedPersistentPythonCodeActEnv.

Covers: basic execution, state persistence across steps, memory limits,
timeout, crash recovery, transform/reward pipeline, and reset behaviour.
"""

from __future__ import annotations

import os

import pytest

from coding_env.models import CodeAction
from skyrl_agent.servers.memlimited_persistent_coding_env import (
    MemlimitedPersistentPythonCodeActEnv,
)


@pytest.fixture
def env():
    old_mem = os.environ.get("OPENENV_MEMORY_LIMIT_MB")
    old_timeout = os.environ.get("OPENENV_TIMEOUT_S")
    os.environ["OPENENV_MEMORY_LIMIT_MB"] = "512"
    os.environ["OPENENV_TIMEOUT_S"] = "10"
    try:
        e = MemlimitedPersistentPythonCodeActEnv()
        e.reset()
        yield e
    finally:
        try:
            e._executor.close()
        except Exception:
            pass
        if old_mem is None:
            os.environ.pop("OPENENV_MEMORY_LIMIT_MB", None)
        else:
            os.environ["OPENENV_MEMORY_LIMIT_MB"] = old_mem
        if old_timeout is None:
            os.environ.pop("OPENENV_TIMEOUT_S", None)
        else:
            os.environ["OPENENV_TIMEOUT_S"] = old_timeout


class TestBasicExecution:
    def test_hello_world(self, env: MemlimitedPersistentPythonCodeActEnv):
        obs = env.step(CodeAction(code="print('hello world')"))
        assert obs.exit_code == 0
        assert "hello" in obs.stdout.lower()

    def test_arithmetic(self, env: MemlimitedPersistentPythonCodeActEnv):
        obs = env.step(CodeAction(code="x = 2 + 3\nprint(x)"))
        assert obs.exit_code == 0
        assert "5" in obs.stdout

    def test_syntax_error(self, env: MemlimitedPersistentPythonCodeActEnv):
        obs = env.step(CodeAction(code="def f(\n"))
        assert obs.exit_code != 0
        assert obs.stderr

    def test_runtime_error(self, env: MemlimitedPersistentPythonCodeActEnv):
        obs = env.step(CodeAction(code="1 / 0"))
        assert obs.exit_code != 0
        assert obs.stderr


class TestStatePersistence:
    def test_variables_persist_across_steps(
        self, env: MemlimitedPersistentPythonCodeActEnv
    ):
        obs1 = env.step(CodeAction(code="my_var = 42"))
        assert obs1.exit_code == 0

        obs2 = env.step(CodeAction(code="print(my_var)"))
        assert obs2.exit_code == 0
        assert "42" in obs2.stdout

    def test_functions_persist_across_steps(
        self, env: MemlimitedPersistentPythonCodeActEnv
    ):
        env.step(CodeAction(code="def greet(name):\n    return f'hello {name}'"))
        obs = env.step(CodeAction(code="print(greet('world'))"))
        assert obs.exit_code == 0
        assert "hello world" in obs.stdout

    def test_imports_persist_across_steps(
        self, env: MemlimitedPersistentPythonCodeActEnv
    ):
        obs1 = env.step(CodeAction(code="import math"))
        assert obs1.exit_code == 0

        obs2 = env.step(CodeAction(code="print(math.pi)"))
        assert obs2.exit_code == 0
        assert "3.14" in obs2.stdout

    def test_incremental_computation(
        self, env: MemlimitedPersistentPythonCodeActEnv
    ):
        env.step(CodeAction(code="total = 0"))
        env.step(CodeAction(code="total += 10"))
        env.step(CodeAction(code="total += 20"))
        obs = env.step(CodeAction(code="print(total)"))
        assert obs.exit_code == 0
        assert "30" in obs.stdout

    def test_state_survives_runtime_error(
        self, env: MemlimitedPersistentPythonCodeActEnv
    ):
        env.step(CodeAction(code="keep_me = 'safe'"))
        obs_err = env.step(CodeAction(code="1 / 0"))
        assert obs_err.exit_code != 0

        obs = env.step(CodeAction(code="print(keep_me)"))
        assert obs.exit_code == 0
        assert "safe" in obs.stdout


class TestMemoryLimit:
    def test_large_allocation_is_stopped(self):
        os.environ["OPENENV_MEMORY_LIMIT_MB"] = "512"
        os.environ["OPENENV_TIMEOUT_S"] = "10"
        try:
            e = MemlimitedPersistentPythonCodeActEnv()
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
            try:
                e._executor.close()
            except Exception:
                pass
            os.environ.pop("OPENENV_MEMORY_LIMIT_MB", None)
            os.environ.pop("OPENENV_TIMEOUT_S", None)


class TestTimeout:
    def test_infinite_loop_is_killed(self):
        os.environ["OPENENV_TIMEOUT_S"] = "2"
        try:
            e = MemlimitedPersistentPythonCodeActEnv()
            e.reset()
            obs = e.step(CodeAction(code="while True: pass"))
            assert obs.exit_code != 0
            assert "TIMEOUT" in obs.stderr or "limit" in obs.stderr.lower()
        finally:
            try:
                e._executor.close()
            except Exception:
                pass
            os.environ.pop("OPENENV_TIMEOUT_S", None)


class TestRecoveryAfterCrash:
    def test_recovery_after_oom(self):
        os.environ["OPENENV_MEMORY_LIMIT_MB"] = "512"
        os.environ["OPENENV_TIMEOUT_S"] = "10"
        try:
            e = MemlimitedPersistentPythonCodeActEnv()
            e.reset()

            obs_oom = e.step(CodeAction(code="x = [0] * (500_000_000)"))
            assert obs_oom.exit_code != 0

            obs_ok = e.step(CodeAction(code="print('recovered')"))
            assert obs_ok.exit_code == 0
            assert "recovered" in obs_ok.stdout
        finally:
            try:
                e._executor.close()
            except Exception:
                pass
            os.environ.pop("OPENENV_MEMORY_LIMIT_MB", None)
            os.environ.pop("OPENENV_TIMEOUT_S", None)

    def test_recovery_after_timeout(self):
        os.environ["OPENENV_TIMEOUT_S"] = "2"
        try:
            e = MemlimitedPersistentPythonCodeActEnv()
            e.reset()

            obs_timeout = e.step(CodeAction(code="while True: pass"))
            assert obs_timeout.exit_code != 0

            obs_ok = e.step(CodeAction(code="print('back')"))
            assert obs_ok.exit_code == 0
            assert "back" in obs_ok.stdout
        finally:
            try:
                e._executor.close()
            except Exception:
                pass
            os.environ.pop("OPENENV_TIMEOUT_S", None)

    def test_state_lost_after_crash(self):
        os.environ["OPENENV_MEMORY_LIMIT_MB"] = "512"
        os.environ["OPENENV_TIMEOUT_S"] = "10"
        try:
            e = MemlimitedPersistentPythonCodeActEnv()
            e.reset()

            e.step(CodeAction(code="pre_crash_var = 'exists'"))
            e.step(CodeAction(code="x = [0] * (500_000_000)"))

            obs = e.step(CodeAction(code="print(pre_crash_var)"))
            assert obs.exit_code != 0, (
                "State should be lost after worker crash"
            )
        finally:
            try:
                e._executor.close()
            except Exception:
                pass
            os.environ.pop("OPENENV_MEMORY_LIMIT_MB", None)
            os.environ.pop("OPENENV_TIMEOUT_S", None)


class TestTransformRewards:
    def test_safe_code_gets_reward(self, env: MemlimitedPersistentPythonCodeActEnv):
        obs = env.step(CodeAction(code="x = 5"))
        assert obs.reward == pytest.approx(0.1, rel=1e-9)

    def test_dangerous_code_penalized(self, env: MemlimitedPersistentPythonCodeActEnv):
        obs = env.step(CodeAction(code="import os"))
        assert obs.reward == pytest.approx(-0.9, rel=1e-9)
        assert "safety_violation" in obs.metadata

    def test_metadata_contains_last_code(
        self, env: MemlimitedPersistentPythonCodeActEnv
    ):
        code = "print('test')"
        obs = env.step(CodeAction(code=code))
        assert obs.metadata["last_code"] == code


class TestReset:
    def test_reset_resets_step_count(self, env: MemlimitedPersistentPythonCodeActEnv):
        for _ in range(3):
            env.step(CodeAction(code="x = 1"))
        assert env.state.step_count == 3

        env.reset()
        assert env.state.step_count == 0

    def test_reset_changes_episode_id(self, env: MemlimitedPersistentPythonCodeActEnv):
        ep1 = env.state.episode_id
        env.step(CodeAction(code="x = 1"))
        env.reset()
        assert env.state.episode_id != ep1

    def test_reset_clears_persisted_state(
        self, env: MemlimitedPersistentPythonCodeActEnv
    ):
        env.step(CodeAction(code="persisted_var = 123"))
        obs_before = env.step(CodeAction(code="print(persisted_var)"))
        assert obs_before.exit_code == 0

        env.reset()
        obs_after = env.step(CodeAction(code="print(persisted_var)"))
        assert obs_after.exit_code != 0, (
            "State should be cleared after reset"
        )

    def test_reward_works_after_reset(self, env: MemlimitedPersistentPythonCodeActEnv):
        env.step(CodeAction(code="x = 1"))
        env.reset()
        obs = env.step(CodeAction(code="y = 2"))
        assert obs.reward == pytest.approx(0.1, rel=1e-9)
