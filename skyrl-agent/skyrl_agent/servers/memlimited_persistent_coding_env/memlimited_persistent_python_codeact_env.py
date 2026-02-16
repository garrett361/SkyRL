"""Memory-limited persistent Python Code Action Environment.

Subclasses PythonCodeActEnv and replaces the in-process PyExecutor with
PersistentSubprocessPyExecutor, which keeps a long-lived child process with
RLIMIT_AS and RLIMIT_CPU limits.

Key differences from MemlimitedPythonCodeActEnv (the stateless variant):
    - State PERSISTS between steps (variables, imports, functions survive).
    - Memory and CPU limits are still enforced; OOM/timeout kills the worker
      and the next step starts with a fresh namespace.

Key differences from PythonCodeActEnv (the in-process variant):
    - Execution happens in a subprocess, so OOM or infinite loops cannot crash
      the server.
    - Import allow-list and stripped builtins provide sandboxing.

Designed for WebSocket sessions (e.g. TORL) where agents build on prior
computations across multiple tool calls.

Configure limits via environment variables:
    OPENENV_MEMORY_LIMIT_MB        (default 4096)
    OPENENV_TIMEOUT_S              (default 30)
    OPENENV_CPU_LIMIT_S            (default 60)
    OPENENV_CPU_LIMIT_MULTIPLIER   (default 20)
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import Observation

from coding_env.server.python_codeact_env import PythonCodeActEnv
from .persistent_subprocess_executor import PersistentSubprocessPyExecutor


class MemlimitedPersistentPythonCodeActEnv(PythonCodeActEnv):
    """PythonCodeActEnv with a persistent, memory-limited subprocess.

    Each ``step()`` sends code to a long-lived child process.  Variables,
    imports, and function definitions carry over between steps.  The child
    is killed and respawned (losing state) only on OOM, timeout, or reset.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, additional_imports: list[str] | None = None):
        super().__init__()
        self._additional_imports = additional_imports or []
        # super().__init__() creates a PyExecutor on self._executor; replace it.
        self._executor = PersistentSubprocessPyExecutor(
            additional_imports=self._additional_imports
        )

    def reset(self) -> Observation:
        # Save and close the persistent executor before super().reset()
        # replaces self._executor with a plain PyExecutor.
        old_executor = self._executor
        if isinstance(old_executor, PersistentSubprocessPyExecutor):
            old_executor.close()

        obs = super().reset()
        # super().reset() assigned a new PyExecutor; replace it again.
        self._executor = PersistentSubprocessPyExecutor(
            additional_imports=self._additional_imports
        )
        return obs
