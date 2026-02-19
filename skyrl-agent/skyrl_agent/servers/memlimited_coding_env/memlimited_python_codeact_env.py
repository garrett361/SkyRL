"""Memory-limited Python Code Action Environment.

Subclasses PythonCodeActEnv and replaces the in-process PyExecutor with
SubprocessPyExecutor, which spawns a fresh child process per step() with
RLIMIT_AS and RLIMIT_CPU limits.

Key differences from PythonCodeActEnv:
    - State does NOT persist between steps (each step is a fresh process).
    - Memory and CPU limits are enforced per execution.
    - A child OOM or infinite loop is killed without affecting the server.

Designed for stateless HTTP endpoints (e.g. TORL ``POST /step``) where
each request is independent.  For interactive WebSocket sessions that need
cross-step state persistence, use PythonCodeActEnv directly.

Configure limits via environment variables:
    OPENENV_MEMORY_LIMIT_MB  (default 4096)
    OPENENV_TIMEOUT_S        (default 30)
    OPENENV_CPU_LIMIT_S      (default 60)
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import Observation

from coding_env.server.python_codeact_env import PythonCodeActEnv
from .subprocess_executor import SubprocessPyExecutor


class MemlimitedPythonCodeActEnv(PythonCodeActEnv):
    """PythonCodeActEnv with subprocess-based memory/CPU limits.

    Drop-in replacement for PythonCodeActEnv on stateless HTTP paths.
    Each ``step()`` runs code in a subprocess with resource limits; variables
    and imports do not carry over between steps.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, additional_imports: list[str] | None = None):
        super().__init__()
        self._additional_imports = additional_imports or []
        self._executor = SubprocessPyExecutor(
            additional_imports=self._additional_imports
        )

    def reset(self) -> Observation:
        obs = super().reset()
        self._executor = SubprocessPyExecutor(
            additional_imports=self._additional_imports
        )
        return obs
