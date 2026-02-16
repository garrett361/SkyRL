"""Persistent subprocess Python executor with memory limits and import guards.

Like SubprocessPyExecutor, but keeps a **long-lived** child process so that
variables, imports, and function definitions persist across ``run()`` calls.
The child loops, executing code snippets against a shared ``exec_globals``
namespace.

Worker lifecycle::

    SubprocessPyExecutor (stateless):     run() -> fork -> exec -> exit
    PersistentSubprocessPyExecutor:       run() -> send code to living child -> recv result
                                          (child loops, keeping exec_globals between calls)

Failure recovery:
    - Timeout (wall-clock): parent kills worker; next call spawns fresh one (state lost).
    - OOM (MemoryError): child sends error then exits; next call respawns.
    - RLIMIT_CPU exceeded: child killed by SIGXCPU; next call respawns.
    - Syntax/runtime errors: error returned, namespace preserved (partial side-effects may exist).

Configure via environment variables:
    OPENENV_MEMORY_LIMIT_MB          Max additional virtual memory.  Default: 4096.
    OPENENV_TIMEOUT_S                Wall-clock timeout per execution.  Default: 30.
    OPENENV_CPU_LIMIT_S              Per-call CPU-time budget (used to compute session limit).
                                     Default: 60.
    OPENENV_CPU_LIMIT_MULTIPLIER     Session CPU limit = cpu_limit_s * multiplier.  Default: 20.
"""

from __future__ import annotations

import builtins as _builtins_mod
import contextlib
import io
import logging
import multiprocessing as mp
import os
import resource
import signal
import sys
import traceback
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_DEFAULT_MEMORY_LIMIT_MB = 4096
_DEFAULT_TIMEOUT_S = 30
_DEFAULT_CPU_LIMIT_S = 60
_DEFAULT_CPU_LIMIT_MULTIPLIER = 20

_SENTINEL_SHUTDOWN = None


@dataclass
class CodeExecResult:
    """Result of code execution containing stdout, stderr, and exit code."""

    stdout: str
    stderr: str
    exit_code: int


_DEFAULT_ALLOWED_MODULES: frozenset[str] = frozenset({
    "math", "cmath", "statistics", "random",
    "collections", "itertools", "functools", "operator",
    "string", "re", "json", "csv",
    "datetime", "time", "calendar",
    "decimal", "fractions",
    "copy", "pprint", "textwrap", "unicodedata",
    "hashlib", "hmac", "base64", "binascii",
    "struct", "array", "bisect", "heapq",
    "enum", "dataclasses", "typing", "numbers",
    "abc", "contextlib",
    "numpy", "pandas", "scipy", "sympy", "sklearn",
})

_BLOCKED_AUDIT_EVENTS: frozenset[str] = frozenset({
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.spawn",
    "socket.connect",
    "socket.bind",
    "socket.sendto",
    "webbrowser.open",
})


def _make_restricted_import(allowed: frozenset[str]) -> callable:
    """Return an ``__import__`` replacement that only allows *allowed* modules."""
    original = _builtins_mod.__import__

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            return original(name, globals, locals, fromlist, level)
        top_level = name.split(".")[0]
        if top_level not in allowed:
            raise ImportError(
                f"Import of '{name}' is not allowed in this sandbox. "
                f"Permitted top-level modules: {sorted(allowed)}"
            )
        return original(name, globals, locals, fromlist, level)

    return _restricted_import


def _make_safe_builtins(allowed_modules: frozenset[str]) -> dict:
    """Build a builtins dict with dangerous functions removed."""
    dangerous = {"exec", "eval", "compile", "open", "__import__", "breakpoint"}
    safe = {k: v for k, v in vars(_builtins_mod).items() if k not in dangerous}
    safe["__import__"] = _make_restricted_import(allowed_modules)
    return safe


def _install_audit_hook() -> None:
    """Install a one-way audit hook blocking dangerous syscall patterns."""

    def _hook(event: str, args):
        if event in _BLOCKED_AUDIT_EVENTS:
            raise RuntimeError(
                f"Blocked operation: {event}. "
                "OS commands, subprocesses, and network access are not "
                "permitted in this sandbox."
            )

    sys.addaudithook(_hook)


def _get_vm_size_bytes() -> int:
    """Read current virtual memory size from /proc/self/status (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _persistent_child_worker(
    memory_limit_bytes: int,
    cpu_limit_s: int,
    allowed_modules: frozenset[str],
    conn: mp.connection.Connection,
) -> None:
    """Long-lived child process that executes code snippets in a loop.

    Sets resource limits and security hooks once at startup, then loops:
    recv(code) -> exec(code, exec_globals) -> send((stdout, stderr, exit_code)).

    The ``exec_globals`` namespace persists across iterations, so variables,
    imports, and function definitions survive between calls.
    """
    try:
        current_vm = _get_vm_size_bytes()
        new_limit = current_vm + memory_limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, new_limit))
    except (ValueError, OSError):
        logger.debug("Could not set RLIMIT_AS", exc_info=True)
        raise

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_s, cpu_limit_s))
    except (ValueError, OSError):
        logger.debug("Could not set RLIMIT_CPU", exc_info=True)
        raise

    _install_audit_hook()
    safe_builtins = _make_safe_builtins(allowed_modules)
    exec_globals: dict = {"__builtins__": safe_builtins}

    while True:
        try:
            code = conn.recv()
        except (EOFError, OSError):
            break

        if code is _SENTINEL_SHUTDOWN:
            break

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code = 0

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(
                stderr_buf
            ):
                exec(compile(code, "<agent>", "exec"), exec_globals)
        except MemoryError:
            try:
                conn.send(("", "MemoryError: execution exceeded memory limit", 137))
            except Exception:
                logger.debug(
                    "Child could not send MemoryError through pipe", exc_info=True
                )
            return
        except Exception:
            stderr_buf.write(traceback.format_exc())
            exit_code = 1

        try:
            conn.send((stdout_buf.getvalue(), stderr_buf.getvalue(), exit_code))
        except Exception:
            logger.debug("Child could not send result through pipe", exc_info=True)
            return


class PersistentSubprocessPyExecutor:
    """Execute Python code in a persistent subprocess with memory/time limits.

    Unlike :class:`SubprocessPyExecutor` which spawns a fresh child per call,
    this executor keeps a long-lived child process whose namespace persists
    across calls.  Variables, imports, and function definitions survive between
    ``run()`` invocations within the same session.

    The child is lazily spawned on first ``run()`` and respawned automatically
    if it dies (OOM, timeout, CPU limit).  Call ``close()`` or ``reset()``
    to explicitly kill the worker and discard accumulated state.
    """

    def __init__(self, additional_imports: list[str] | None = None):
        self._allowed_modules = _DEFAULT_ALLOWED_MODULES | frozenset(
            additional_imports or []
        )
        self.memory_limit_mb = int(
            os.environ.get("OPENENV_MEMORY_LIMIT_MB", _DEFAULT_MEMORY_LIMIT_MB)
        )
        self.timeout_s = int(
            os.environ.get("OPENENV_TIMEOUT_S", _DEFAULT_TIMEOUT_S)
        )
        self.cpu_limit_s = int(
            os.environ.get("OPENENV_CPU_LIMIT_S", _DEFAULT_CPU_LIMIT_S)
        )
        self._cpu_multiplier = int(
            os.environ.get("OPENENV_CPU_LIMIT_MULTIPLIER", _DEFAULT_CPU_LIMIT_MULTIPLIER)
        )
        self._proc: mp.Process | None = None
        self._conn: mp.connection.Connection | None = None
        logger.info(
            "PersistentSubprocessPyExecutor: memory_limit=%dMB timeout=%ds "
            "cpu_limit=%ds (session=%ds) allowed_modules=%d",
            self.memory_limit_mb,
            self.timeout_s,
            self.cpu_limit_s,
            self.cpu_limit_s * self._cpu_multiplier,
            len(self._allowed_modules),
        )

    def _ensure_worker(self) -> None:
        """Lazily spawn the persistent child if it is not alive."""
        if self._proc is not None and self._proc.is_alive():
            return

        if self._proc is not None:
            self._cleanup_dead_worker()

        parent_conn, child_conn = mp.Pipe(duplex=True)
        session_cpu_limit = self.cpu_limit_s * self._cpu_multiplier

        proc = mp.Process(
            target=_persistent_child_worker,
            args=(
                self.memory_limit_mb * 1024 * 1024,
                session_cpu_limit,
                self._allowed_modules,
                child_conn,
            ),
            daemon=True,
        )
        proc.start()
        child_conn.close()

        self._proc = proc
        self._conn = parent_conn

    def _cleanup_dead_worker(self) -> None:
        """Clean up references to a dead worker process."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._proc is not None:
            try:
                self._proc.join(timeout=1)
            except Exception:
                pass
            self._proc = None

    def run(self, code: str) -> CodeExecResult:
        """Execute *code* in the persistent subprocess and return the result.

        If the worker is dead (from a prior OOM/timeout), it is automatically
        respawned with a fresh namespace.
        """
        self._ensure_worker()
        assert self._conn is not None
        assert self._proc is not None

        try:
            self._conn.send(code)
        except (OSError, BrokenPipeError):
            self._cleanup_dead_worker()
            return CodeExecResult(
                stdout="",
                stderr="Worker process died unexpectedly; state has been reset.",
                exit_code=1,
            )

        try:
            if self._conn.poll(timeout=self.timeout_s):
                stdout, stderr, exit_code = self._conn.recv()
                # If the child signalled a fatal error (e.g. MemoryError), it
                # will exit shortly.  Proactively clean up so the next call
                # spawns a fresh worker instead of sending to a dying process.
                if exit_code == 137:
                    self._kill_worker()
                return CodeExecResult(
                    stdout=stdout, stderr=stderr, exit_code=exit_code
                )

            self._kill_worker()
            return CodeExecResult(
                stdout="",
                stderr=f"TIMEOUT: execution exceeded {self.timeout_s}s limit",
                exit_code=1,
            )

        except (EOFError, OSError):
            exit_code = self._proc.exitcode if self._proc else None
            self._cleanup_dead_worker()
            if exit_code is not None and exit_code < 0:
                sig_num = -exit_code
                try:
                    sig_name = signal.Signals(sig_num).name
                except (ValueError, KeyError):
                    sig_name = str(sig_num)
                return CodeExecResult(
                    stdout="",
                    stderr=(
                        f"Process killed by {sig_name} "
                        "(likely exceeded memory or CPU limit)"
                    ),
                    exit_code=137,
                )
            return CodeExecResult(
                stdout="",
                stderr=f"Process crashed with exit code {exit_code}",
                exit_code=exit_code or 1,
            )

    def _kill_worker(self) -> None:
        """Terminate then kill the worker process."""
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.join(timeout=5)
        except Exception:
            logger.warning(
                "Failed to terminate child pid=%s", self._proc.pid, exc_info=True
            )
        if self._proc.is_alive():
            try:
                self._proc.kill()
                self._proc.join(timeout=5)
            except Exception:
                logger.warning(
                    "Failed to kill child pid=%s", self._proc.pid, exc_info=True
                )
        self._cleanup_dead_worker()

    def close(self) -> None:
        """Shut down the worker process and release resources."""
        if self._proc is not None and self._proc.is_alive():
            try:
                self._conn.send(_SENTINEL_SHUTDOWN)
                self._proc.join(timeout=5)
            except Exception:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._kill_worker()
        else:
            self._cleanup_dead_worker()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
