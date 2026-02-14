"""Subprocess-isolated Python executor with memory limits and import guards.

Runs each code execution in a separate child process with resource limits
(RLIMIT_AS for virtual memory, RLIMIT_CPU for CPU time).  The child process
is killed on timeout or OOM, preventing unbounded memory usage from affecting
the server.

Security layers (inside the child process):
    1. RLIMIT_AS / RLIMIT_CPU — OS-level resource caps.
    2. Restricted ``__import__`` — only allow-listed modules can be imported.
    3. Stripped builtins — ``open``, ``eval``, ``exec``, ``compile`` removed.
    4. ``sys.addaudithook`` — backstop blocking subprocess, os.system, sockets.

Configure via environment variables:
    OPENENV_MEMORY_LIMIT_MB  Max *additional* virtual memory the child may
                             allocate beyond its inherited footprint.
                             Default: 4096 (4 GB).
    OPENENV_TIMEOUT_S        Wall-clock timeout per execution.  Default: 30.
    OPENENV_CPU_LIMIT_S      CPU-time limit (SIGXCPU on exceed).  Default: 60.
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


@dataclass
class CodeExecResult:
    """Result of code execution containing stdout, stderr, and exit code."""

    stdout: str
    stderr: str
    exit_code: int


_DEFAULT_ALLOWED_MODULES: frozenset[str] = frozenset({
    # stdlib — safe, no OS/network side effects
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
    # third-party — data science
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


def _make_restricted_import(
    allowed: frozenset[str],
) -> callable:
    """Return an ``__import__`` replacement that only allows *allowed* modules."""
    original = _builtins_mod.__import__

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            # Relative imports within an already-allowed package — permit.
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


def _child_worker(
    code: str,
    memory_limit_bytes: int,
    cpu_limit_s: int,
    allowed_modules: frozenset[str],
    conn: mp.connection.Connection,
) -> None:
    """Execute *code* via ``exec()`` inside a resource-limited child process.

    Results are sent back through *conn* as a ``(stdout, stderr, exit_code)``
    tuple.  On ``MemoryError`` or other failures the error is reported via
    the same tuple so the parent never blocks on the pipe.
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

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exit_code = 0

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(
            stderr_buf
        ):
            exec(
                compile(code, "<agent>", "exec"),
                {"__builtins__": safe_builtins},
            )
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
    finally:
        conn.close()


class SubprocessPyExecutor:
    """Execute Python code in an isolated subprocess with memory/time limits.

    Each call to :meth:`run` spawns a child process (via ``fork``) that:

    1. Sets ``RLIMIT_AS`` to ``current_vm + memory_limit`` to cap virtual
       memory.
    2. Sets ``RLIMIT_CPU`` as a CPU-time backstop.
    3. Installs an audit hook and restricted builtins (import allow-list).
    4. Runs the code with ``exec()`` (real CPython, no AST interpreter).
    5. Is ``terminate``/``kill``-ed on timeout.

    The interface (``run(code) -> CodeExecResult``) is identical to
    ``coding_env``'s ``PyExecutor``, so this is a drop-in replacement.
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
        logger.info(
            "SubprocessPyExecutor: memory_limit=%dMB timeout=%ds cpu_limit=%ds "
            "allowed_modules=%d",
            self.memory_limit_mb,
            self.timeout_s,
            self.cpu_limit_s,
            len(self._allowed_modules),
        )

    def run(self, code: str) -> CodeExecResult:
        """Execute *code* in an isolated subprocess and return the result."""
        parent_conn, child_conn = mp.Pipe(duplex=False)

        proc = mp.Process(
            target=_child_worker,
            args=(
                code,
                self.memory_limit_mb * 1024 * 1024,
                self.cpu_limit_s,
                self._allowed_modules,
                child_conn,
            ),
            daemon=True,
        )
        proc.start()
        child_conn.close()

        try:
            if parent_conn.poll(timeout=self.timeout_s):
                stdout, stderr, exit_code = parent_conn.recv()
                proc.join(timeout=5)
                return CodeExecResult(
                    stdout=stdout, stderr=stderr, exit_code=exit_code
                )

            self._kill_process(proc)
            return CodeExecResult(
                stdout="",
                stderr=f"TIMEOUT: execution exceeded {self.timeout_s}s limit",
                exit_code=1,
            )

        except (EOFError, OSError):
            self._kill_process(proc)
            exit_code = proc.exitcode
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
        finally:
            parent_conn.close()
            if proc.is_alive():
                self._kill_process(proc)

    @staticmethod
    def _kill_process(proc: mp.Process) -> None:
        """Terminate then kill *proc*, ensuring cleanup."""
        try:
            proc.terminate()
            proc.join(timeout=5)
        except Exception:
            logger.warning(
                "Failed to terminate child pid=%s", proc.pid, exc_info=True
            )
        if proc.is_alive():
            try:
                proc.kill()
                proc.join(timeout=5)
            except Exception:
                logger.warning(
                    "Failed to kill child pid=%s", proc.pid, exc_info=True
                )
