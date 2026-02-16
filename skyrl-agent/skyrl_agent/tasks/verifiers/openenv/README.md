# OpenEnv Verifier

Analogue of the `sandbox_fusion` verifier, but targeting [OpenEnv](https://github.com/meta-pytorch/OpenEnv) coding RL environments instead of ByteDance's SandboxFusion.

## How it works

`utils.py` provides three functions:

| Function | Purpose |
|----------|---------|
| `call_openenv_api` | `POST /step` to an OpenEnv server with retry logic (3 attempts, linear backoff, retries on `ConnectionError`, `Timeout`, 502/503/504). |
| `_process_single_case` | Run one test case: send code (or a `_build_fn_wrapper` around it when `fn_name` is given), compare stdout against expected output, classify as `success` / `wrong_answer` / `runtime_error` / `timeout` / `api_error`. |
| `check_correctness` | Fan out all test cases via `ThreadPoolExecutor` and collect `(results, metadata)`. |

The API surface mirrors `sandbox_fusion/utils.py` (`check_correctness` signature, status codes `-1`/`-2`/`-3`/`True`/`False`) so callers can swap backends with minimal changes.

## OpenEnv server variants

The verifier is server-agnostic — it only speaks HTTP to `POST /step`. Two server variants are covered in the test:

| Package | Entrypoint | Executor | State between steps | Use case |
|---------|-----------|----------|---------------------|----------|
| `coding_env` (upstream OpenEnv) | `skyrl_agent.servers.coding_env.app:app` | `PyExecutor` (smolagents AST interpreter) | Persists | Interactive / WebSocket sessions |
| `memlimited_coding_env` (SkyRL) | `skyrl_agent.servers.memlimited_coding_env.app:app` | `SubprocessPyExecutor` (fork + RLIMIT, import guards, audit hooks) | Does not persist | TORL training (stateless HTTP, memory-safe) |

`MemlimitedPythonCodeActEnv` subclasses `PythonCodeActEnv` but replaces the executor with `SubprocessPyExecutor`, which spawns a fresh child process per step with `RLIMIT_AS` and `RLIMIT_CPU` limits.

For TORL training, use `memlimited_coding_env` — it prevents agent-generated code from OOM-ing the server.

## Tests

```bash
cd SkyRL/skyrl-agent

# Unit tests (mocked, fast):
python -m pytest tests/test_openenv_verifier.py -v

# Integration tests against in-process coding_env:
python -m pytest tests/test_openenv_verifier_integration.py -v

# Integration tests against memlimited_coding_env (subprocess-isolated):
python -m pytest tests/test_openenv_verifier_memlimited_integration.py -v
```
