"""OpenEnv memlimited_coding_env server with concurrent WebSocket session support.

Uses MemlimitedPythonCodeActEnv which runs each step in a subprocess with
RLIMIT_AS and RLIMIT_CPU limits, preventing OOM or infinite loops from
affecting the server process.

Run with:
    uvicorn skyrl_agent.servers.memlimited_coding_env.app:app --host 0.0.0.0 --port 8000
"""

import os

from openenv.core.env_server import create_app
from coding_env.models import CodeAction, CodeObservation

from .memlimited_python_codeact_env import MemlimitedPythonCodeActEnv

app = create_app(
    MemlimitedPythonCodeActEnv,
    CodeAction,
    CodeObservation,
    env_name="memlimited_coding_env",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", 256)),
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
