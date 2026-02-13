"""OpenEnv coding_env server with concurrent WebSocket session support.

Enables multiple simultaneous WS sessions by setting SUPPORTS_CONCURRENT_SESSIONS
on PythonCodeActEnv at import time, before create_app() validates the flag.

Run with:
    uvicorn skyrl_agent.servers.openenv_concurrent_coding_env_app:app --host 0.0.0.0 --port 8000
"""

import os

from openenv.core.env_server import create_app
from coding_env.models import CodeAction, CodeObservation
from coding_env.server.python_codeact_env import PythonCodeActEnv

# The default is False. Change at runtime for efficient usage.
PythonCodeActEnv.SUPPORTS_CONCURRENT_SESSIONS = True

app = create_app(
    PythonCodeActEnv,
    CodeAction,
    CodeObservation,
    env_name="coding_env",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", 256)),
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
