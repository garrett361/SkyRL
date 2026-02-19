from __future__ import annotations

import asyncio
import itertools
import logging
import os
from functools import lru_cache

import pandas as pd

from skyrl_agent.agents.react.react_runner import ReActTrajectory
from skyrl_agent.tools.ws_session import WSSessionTool

logger = logging.getLogger(__name__)

WS_CONNECT_TIMEOUT_S = 60.0
WS_MESSAGE_TIMEOUT_S = 120.0
WS_PING_TIMEOUT_S = 120.0
WS_CONNECT_MAX_RETRIES = 3
WS_CONNECT_RETRY_DELAY_S = 5.0

_url_counter = itertools.count()


@lru_cache(maxsize=1)
def _get_openenv_urls() -> list[str]:
    """Parse OPENENV_URLS (comma-separated) into a list of base URLs."""
    raw = os.environ.get("OPENENV_URLS", "http://localhost:8000")
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    logger.info("OpenEnv server URLs: %s", urls)
    return urls


class OpenEnvReActTrajectory(ReActTrajectory):
    """ReActTrajectory that manages a per-trajectory WebSocket session.

    Opens a CodingEnv WS connection before agent execution and ensures
    cleanup in a finally block, regardless of success or failure.
    """

    async def generate_trajectory(self) -> None:
        data = self.data
        instance_id = data["instance_id"] if data["instance_id"] else self.cfg.instance_id
        instance = pd.Series(data["instance"])
        self.agent = self.agent_cls(
            traj_config=self.cfg,
            infer_engine=self.infer_engine,
            tokenizer=self.tokenizer,
        )

        ws_tools = [t for t in self.agent.tools.values() if isinstance(t, WSSessionTool)]

        env = None
        if ws_tools:
            # CodingEnv is the typed WS client for both the standard coding_env
            # and the memlimited_coding_env servers -- they share the same
            # CodeAction/CodeObservation/CodeState types and /ws endpoint.
            from coding_env.client import CodingEnv

            urls = _get_openenv_urls()
            openenv_url = urls[next(_url_counter) % len(urls)]
            env = CodingEnv(
                base_url=openenv_url,
                connect_timeout_s=WS_CONNECT_TIMEOUT_S,
                message_timeout_s=WS_MESSAGE_TIMEOUT_S,
            )
            for attempt in range(1, WS_CONNECT_MAX_RETRIES + 1):
                try:
                    await env.connect()
                    break
                except (ConnectionError, TimeoutError) as e:
                    if attempt == WS_CONNECT_MAX_RETRIES:
                        raise
                    logger.warning(
                        "WS connect to %s failed (attempt %d/%d): %s. Retrying in %.0fs...",
                        openenv_url, attempt, WS_CONNECT_MAX_RETRIES, e, WS_CONNECT_RETRY_DELAY_S,
                    )
                    await asyncio.sleep(WS_CONNECT_RETRY_DELAY_S)
            # Extend the keepalive ping timeout on the underlying websocket
            # connection. The default (20s) is too short for persistent servers
            # where reset()/step() can take longer due to subprocess spawning.
            if hasattr(env, "_ws") and env._ws is not None:
                env._ws.ping_timeout = WS_PING_TIMEOUT_S
            await env.reset()
            loop = asyncio.get_running_loop()
            for tool in ws_tools:
                tool.set_ws_session(env, loop)

        try:
            instruction = self.task.get_instruction(instance)
            finish_reason, result = await self.agent.run(instruction, instance)
            tool_profile = None
            try:
                tool_profile = self.agent.get_tool_profile()
            except Exception:
                pass
            self.result = {
                "instance_id": instance_id,
                "trajectory_id": self.cfg.trajectory_id,
                "messages": self.agent.get_messages(),
                "transitions": self.agent.get_transitions(),
                "results": result,
                "finish_reason": finish_reason,
                "state": {"tool_profile": tool_profile} if tool_profile is not None else {},
            }
        finally:
            for tool in ws_tools:
                tool.close_ws_session()
            if env is not None:
                await env.close()
