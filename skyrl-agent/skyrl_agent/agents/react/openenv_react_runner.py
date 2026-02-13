from __future__ import annotations

import asyncio
import logging
import os

import pandas as pd

from skyrl_agent.agents.react.react_runner import ReActTrajectory
from skyrl_agent.tools.ws_session import WSSessionTool

logger = logging.getLogger(__name__)

WS_CONNECT_TIMEOUT_S = 30.0
WS_MESSAGE_TIMEOUT_S = 120.0


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

            openenv_url = os.environ.get("OPENENV_URL", "http://localhost:8000")
            env = CodingEnv(
                base_url=openenv_url,
                connect_timeout_s=WS_CONNECT_TIMEOUT_S,
                message_timeout_s=WS_MESSAGE_TIMEOUT_S,
            )
            await env.connect()
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
