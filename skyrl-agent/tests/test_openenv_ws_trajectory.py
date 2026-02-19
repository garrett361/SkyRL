"""Tests for OpenEnvReActTrajectory, OpenEnvReActAgent alias, and registry entries."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skyrl_agent.agents.mapping import AGENT_GENERATOR_REGISTRY, AGENT_TRAJECTORY_REGISTRY
from skyrl_agent.agents.react import OpenEnvReActAgent, OpenEnvReActTrajectory
from skyrl_agent.agents.react.react_agent import ReActAgent
from skyrl_agent.agents.react.react_runner import ReActTrajectory


class TestRegistryEntries:
    def test_generator_registry_has_openenv_agent(self):
        key = "skyrl_agent.agents.react.OpenEnvReActAgent"
        assert key in AGENT_GENERATOR_REGISTRY
        assert AGENT_GENERATOR_REGISTRY[key] == "skyrl_agent.agents.base.AgentRunner"

    def test_trajectory_registry_has_openenv_agent(self):
        key = "skyrl_agent.agents.react.OpenEnvReActAgent"
        assert key in AGENT_TRAJECTORY_REGISTRY
        expected = "skyrl_agent.agents.react.openenv_react_runner.OpenEnvReActTrajectory"
        assert AGENT_TRAJECTORY_REGISTRY[key] == expected

    def test_openenv_react_agent_is_react_agent_subclass(self):
        assert issubclass(OpenEnvReActAgent, ReActAgent)

    def test_trajectory_inherits_react_trajectory(self):
        assert issubclass(OpenEnvReActTrajectory, ReActTrajectory)


def _make_mock_traj_config(tools=None):
    cfg = MagicMock()
    cfg.instance_id = "test-instance"
    cfg.trajectory_id = 0
    cfg.tools = tools or ["em_finish", "openenv_ws_code_interpreter"]
    cfg.agent_cls = "skyrl_agent.agents.react.OpenEnvReActAgent"
    cfg.max_prompt_length = 4096
    cfg.sampling_params = {"temperature": 1.0, "max_tokens": 4096}
    cfg.qwen3_enable_thinking = False
    cfg.qwen3_acc_thinking = False
    cfg.max_iterations = 5
    cfg.early_step_threshold = 0
    cfg.debug_log = False
    cfg.profile_tools = False
    cfg.vision_is_active = False
    cfg.enable_turn_reminder = False
    return cfg


def _make_trajectory(cfg=None, data=None, ws_tools_in_agent=True):
    """Build an OpenEnvReActTrajectory with mocked dependencies."""
    if cfg is None:
        cfg = _make_mock_traj_config()
    if data is None:
        data = {
            "instance_id": "test-instance",
            "instance": {"problem": "test problem"},
            "data_source": "test",
        }

    mock_infer_engine = MagicMock()
    mock_tokenizer = MagicMock()
    mock_task = MagicMock()
    mock_task.get_instruction.return_value = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Solve this."},
    ]

    traj = OpenEnvReActTrajectory(
        cfg=cfg,
        data=data,
        infer_engine=mock_infer_engine,
        tokenizer=mock_tokenizer,
        task=mock_task,
    )
    return traj


class TestWSLifecycleMocked:
    @pytest.mark.asyncio
    async def test_ws_session_lifecycle(self):
        """WS session is created, injected, and cleaned up on success."""
        mock_env = AsyncMock()
        mock_env.connect = AsyncMock()
        mock_env.reset = AsyncMock()
        mock_env.close = AsyncMock()

        mock_ws_tool = MagicMock()
        mock_ws_tool.set_ws_session = MagicMock()
        mock_ws_tool.close_ws_session = MagicMock()

        mock_agent = MagicMock()
        mock_agent.tools = {"openenv_ws_code_interpreter": mock_ws_tool}
        mock_agent.run = AsyncMock(return_value=("FINISH_TOOL", {"answer": "42"}))
        mock_agent.get_messages.return_value = []
        mock_agent.get_transitions.return_value = []
        mock_agent.get_tool_profile.return_value = None

        traj = _make_trajectory()

        with (
            patch.object(traj, "agent_cls", return_value=mock_agent),
            patch(
                "coding_env.client.CodingEnv",
            ) as MockCodingEnvCls,
        ):
            MockCodingEnvCls.return_value = mock_env
            await traj.generate_trajectory()

        mock_env.connect.assert_awaited_once()
        mock_env.reset.assert_awaited_once()
        mock_ws_tool.set_ws_session.assert_called_once()
        session_arg, loop_arg = mock_ws_tool.set_ws_session.call_args[0]
        assert session_arg is mock_env
        assert isinstance(loop_arg, asyncio.AbstractEventLoop)
        mock_agent.run.assert_awaited_once()
        mock_ws_tool.close_ws_session.assert_called_once()
        mock_env.close.assert_awaited_once()
        assert traj.result["finish_reason"] == "FINISH_TOOL"

    @pytest.mark.asyncio
    async def test_ws_cleanup_on_error(self):
        """WS session is cleaned up even when agent.run() raises."""
        mock_env = AsyncMock()
        mock_env.connect = AsyncMock()
        mock_env.reset = AsyncMock()
        mock_env.close = AsyncMock()

        mock_ws_tool = MagicMock()
        mock_ws_tool.set_ws_session = MagicMock()
        mock_ws_tool.close_ws_session = MagicMock()

        mock_agent = MagicMock()
        mock_agent.tools = {"openenv_ws_code_interpreter": mock_ws_tool}
        mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM backend failed"))

        traj = _make_trajectory()

        with (
            patch.object(traj, "agent_cls", return_value=mock_agent),
            patch(
                "coding_env.client.CodingEnv",
            ) as MockCodingEnvCls,
        ):
            MockCodingEnvCls.return_value = mock_env
            with pytest.raises(RuntimeError, match="LLM backend failed"):
                await traj.generate_trajectory()

        mock_ws_tool.close_ws_session.assert_called_once()
        mock_env.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_ws_tool_skips_session(self):
        """When no tools have set_session, no WS session is created."""
        mock_finish_tool = MagicMock(spec=["call", "name", "get_tool_param", "get_system_prompt_prefix"])

        mock_agent = MagicMock()
        mock_agent.tools = {"em_finish": mock_finish_tool}
        mock_agent.run = AsyncMock(return_value=("FINISH_TOOL", {"answer": "42"}))
        mock_agent.get_messages.return_value = []
        mock_agent.get_transitions.return_value = []
        mock_agent.get_tool_profile.return_value = None

        cfg = _make_mock_traj_config(tools=["em_finish"])
        traj = _make_trajectory(cfg=cfg)

        with patch.object(traj, "agent_cls", return_value=mock_agent):
            await traj.generate_trajectory()

        assert traj.result["finish_reason"] == "FINISH_TOOL"
