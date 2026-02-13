from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Union

from skyrl_agent.tools.base import BaseTool, register_tool
from skyrl_agent.tools.sandbox_fusion import CodeInterpreter

if TYPE_CHECKING:
    from coding_env.client import CodingEnv

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60.0


@register_tool("openenv_ws_code_interpreter")
class OpenEnvWSCodeInterpreter(BaseTool):
    name = "openenv_ws_code_interpreter"
    description = "Executes Python code via a persistent OpenEnv WebSocket session."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the OpenEnv sandbox.",
            },
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self._session: Optional[CodingEnv] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_ws_session(
        self, session: CodingEnv, event_loop: asyncio.AbstractEventLoop
    ) -> None:
        self._session = session
        self._event_loop = event_loop

    def close_ws_session(self) -> None:
        self._session = None
        self._event_loop = None

    def _execute_code(self, code: str) -> str:
        if self._session is None or self._event_loop is None:
            raise RuntimeError(
                "WebSocket session not initialized. "
                "Call set_ws_session() before executing code."
            )

        from coding_env.models import CodeAction

        action = CodeAction(code=code)
        future = asyncio.run_coroutine_threadsafe(
            self._session.step(action), self._event_loop
        )
        result = future.result(timeout=DEFAULT_TIMEOUT_S)
        return result.observation.stdout + result.observation.stderr

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, dict]:
        try:
            params = self._verify_json_format_args(params)
        except (ValueError, Exception) as exc:
            return {"error": f"Invalid parameters: {exc}"}

        code = params.get("code", "")
        code = CodeInterpreter._post_process_code(code)

        if not code:
            return {"error": "Code parameter is required."}

        try:
            result = self._execute_code(code)
        except RuntimeError as exc:
            return {"error": str(exc)}

        return result
