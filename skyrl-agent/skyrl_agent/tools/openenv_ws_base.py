"""Base class for openenv WebSocket code interpreter tools.

Provides the shared WS session lifecycle and code execution logic.
Subclasses only need to set ``name``, ``description``, and ``parameters``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Union

from skyrl_agent.tools.base import BaseTool
from skyrl_agent.tools.code_postprocess import post_process_code

if TYPE_CHECKING:
    from coding_env.client import CodingEnv

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60.0


class OpenEnvWSCodeInterpreterBase(BaseTool):
    """Base for code interpreter tools that execute via a WebSocket session."""

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
        code = post_process_code(code)

        if not code:
            return {"error": "Code parameter is required."}

        try:
            result = self._execute_code(code)
        except RuntimeError as exc:
            return {"error": str(exc)}

        return result
