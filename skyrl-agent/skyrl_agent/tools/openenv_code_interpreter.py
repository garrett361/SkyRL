import logging
import os
import time
from typing import Union

import requests

from skyrl_agent.tools.base import BaseTool, register_tool
from skyrl_agent.tools.sandbox_fusion import CodeInterpreter

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = 5
RETRY_DELAY_S = 2.0
RETRYABLE_STATUS_CODES = {502, 503, 504}
DEFAULT_TIMEOUT_S = 30.0


@register_tool("openenv_code_interpreter")
class OpenEnvCodeInterpreter(BaseTool):
    name = "openenv_code_interpreter"
    description = "Executes Python code via the OpenEnv sandbox API."
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

    _post_process_code = staticmethod(CodeInterpreter._post_process_code)

    def _get_openenv_url(self) -> str:
        url = os.getenv("OPENENV_URL")
        if not url:
            raise ValueError(
                "OPENENV_URL environment variable must be set to use OpenEnvCodeInterpreter"
            )
        return url.rstrip("/")

    def _execute_code(self, code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> str | None:
        url = self._get_openenv_url()
        endpoint = f"{url}/step"
        body = {"action": {"code": code}, "timeout_s": timeout_s}

        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = requests.post(endpoint, json=body)
            except (requests.ConnectionError, requests.Timeout) as exc:
                logger.warning(
                    "Attempt %d/%d failed: %s", attempt + 1, RETRY_ATTEMPTS, exc
                )
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY_S)
                continue

            if resp.status_code == 200:
                data = resp.json()
                obs = data["observation"]
                return obs["stdout"] + obs["stderr"]

            if resp.status_code in RETRYABLE_STATUS_CODES:
                logger.warning(
                    "Attempt %d/%d: HTTP %d — %s",
                    attempt + 1,
                    RETRY_ATTEMPTS,
                    resp.status_code,
                    resp.text,
                )
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY_S)
                continue

            # 4xx and other non-retryable errors
            return "no stdout here"

        return None

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, dict]:
        try:
            params = self._verify_json_format_args(params)
        except (ValueError, Exception) as exc:
            return {"error": f"Invalid parameters: {exc}"}

        code = params.get("code", "")
        code = self._post_process_code(code)

        if not code:
            return {"error": "Code parameter is required."}

        try:
            result = self._execute_code(code)
        except ValueError as exc:
            return {"error": str(exc)}

        if result is None:
            return {"error": "Max retries exceeded"}
        return result
