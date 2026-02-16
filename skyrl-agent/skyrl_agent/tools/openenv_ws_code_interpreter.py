from __future__ import annotations

from skyrl_agent.tools.base import register_tool
from skyrl_agent.tools.openenv_ws_base import OpenEnvWSCodeInterpreterBase


@register_tool("openenv_ws_code_interpreter")
class OpenEnvWSCodeInterpreter(OpenEnvWSCodeInterpreterBase):
    name = "openenv_ws_code_interpreter"
    description = (
        "Executes Python code in a sandboxed environment. "
        "IMPORTANT: Each invocation runs in a fresh namespace — variables, "
        "functions, and imports from previous calls are NOT available. You "
        "must re-import modules and redefine any functions in every call. "
        "The last expression in the code is automatically printed (like a "
        "Python REPL), so you do not need to wrap it in print()."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute. Each call starts with an empty "
                    "namespace, so always include all necessary imports and "
                    "definitions."
                ),
            },
        },
        "required": ["code"],
    }
