from __future__ import annotations

from skyrl_agent.tools.base import register_tool
from skyrl_agent.tools.openenv_ws_base import OpenEnvWSCodeInterpreterBase


@register_tool("openenv_ws_persistent_code_interpreter")
class OpenEnvWSPersistentCodeInterpreter(OpenEnvWSCodeInterpreterBase):
    name = "openenv_ws_persistent_code_interpreter"
    description = (
        "Executes Python code in a sandboxed environment with persistent state. "
        "Variables, imports, and function definitions are preserved across calls "
        "within the same session, so you can build on prior results without "
        "re-importing or redefining. State is only lost if execution hits the "
        "memory limit or times out. "
        "The last expression in the code is automatically printed (like a "
        "Python REPL), so you do not need to wrap it in print()."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute. The namespace persists across calls, "
                    "so previously defined variables, functions, and imports are "
                    "available."
                ),
            },
        },
        "required": ["code"],
    }
