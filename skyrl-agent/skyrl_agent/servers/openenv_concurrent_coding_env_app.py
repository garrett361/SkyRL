"""Backward-compatibility shim.

The actual implementation has moved to skyrl_agent.servers.coding_env.app.
This module re-exports app and main so existing uvicorn launch commands
and YAML configs that reference this module path continue to work.
"""

from skyrl_agent.servers.coding_env.app import app, main  # noqa: F401
