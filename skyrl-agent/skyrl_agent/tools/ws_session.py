from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openenv.core import EnvClient


@runtime_checkable
class WSSessionTool(Protocol):
    """Protocol for tools that require a per-trajectory WebSocket session.

    Tools implementing this protocol will have their WS session lifecycle
    managed by the trajectory class: connect before agent.run(), inject
    via set_ws_session(), and close in a finally block.

    The session is an openenv EnvClient instance (e.g. CodingEnv) that
    communicates with the environment server over a persistent WebSocket.
    """

    def set_ws_session(self, session: EnvClient, event_loop: asyncio.AbstractEventLoop) -> None:
        """Inject the WS session and event loop for dispatching async calls."""
        ...

    def close_ws_session(self) -> None:
        """Release the WS session reference (connection closed by trajectory)."""
        ...
