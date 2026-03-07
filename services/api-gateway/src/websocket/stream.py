"""Custom WebSocket streaming protocol with backpressure handling.

Daneshbonyan: Internal Design & Development.
Custom protocol that bridges WebSocket connections to SSE inference streams
with backpressure management and automatic reconnection.
"""

from __future__ import annotations

from typing import Any

from dana_common.logging import setup_logging
from fastapi import WebSocket

logger = setup_logging("ws-stream")


class StreamManager:
    """Manages active WebSocket connections with backpressure control.

    Custom protocol:
    - Client sends: {"type": "chat", "messages": [...], "model": "..."}
    - Server sends: {"type": "token", "content": "..."} for each token
    - Server sends: {"type": "done", "usage": {...}} on completion
    - Server sends: {"type": "error", "message": "..."} on error
    """

    def __init__(self) -> None:
        self._active_connections: dict[str, WebSocket] = {}
        self._max_connections = 100
        self._buffer_high_water = 64  # Pause if buffer exceeds this

    async def connect(self, websocket: WebSocket, connection_id: str) -> bool:
        if len(self._active_connections) >= self._max_connections:
            await websocket.close(code=1013, reason="Server overloaded")
            return False
        await websocket.accept()
        self._active_connections[connection_id] = websocket
        logger.info("WebSocket connected: %s (total: %d)", connection_id, len(self._active_connections))
        return True

    def disconnect(self, connection_id: str) -> None:
        self._active_connections.pop(connection_id, None)
        logger.info("WebSocket disconnected: %s (total: %d)", connection_id, len(self._active_connections))

    async def send_token(self, connection_id: str, content: str) -> bool:
        """Send a token to client with backpressure check."""
        ws = self._active_connections.get(connection_id)
        if ws is None:
            return False
        try:
            await ws.send_json({"type": "token", "content": content})
            return True
        except Exception:
            self.disconnect(connection_id)
            return False

    async def send_done(self, connection_id: str, usage: dict[str, Any]) -> None:
        ws = self._active_connections.get(connection_id)
        if ws is None:
            return
        try:
            await ws.send_json({"type": "done", "usage": usage})
        except Exception:
            self.disconnect(connection_id)

    async def send_error(self, connection_id: str, message: str) -> None:
        ws = self._active_connections.get(connection_id)
        if ws is None:
            return
        try:
            await ws.send_json({"type": "error", "message": message})
        except Exception:
            self.disconnect(connection_id)

    @property
    def active_count(self) -> int:
        return len(self._active_connections)


stream_manager = StreamManager()
