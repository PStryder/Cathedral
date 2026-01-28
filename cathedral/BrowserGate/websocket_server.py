"""
WebSocket server for browser extension communication.

Receives page content, selections, and commands from the Cathedral browser extension.
"""

import json
import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from .models import ExtensionMessage, ExtensionResponse, BrowserConfig

logger = logging.getLogger(__name__)


class ExtensionHandler:
    """
    Handles messages from the browser extension.

    Override or inject handlers for custom behavior.
    """

    def __init__(self):
        self.handlers: Dict[str, Callable] = {
            "send_to_cathedral": self._handle_send_to_cathedral,
            "send_to_scripture": self._handle_send_to_scripture,
            "search_memory": self._handle_search_memory,
            "ping": self._handle_ping,
        }

        # Callbacks injected by Cathedral
        self.on_chat_message: Optional[Callable] = None
        self.on_store_scripture: Optional[Callable] = None
        self.on_search_memory: Optional[Callable] = None

    async def handle(self, message: ExtensionMessage) -> ExtensionResponse:
        """Route message to appropriate handler."""
        action = message.action or "send_to_cathedral"

        handler = self.handlers.get(action)
        if not handler:
            return ExtensionResponse(
                success=False,
                message=f"Unknown action: {action}"
            )

        try:
            return await handler(message)
        except Exception as e:
            logger.exception(f"Handler error for {action}")
            return ExtensionResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

    async def _handle_send_to_cathedral(self, msg: ExtensionMessage) -> ExtensionResponse:
        """Send content to Cathedral chat."""
        content = msg.selection or msg.content or ""
        if not content:
            return ExtensionResponse(success=False, message="No content to send")

        # Build context message
        context = f"[From browser: {msg.title or msg.url or 'unknown'}]\n\n{content}"

        if self.on_chat_message:
            try:
                result = await self.on_chat_message(context, msg.url)
                return ExtensionResponse(
                    success=True,
                    message="Sent to Cathedral",
                    data={"queued": True, "context_length": len(context)}
                )
            except Exception as e:
                return ExtensionResponse(success=False, message=str(e))

        # No handler registered - just acknowledge receipt
        return ExtensionResponse(
            success=True,
            message="Received (no chat handler)",
            data={"content_length": len(content)}
        )

    async def _handle_send_to_scripture(self, msg: ExtensionMessage) -> ExtensionResponse:
        """Store content as scripture."""
        content = msg.content or msg.selection or ""
        if not content:
            return ExtensionResponse(success=False, message="No content to store")

        if self.on_store_scripture:
            try:
                result = await self.on_store_scripture(
                    content=content,
                    title=msg.title,
                    url=msg.url
                )
                return ExtensionResponse(
                    success=True,
                    message="Stored as scripture",
                    data=result
                )
            except Exception as e:
                return ExtensionResponse(success=False, message=str(e))

        return ExtensionResponse(
            success=False,
            message="Scripture storage not configured"
        )

    async def _handle_search_memory(self, msg: ExtensionMessage) -> ExtensionResponse:
        """Search memory for related content."""
        query = msg.selection or msg.content or ""
        if not query:
            return ExtensionResponse(success=False, message="No query provided")

        if self.on_search_memory:
            try:
                results = await self.on_search_memory(query[:500])
                return ExtensionResponse(
                    success=True,
                    message=f"Found {len(results)} results",
                    data={"results": results}
                )
            except Exception as e:
                return ExtensionResponse(success=False, message=str(e))

        return ExtensionResponse(
            success=False,
            message="Memory search not configured"
        )

    async def _handle_ping(self, msg: ExtensionMessage) -> ExtensionResponse:
        """Health check."""
        return ExtensionResponse(
            success=True,
            message="pong",
            data={"timestamp": datetime.utcnow().isoformat()}
        )


class WebSocketServer:
    """
    WebSocket server for browser extension.

    Usage:
        server = WebSocketServer(port=8765)
        await server.start()
        # ... later
        await server.stop()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        handler: Optional[ExtensionHandler] = None
    ):
        self.host = host
        self.port = port
        self.handler = handler or ExtensionHandler()
        self._server = None
        self._clients: set = set()

    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets not installed. Install with: pip install websockets"
            )

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server stopped")

    async def _handle_connection(self, websocket, path):
        """Handle a new WebSocket connection."""
        self._clients.add(websocket)
        client_info = f"{websocket.remote_address}"
        logger.info(f"Extension connected: {client_info}")

        try:
            async for raw_message in websocket:
                response = await self._process_message(raw_message)
                await websocket.send(json.dumps(response.model_dump()))
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            self._clients.discard(websocket)
            logger.info(f"Extension disconnected: {client_info}")

    async def _process_message(self, raw: str) -> ExtensionResponse:
        """Parse and process incoming message."""
        try:
            data = json.loads(raw)
            message = ExtensionMessage(**data)
            return await self.handler.handle(message)
        except json.JSONDecodeError:
            return ExtensionResponse(success=False, message="Invalid JSON")
        except Exception as e:
            logger.exception("Message processing error")
            return ExtensionResponse(success=False, message=str(e))

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        if not self._clients:
            return

        payload = json.dumps(message)
        await asyncio.gather(
            *[client.send(payload) for client in self._clients],
            return_exceptions=True
        )

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server is not None and self._server.is_serving()


# Global server instance
_server: Optional[WebSocketServer] = None


def get_server(config: Optional[BrowserConfig] = None) -> WebSocketServer:
    """Get or create global WebSocket server."""
    global _server
    if _server is None:
        cfg = config or BrowserConfig()
        _server = WebSocketServer(port=cfg.websocket_port)
    return _server


async def start_server(config: Optional[BrowserConfig] = None) -> WebSocketServer:
    """Start the WebSocket server."""
    server = get_server(config)
    if not server.is_running:
        await server.start()
    return server


async def stop_server():
    """Stop the WebSocket server."""
    global _server
    if _server:
        await _server.stop()
        _server = None
