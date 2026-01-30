"""
MCPClient - Single Server Connection Handler.

Manages the connection lifecycle and tool execution for a single MCP server.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from cathedral.shared.gate import GateLogger
from cathedral.MCPClient.models import (
    ConnectionStatus,
    MCPConnection,
    MCPServerConfig,
    MCPTool,
    MCPToolCallResult,
)
from cathedral.MCPClient.transport import StdioTransport

_log = GateLogger.get("MCPClient.Client")

# MCP Protocol version
MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPClient:
    """
    Client for a single MCP server connection.

    Handles:
    - Connection lifecycle (connect, disconnect)
    - Tool discovery via tools/list
    - Tool execution via tools/call
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize client for a server.

        Args:
            config: Server configuration
        """
        self.config = config
        self._transport: Optional[StdioTransport] = None
        self._connection = MCPConnection(server_id=config.id)
        self._tools: Dict[str, MCPTool] = {}

    @property
    def server_id(self) -> str:
        """Get server ID."""
        return self.config.id

    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._connection.status

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connection.status == ConnectionStatus.CONNECTED

    @property
    def tools(self) -> List[MCPTool]:
        """Get list of available tools."""
        return list(self._tools.values())

    def get_connection_info(self) -> MCPConnection:
        """Get connection information."""
        self._connection.tools = list(self._tools.values())
        return self._connection

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Starts subprocess, sends initialize request, and fetches tools.

        Returns:
            True if connected successfully
        """
        if self._connection.status == ConnectionStatus.CONNECTED:
            _log.info(f"Already connected to {self.config.name}")
            return True

        self._connection.status = ConnectionStatus.CONNECTING
        self._connection.error_message = None

        try:
            # Create and start transport
            self._transport = StdioTransport(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )

            if not await self._transport.start():
                raise RuntimeError("Failed to start transport")

            # Send initialize request
            init_response = await self._transport.send_request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {},
                    },
                    "clientInfo": {
                        "name": "Cathedral",
                        "version": "1.0.0",
                    },
                },
            )

            if init_response.is_error():
                error_msg = init_response.error.get("message", "Unknown error")
                raise RuntimeError(f"Initialize failed: {error_msg}")

            _log.info(f"Initialized connection to {self.config.name}")

            # Send initialized notification
            await self._transport.send_request("notifications/initialized", {})

            # Fetch available tools
            await self._refresh_tools()

            self._connection.status = ConnectionStatus.CONNECTED
            self._connection.connected_at = datetime.now()
            self._connection.pid = self._transport.pid

            _log.info(
                f"Connected to {self.config.name} with {len(self._tools)} tools"
            )
            return True

        except Exception as e:
            error_msg = str(e)
            _log.error(f"Failed to connect to {self.config.name}: {error_msg}")

            self._connection.status = ConnectionStatus.ERROR
            self._connection.error_message = error_msg

            # Cleanup transport
            if self._transport:
                await self._transport.stop()
                self._transport = None

            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._transport:
            await self._transport.stop()
            self._transport = None

        self._connection.status = ConnectionStatus.DISCONNECTED
        self._connection.connected_at = None
        self._connection.pid = None
        self._tools.clear()

        _log.info(f"Disconnected from {self.config.name}")

    async def _refresh_tools(self) -> None:
        """Fetch tools from the server."""
        if not self._transport:
            return

        response = await self._transport.send_request("tools/list", {})

        if response.is_error():
            _log.warning(
                f"Failed to list tools from {self.config.name}: "
                f"{response.error.get('message', 'Unknown error')}"
            )
            return

        self._tools.clear()

        tools_data = response.result.get("tools", []) if response.result else []
        for tool_data in tools_data:
            try:
                tool = MCPTool.from_dict(tool_data)
                self._tools[tool.name] = tool
            except Exception as e:
                _log.warning(f"Failed to parse tool: {e}")

        _log.debug(f"Loaded {len(self._tools)} tools from {self.config.name}")

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> MCPToolCallResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            MCPToolCallResult with success/error status
        """
        if not self.is_connected or not self._transport:
            return MCPToolCallResult(
                success=False,
                error="Not connected to MCP server",
            )

        if tool_name not in self._tools:
            return MCPToolCallResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            response = await self._transport.send_request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments or {},
                },
            )

            if response.is_error():
                error_msg = response.error.get("message", "Unknown error")
                return MCPToolCallResult(
                    success=False,
                    error=error_msg,
                    is_error=True,
                )

            # Parse result content
            result = response.result or {}
            content = result.get("content", [])
            is_error = result.get("isError", False)

            # Extract text content from content array
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)

            return MCPToolCallResult(
                success=not is_error,
                content=content if len(content) > 1 else (text_parts[0] if text_parts else content),
                is_error=is_error,
                error=text_parts[0] if is_error and text_parts else None,
            )

        except Exception as e:
            _log.error(f"Tool call {tool_name} failed: {e}")
            return MCPToolCallResult(
                success=False,
                error=str(e),
            )

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self._tools


__all__ = ["MCPClient"]
