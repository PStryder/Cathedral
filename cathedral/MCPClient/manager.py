"""
MCPClient Manager.

Manages multiple MCP server connections and provides unified tool access.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cathedral.shared.gate import GateLogger
from cathedral.MCPClient.client import MCPClient
from cathedral.MCPClient.models import (
    ConnectionStatus,
    MCPConnection,
    MCPServerConfig,
    MCPTool,
    MCPToolCallResult,
)

_log = GateLogger.get("MCPClient.Manager")


class MCPManager:
    """
    Manager for multiple MCP server connections.

    Handles:
    - Multi-server lifecycle management
    - Tool aggregation across servers
    - Routing tool calls to correct client
    - ToolGate registration
    """

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, MCPClient] = {}
        self._configs: Dict[str, MCPServerConfig] = {}

    def add_server(self, config: MCPServerConfig) -> None:
        """
        Add a server configuration.

        Args:
            config: Server configuration
        """
        self._configs[config.id] = config
        _log.debug(f"Added server config: {config.id}")

    def remove_server(self, server_id: str) -> bool:
        """
        Remove a server configuration.

        Note: This only removes the config. For connected servers,
        call disconnect_server() first to properly cleanup.

        Args:
            server_id: Server ID to remove

        Returns:
            True if removed
        """
        if server_id in self._configs:
            del self._configs[server_id]
            # Also remove from clients dict (but don't disconnect - that's async)
            self._clients.pop(server_id, None)
            _log.debug(f"Removed server config: {server_id}")
            return True
        return False

    def get_server_config(self, server_id: str) -> Optional[MCPServerConfig]:
        """Get server configuration by ID."""
        return self._configs.get(server_id)

    def list_server_configs(self) -> List[MCPServerConfig]:
        """List all server configurations."""
        return list(self._configs.values())

    def get_client(self, server_id: str) -> Optional[MCPClient]:
        """Get client by server ID."""
        return self._clients.get(server_id)

    async def connect_server(self, server_id: str) -> bool:
        """
        Connect to a specific server.

        Args:
            server_id: Server ID to connect

        Returns:
            True if connected successfully
        """
        config = self._configs.get(server_id)
        if not config:
            _log.error(f"Unknown server: {server_id}")
            return False

        # Create client if needed
        if server_id not in self._clients:
            self._clients[server_id] = MCPClient(config)

        client = self._clients[server_id]

        # Connect
        success = await client.connect()

        if success:
            # Register tools with ToolGate
            self._register_client_tools(client)

        return success

    async def disconnect_server(self, server_id: str) -> bool:
        """
        Disconnect from a server.

        Args:
            server_id: Server ID to disconnect

        Returns:
            True if disconnected
        """
        client = self._clients.get(server_id)
        if not client:
            return False

        # Unregister tools first
        self._unregister_client_tools(client)

        # Disconnect
        await client.disconnect()

        return True

    async def connect_enabled_servers(self) -> Dict[str, bool]:
        """
        Connect to all enabled servers with auto_connect=True.

        Returns:
            Dict of server_id -> success status
        """
        results = {}

        for config in self._configs.values():
            if config.enabled and config.auto_connect:
                try:
                    results[config.id] = await self.connect_server(config.id)
                except Exception as e:
                    _log.error(f"Failed to connect to {config.id}: {e}")
                    results[config.id] = False

        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for server_id in list(self._clients.keys()):
            try:
                await self.disconnect_server(server_id)
            except Exception as e:
                _log.error(f"Error disconnecting {server_id}: {e}")

    def get_connection_status(self, server_id: str) -> Optional[MCPConnection]:
        """Get connection status for a server."""
        client = self._clients.get(server_id)
        if client:
            return client.get_connection_info()

        # Return disconnected status for configured but not connected servers
        if server_id in self._configs:
            return MCPConnection(server_id=server_id)

        return None

    def list_connections(self) -> List[MCPConnection]:
        """List all connection statuses."""
        connections = []

        for config in self._configs.values():
            client = self._clients.get(config.id)
            if client:
                connections.append(client.get_connection_info())
            else:
                connections.append(MCPConnection(server_id=config.id))

        return connections

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools from all connected servers.

        Returns:
            List of tool dicts with server_id
        """
        tools = []

        for server_id, client in self._clients.items():
            if client.is_connected:
                for tool in client.tools:
                    tools.append({
                        "server_id": server_id,
                        "server_name": client.config.name,
                        **tool.to_dict(),
                    })

        return tools

    def get_server_tools(self, server_id: str) -> List[MCPTool]:
        """Get tools for a specific server."""
        client = self._clients.get(server_id)
        if client and client.is_connected:
            return client.tools
        return []

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> MCPToolCallResult:
        """
        Call a tool on a specific server.

        Args:
            server_id: Server ID
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        client = self._clients.get(server_id)
        if not client:
            return MCPToolCallResult(
                success=False,
                error=f"Unknown server: {server_id}",
            )

        if not client.is_connected:
            return MCPToolCallResult(
                success=False,
                error=f"Server not connected: {server_id}",
            )

        return await client.call_tool(tool_name, arguments)

    async def call_tool_by_full_name(
        self,
        full_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> MCPToolCallResult:
        """
        Call a tool by its full registered name.

        Tool name format: MCP.{server_id}.{tool_name}

        Args:
            full_name: Full tool name
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        parts = full_name.split(".", 2)
        if len(parts) != 3 or parts[0] != "MCP":
            return MCPToolCallResult(
                success=False,
                error=f"Invalid MCP tool name format: {full_name}",
            )

        server_id = parts[1]
        tool_name = parts[2]

        return await self.call_tool(server_id, tool_name, arguments)

    def _register_client_tools(self, client: MCPClient) -> None:
        """Register client's tools with ToolGate."""
        try:
            from cathedral.ToolGate.registry import ToolRegistry

            for tool in client.tools:
                ToolRegistry.register_external_tool(
                    server_id=client.server_id,
                    server_name=client.config.name,
                    tool=tool,
                    policy_class=client.config.default_policy,
                )

            _log.info(
                f"Registered {len(client.tools)} tools from {client.config.name}"
            )

        except ImportError:
            _log.debug("ToolGate not available, skipping tool registration")
        except AttributeError:
            _log.debug("ToolRegistry.register_external_tool not implemented yet")
        except Exception as e:
            _log.warning(f"Failed to register tools: {e}")

    def _unregister_client_tools(self, client: MCPClient) -> None:
        """Unregister client's tools from ToolGate."""
        try:
            from cathedral.ToolGate.registry import ToolRegistry
            ToolRegistry.unregister_server_tools(client.server_id)
            _log.info(f"Unregistered tools from {client.config.name}")
        except ImportError:
            _log.debug("ToolGate not available")
        except AttributeError:
            _log.debug("ToolRegistry.unregister_server_tools not implemented yet")
        except Exception as e:
            _log.warning(f"Failed to unregister tools: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        connected = sum(
            1 for c in self._clients.values() if c.is_connected
        )
        total_tools = sum(
            len(c.tools) for c in self._clients.values() if c.is_connected
        )

        return {
            "configured_servers": len(self._configs),
            "connected_servers": connected,
            "total_tools": total_tools,
        }


# Global manager instance
_manager: Optional[MCPManager] = None


def get_manager() -> MCPManager:
    """Get or create the global manager instance."""
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager


def reset_manager() -> None:
    """Reset the global manager (for testing)."""
    global _manager
    _manager = None


__all__ = ["MCPManager", "get_manager", "reset_manager"]
