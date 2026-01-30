"""
MCPClient - Model Context Protocol Client for Cathedral.

Enables agents to use tools from external MCP servers via stdio transport.

Gate Interface:
    - initialize() - Load config and prepare manager
    - is_initialized() - Check if initialized
    - is_healthy() - Check overall health
    - get_health_status() - Get detailed health info
    - get_dependencies() - List dependencies

Server Management:
    - list_servers() - List configured servers
    - get_server(server_id) - Get server config
    - add_server(config) - Add server configuration
    - update_server(config) - Update server configuration
    - remove_server(server_id) - Remove server configuration

Connection Management:
    - connect_server(server_id) - Connect to a server
    - disconnect_server(server_id) - Disconnect from a server
    - connect_enabled_servers() - Connect all enabled servers
    - list_connections() - List connection statuses

Tool Access:
    - list_tools() - List all MCP tools
    - get_server_tools(server_id) - List tools for a server
    - call_mcp_tool(server_id, tool_name, args) - Execute a tool
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cathedral.shared.gate import GateLogger, build_health_status
from cathedral.MCPClient.config import (
    load_config,
    save_config,
    add_server as config_add_server,
    update_server as config_update_server,
    remove_server as config_remove_server,
)
from cathedral.MCPClient.manager import MCPManager, get_manager
from cathedral.MCPClient.models import (
    ConnectionStatus,
    MCPConnection,
    MCPServerConfig,
    MCPServersConfig,
    MCPTool,
    MCPToolCallResult,
)

_log = GateLogger.get("MCPClient")

# Module state
_initialized = False


# =============================================================================
# Lifecycle
# =============================================================================


def initialize() -> bool:
    """
    Initialize the MCPClient gate.

    Loads server configurations into the manager.

    Returns:
        True if initialized successfully
    """
    global _initialized

    if _initialized:
        return True

    try:
        # Load configuration
        config = load_config()
        manager = get_manager()

        # Add servers to manager
        for server_config in config.servers:
            manager.add_server(server_config)

        _initialized = True
        _log.info(
            f"MCPClient initialized with {len(config.servers)} server(s) configured"
        )
        return True

    except Exception as e:
        _log.error(f"MCPClient initialization failed: {e}")
        return False


def is_initialized() -> bool:
    """Check if the gate is initialized."""
    return _initialized


def is_healthy() -> bool:
    """Check if the gate is healthy."""
    if not _initialized:
        return False

    manager = get_manager()
    stats = manager.get_stats()

    # Healthy if initialized (servers being connected is optional)
    return True


def get_health_status() -> Dict[str, Any]:
    """Get detailed health status."""
    manager = get_manager() if _initialized else None
    stats = manager.get_stats() if manager else {}

    connections = list_connections() if _initialized else []
    connected_count = sum(
        1 for c in connections if c.status == ConnectionStatus.CONNECTED
    )
    error_count = sum(
        1 for c in connections if c.status == ConnectionStatus.ERROR
    )

    return build_health_status(
        gate_name="MCPClient",
        initialized=_initialized,
        dependencies=["subprocess", "json-rpc"],
        checks={
            "manager_ready": manager is not None,
        },
        details={
            "configured_servers": stats.get("configured_servers", 0),
            "connected_servers": connected_count,
            "error_servers": error_count,
            "total_tools": stats.get("total_tools", 0),
        },
    )


def get_dependencies() -> List[str]:
    """Get list of external dependencies."""
    return ["subprocess", "json-rpc"]


# =============================================================================
# Server Configuration
# =============================================================================


def list_servers() -> List[MCPServerConfig]:
    """
    List all configured MCP servers.

    Returns:
        List of server configurations
    """
    manager = get_manager()
    return manager.list_server_configs()


def get_server(server_id: str) -> Optional[MCPServerConfig]:
    """
    Get a server configuration by ID.

    Args:
        server_id: Server ID

    Returns:
        Server configuration or None
    """
    manager = get_manager()
    return manager.get_server_config(server_id)


def add_server(
    id: str,
    name: str,
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    enabled: bool = True,
    auto_connect: bool = True,
    default_policy: str = "network",
) -> bool:
    """
    Add a new MCP server configuration.

    Args:
        id: Unique server ID
        name: Display name
        command: Executable command
        args: Command arguments
        env: Environment variables
        enabled: Whether server is active
        auto_connect: Connect on startup
        default_policy: Default policy class for tools

    Returns:
        True if added successfully
    """
    config = MCPServerConfig(
        id=id,
        name=name,
        command=command,
        args=args or [],
        env=env or {},
        enabled=enabled,
        auto_connect=auto_connect,
        default_policy=default_policy,
    )

    # Add to manager
    manager = get_manager()
    manager.add_server(config)

    # Persist to file
    return config_add_server(config)


def update_server(
    id: str,
    name: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    enabled: Optional[bool] = None,
    auto_connect: Optional[bool] = None,
    default_policy: Optional[str] = None,
) -> bool:
    """
    Update an MCP server configuration.

    Args:
        id: Server ID to update
        name: New display name
        command: New command
        args: New arguments
        env: New environment variables
        enabled: New enabled state
        auto_connect: New auto_connect state
        default_policy: New default policy

    Returns:
        True if updated successfully
    """
    existing = get_server(id)
    if not existing:
        _log.error(f"Server not found: {id}")
        return False

    # Update fields
    updated = MCPServerConfig(
        id=id,
        name=name if name is not None else existing.name,
        command=command if command is not None else existing.command,
        args=args if args is not None else existing.args,
        env=env if env is not None else existing.env,
        enabled=enabled if enabled is not None else existing.enabled,
        auto_connect=auto_connect if auto_connect is not None else existing.auto_connect,
        default_policy=default_policy if default_policy is not None else existing.default_policy,
    )

    # Update in manager
    manager = get_manager()
    manager.remove_server(id)
    manager.add_server(updated)

    # Persist to file
    return config_update_server(updated)


def remove_server(server_id: str) -> bool:
    """
    Remove an MCP server configuration.

    Args:
        server_id: Server ID to remove

    Returns:
        True if removed successfully
    """
    manager = get_manager()
    manager.remove_server(server_id)

    return config_remove_server(server_id)


# =============================================================================
# Connection Management
# =============================================================================


async def connect_server(server_id: str) -> bool:
    """
    Connect to an MCP server.

    Args:
        server_id: Server ID to connect

    Returns:
        True if connected successfully
    """
    manager = get_manager()
    return await manager.connect_server(server_id)


async def disconnect_server(server_id: str) -> bool:
    """
    Disconnect from an MCP server.

    Args:
        server_id: Server ID to disconnect

    Returns:
        True if disconnected successfully
    """
    manager = get_manager()
    return await manager.disconnect_server(server_id)


async def connect_enabled_servers() -> Dict[str, bool]:
    """
    Connect to all enabled servers with auto_connect=True.

    Returns:
        Dict of server_id -> success status
    """
    manager = get_manager()
    return await manager.connect_enabled_servers()


async def disconnect_all() -> None:
    """Disconnect from all servers."""
    manager = get_manager()
    await manager.disconnect_all()


def list_connections() -> List[MCPConnection]:
    """
    List all connection statuses.

    Returns:
        List of connection info objects
    """
    manager = get_manager()
    return manager.list_connections()


def get_connection_status(server_id: str) -> Optional[MCPConnection]:
    """
    Get connection status for a server.

    Args:
        server_id: Server ID

    Returns:
        Connection info or None
    """
    manager = get_manager()
    return manager.get_connection_status(server_id)


# =============================================================================
# Tool Access
# =============================================================================


def list_tools() -> List[Dict[str, Any]]:
    """
    List all tools from all connected servers.

    Returns:
        List of tool dicts with server info
    """
    manager = get_manager()
    return manager.list_all_tools()


def get_server_tools(server_id: str) -> List[MCPTool]:
    """
    List tools for a specific server.

    Args:
        server_id: Server ID

    Returns:
        List of tools
    """
    manager = get_manager()
    return manager.get_server_tools(server_id)


async def call_mcp_tool(
    server_id: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> MCPToolCallResult:
    """
    Call a tool on an MCP server.

    Args:
        server_id: Server ID
        tool_name: Tool name
        arguments: Tool arguments

    Returns:
        Tool call result
    """
    manager = get_manager()
    return await manager.call_tool(server_id, tool_name, arguments)


async def call_tool_by_full_name(
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
    manager = get_manager()
    return await manager.call_tool_by_full_name(full_name, arguments)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Lifecycle
    "initialize",
    "is_initialized",
    "is_healthy",
    "get_health_status",
    "get_dependencies",
    # Server Configuration
    "list_servers",
    "get_server",
    "add_server",
    "update_server",
    "remove_server",
    # Connection Management
    "connect_server",
    "disconnect_server",
    "connect_enabled_servers",
    "disconnect_all",
    "list_connections",
    "get_connection_status",
    # Tool Access
    "list_tools",
    "get_server_tools",
    "call_mcp_tool",
    "call_tool_by_full_name",
    # Models (re-export for convenience)
    "MCPServerConfig",
    "MCPConnection",
    "MCPTool",
    "MCPToolCallResult",
    "ConnectionStatus",
]
