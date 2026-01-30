"""
MCPClient Configuration.

Handles loading and saving MCP server configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from cathedral.shared.gate import GateLogger, PathUtils
from cathedral.MCPClient.models import MCPServerConfig, MCPServersConfig

_log = GateLogger.get("MCPClient.Config")

# Default config path
DEFAULT_CONFIG_PATH = Path("data/mcp_servers.json")


def get_config_path() -> Path:
    """Get the configuration file path."""
    return DEFAULT_CONFIG_PATH


def load_config(path: Optional[Path] = None) -> MCPServersConfig:
    """
    Load MCP servers configuration from file.

    Args:
        path: Config file path (defaults to data/mcp_servers.json)

    Returns:
        MCPServersConfig instance
    """
    config_path = path or get_config_path()

    if not config_path.exists():
        _log.debug(f"Config file not found, using defaults: {config_path}")
        return MCPServersConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = MCPServersConfig.from_dict(data)
        _log.info(f"Loaded {len(config.servers)} MCP server configs from {config_path}")
        return config

    except json.JSONDecodeError as e:
        _log.error(f"Invalid JSON in config file: {e}")
        return MCPServersConfig()
    except Exception as e:
        _log.error(f"Failed to load config: {e}")
        return MCPServersConfig()


def save_config(config: MCPServersConfig, path: Optional[Path] = None) -> bool:
    """
    Save MCP servers configuration to file.

    Args:
        config: Configuration to save
        path: Config file path (defaults to data/mcp_servers.json)

    Returns:
        True if saved successfully
    """
    config_path = path or get_config_path()

    try:
        # Ensure parent directory exists
        PathUtils.ensure_dirs(config_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        _log.info(f"Saved {len(config.servers)} MCP server configs to {config_path}")
        return True

    except Exception as e:
        _log.error(f"Failed to save config: {e}")
        return False


def add_server(server: MCPServerConfig, path: Optional[Path] = None) -> bool:
    """
    Add a server to the configuration.

    Args:
        server: Server configuration to add
        path: Config file path

    Returns:
        True if added successfully
    """
    config = load_config(path)

    # Check for duplicate ID
    for existing in config.servers:
        if existing.id == server.id:
            _log.warning(f"Server with ID {server.id} already exists")
            return False

    config.servers.append(server)
    return save_config(config, path)


def update_server(server: MCPServerConfig, path: Optional[Path] = None) -> bool:
    """
    Update an existing server configuration.

    Args:
        server: Updated server configuration
        path: Config file path

    Returns:
        True if updated successfully
    """
    config = load_config(path)

    for i, existing in enumerate(config.servers):
        if existing.id == server.id:
            config.servers[i] = server
            return save_config(config, path)

    _log.warning(f"Server with ID {server.id} not found")
    return False


def remove_server(server_id: str, path: Optional[Path] = None) -> bool:
    """
    Remove a server from the configuration.

    Args:
        server_id: Server ID to remove
        path: Config file path

    Returns:
        True if removed successfully
    """
    config = load_config(path)

    original_count = len(config.servers)
    config.servers = [s for s in config.servers if s.id != server_id]

    if len(config.servers) == original_count:
        _log.warning(f"Server with ID {server_id} not found")
        return False

    return save_config(config, path)


def get_server(server_id: str, path: Optional[Path] = None) -> Optional[MCPServerConfig]:
    """
    Get a server configuration by ID.

    Args:
        server_id: Server ID
        path: Config file path

    Returns:
        Server configuration or None
    """
    config = load_config(path)

    for server in config.servers:
        if server.id == server_id:
            return server

    return None


def list_servers(path: Optional[Path] = None) -> List[MCPServerConfig]:
    """
    List all server configurations.

    Args:
        path: Config file path

    Returns:
        List of server configurations
    """
    config = load_config(path)
    return config.servers


__all__ = [
    "get_config_path",
    "load_config",
    "save_config",
    "add_server",
    "update_server",
    "remove_server",
    "get_server",
    "list_servers",
]
