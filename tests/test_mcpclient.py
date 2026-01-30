"""
Tests for MCPClient module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cathedral.MCPClient.models import (
    MCPServerConfig,
    MCPServersConfig,
    MCPTool,
    MCPToolInputSchema,
    MCPConnection,
    ConnectionStatus,
    MCPToolCallResult,
    TransportType,
)
from cathedral.MCPClient.config import (
    load_config,
    save_config,
)
from cathedral.MCPClient.manager import MCPManager


class TestModels:
    """Test Pydantic models."""

    def test_server_config_creation(self):
        """Test MCPServerConfig creation."""
        config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            command="npx",
            args=["-y", "@test/server"],
            env={"DEBUG": "true"},
        )

        assert config.id == "test-server"
        assert config.name == "Test Server"
        assert config.command == "npx"
        assert config.args == ["-y", "@test/server"]
        assert config.env == {"DEBUG": "true"}
        assert config.transport == TransportType.STDIO
        assert config.enabled is True
        assert config.auto_connect is True
        assert config.default_policy == "network"

    def test_server_config_to_dict(self):
        """Test MCPServerConfig serialization."""
        config = MCPServerConfig(
            id="test",
            name="Test",
            command="python",
        )

        data = config.to_dict()

        assert data["id"] == "test"
        assert data["name"] == "Test"
        assert data["command"] == "python"
        assert data["transport"] == "stdio"

    def test_server_config_from_dict(self):
        """Test MCPServerConfig deserialization."""
        data = {
            "id": "test",
            "name": "Test",
            "command": "node",
            "args": ["server.js"],
            "transport": "stdio",
        }

        config = MCPServerConfig.from_dict(data)

        assert config.id == "test"
        assert config.command == "node"
        assert config.transport == TransportType.STDIO

    def test_mcp_tool_creation(self):
        """Test MCPTool creation."""
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema=MCPToolInputSchema(
                properties={"query": {"type": "string"}},
                required=["query"],
            ),
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "query" in tool.input_schema.properties

    def test_mcp_tool_from_dict(self):
        """Test MCPTool deserialization."""
        data = {
            "name": "search",
            "description": "Search for items",
            "inputSchema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }

        tool = MCPTool.from_dict(data)

        assert tool.name == "search"
        assert "q" in tool.input_schema.properties

    def test_mcp_connection(self):
        """Test MCPConnection."""
        conn = MCPConnection(server_id="test")

        assert conn.server_id == "test"
        assert conn.status == ConnectionStatus.DISCONNECTED
        assert conn.tools == []

    def test_mcp_tool_call_result(self):
        """Test MCPToolCallResult."""
        result = MCPToolCallResult(
            success=True,
            content={"data": "test"},
        )

        assert result.success is True
        assert result.content == {"data": "test"}
        assert result.error is None

    def test_servers_config(self):
        """Test MCPServersConfig."""
        config = MCPServersConfig(
            servers=[
                MCPServerConfig(id="s1", name="Server 1", command="cmd1"),
                MCPServerConfig(id="s2", name="Server 2", command="cmd2"),
            ]
        )

        assert len(config.servers) == 2
        assert config.version == "1.0"

        data = config.to_dict()
        assert len(data["servers"]) == 2

        restored = MCPServersConfig.from_dict(data)
        assert len(restored.servers) == 2


class TestManager:
    """Test MCPManager."""

    def test_manager_creation(self):
        """Test MCPManager initialization."""
        manager = MCPManager()

        assert len(manager.list_server_configs()) == 0
        assert len(manager.list_connections()) == 0

    def test_add_server(self):
        """Test adding a server config."""
        manager = MCPManager()

        config = MCPServerConfig(
            id="test",
            name="Test",
            command="echo",
        )

        manager.add_server(config)

        assert len(manager.list_server_configs()) == 1
        assert manager.get_server_config("test") is not None

    def test_remove_server(self):
        """Test removing a server config."""
        manager = MCPManager()

        config = MCPServerConfig(id="test", name="Test", command="echo")
        manager.add_server(config)

        assert manager.remove_server("test") is True
        assert manager.get_server_config("test") is None

    def test_get_stats(self):
        """Test getting manager stats."""
        manager = MCPManager()

        config = MCPServerConfig(id="test", name="Test", command="echo")
        manager.add_server(config)

        stats = manager.get_stats()

        assert stats["configured_servers"] == 1
        assert stats["connected_servers"] == 0
        assert stats["total_tools"] == 0


class TestConfig:
    """Test config file I/O."""

    def test_load_missing_config(self, tmp_path):
        """Test loading a non-existent config file."""
        config_path = tmp_path / "missing.json"

        config = load_config(config_path)

        assert isinstance(config, MCPServersConfig)
        assert len(config.servers) == 0

    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading config."""
        config_path = tmp_path / "servers.json"

        original = MCPServersConfig(
            servers=[
                MCPServerConfig(id="s1", name="Server 1", command="cmd1"),
            ]
        )

        assert save_config(original, config_path) is True

        loaded = load_config(config_path)

        assert len(loaded.servers) == 1
        assert loaded.servers[0].id == "s1"


class TestGateInterface:
    """Test MCPClient gate interface."""

    def test_initialize(self):
        """Test MCPClient initialization."""
        from cathedral import MCPClient
        from cathedral.MCPClient.manager import reset_manager

        # Reset for clean test
        reset_manager()

        result = MCPClient.initialize()

        assert result is True
        assert MCPClient.is_initialized() is True

    def test_health_status(self):
        """Test health status."""
        from cathedral import MCPClient

        # Ensure initialized
        MCPClient.initialize()

        status = MCPClient.get_health_status()

        assert "gate" in status
        assert status["gate"] == "MCPClient"
        assert "initialized" in status
        assert "healthy" in status

    def test_list_servers_empty(self):
        """Test listing servers when none configured."""
        from cathedral import MCPClient
        from cathedral.MCPClient.manager import reset_manager

        reset_manager()
        MCPClient.initialize()

        servers = MCPClient.list_servers()

        # May or may not have servers depending on data/mcp_servers.json
        assert isinstance(servers, list)

    def test_list_tools_empty(self):
        """Test listing tools when no servers connected."""
        from cathedral import MCPClient
        from cathedral.MCPClient.manager import reset_manager

        reset_manager()
        MCPClient.initialize()

        tools = MCPClient.list_tools()

        assert isinstance(tools, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
