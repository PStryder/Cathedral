"""
MCPClient Models.

Pydantic models for MCP server configuration, connections, and tools.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TransportType(str, Enum):
    """Supported MCP transport types."""

    STDIO = "stdio"


class ConnectionStatus(str, Enum):
    """MCP server connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    id: str = Field(description="Unique identifier for the server")
    name: str = Field(description="Human-readable display name")
    command: str = Field(description="Executable command (e.g., 'npx', 'python')")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    transport: TransportType = Field(default=TransportType.STDIO)
    enabled: bool = Field(default=True, description="Whether server is active")
    auto_connect: bool = Field(default=True, description="Connect on startup")
    default_policy: str = Field(
        default="network", description="Policy class for tools (read_only, network, write, privileged)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "transport": self.transport.value,
            "enabled": self.enabled,
            "auto_connect": self.auto_connect,
            "default_policy": self.default_policy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
        """Create from dictionary."""
        if "transport" in data and isinstance(data["transport"], str):
            data = dict(data)
            data["transport"] = TransportType(data["transport"])
        return cls(**data)


class MCPToolInputSchema(BaseModel):
    """JSON Schema for MCP tool input."""

    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class MCPTool(BaseModel):
    """MCP tool definition returned from tools/list."""

    name: str = Field(description="Tool name")
    description: str = Field(default="", description="Tool description")
    input_schema: MCPToolInputSchema = Field(
        default_factory=MCPToolInputSchema, alias="inputSchema"
    )

    class Config:
        populate_by_name = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": self.input_schema.type,
                "properties": self.input_schema.properties,
                "required": self.input_schema.required,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTool":
        """Create from dictionary."""
        input_schema_data = data.get("inputSchema", data.get("input_schema", {}))
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_schema=MCPToolInputSchema(
                type=input_schema_data.get("type", "object"),
                properties=input_schema_data.get("properties", {}),
                required=input_schema_data.get("required", []),
            ),
        )


class MCPConnection(BaseModel):
    """Represents an active MCP server connection."""

    server_id: str = Field(description="Server configuration ID")
    status: ConnectionStatus = Field(default=ConnectionStatus.DISCONNECTED)
    error_message: Optional[str] = Field(default=None)
    connected_at: Optional[datetime] = Field(default=None)
    tools: List[MCPTool] = Field(default_factory=list)
    pid: Optional[int] = Field(default=None, description="Subprocess PID")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_id": self.server_id,
            "status": self.status.value,
            "error_message": self.error_message,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "tool_count": len(self.tools),
            "tools": [t.to_dict() for t in self.tools],
            "pid": self.pid,
        }


class MCPServersConfig(BaseModel):
    """Root configuration for all MCP servers."""

    version: str = Field(default="1.0")
    servers: List[MCPServerConfig] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "servers": [s.to_dict() for s in self.servers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServersConfig":
        """Create from dictionary."""
        servers = [MCPServerConfig.from_dict(s) for s in data.get("servers", [])]
        return cls(version=data.get("version", "1.0"), servers=servers)


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: str = "2.0"
    id: int
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response."""

    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    def is_error(self) -> bool:
        """Check if response is an error."""
        return self.error is not None


class MCPToolCallResult(BaseModel):
    """Result from calling an MCP tool."""

    success: bool
    content: Optional[Any] = None
    error: Optional[str] = None
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "is_error": self.is_error,
        }


__all__ = [
    "TransportType",
    "ConnectionStatus",
    "MCPServerConfig",
    "MCPToolInputSchema",
    "MCPTool",
    "MCPConnection",
    "MCPServersConfig",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPToolCallResult",
]
