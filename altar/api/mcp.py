"""
MCP API Router.

API endpoints for MCP server management, connection control, and tool listing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


class MCPServerCreate(BaseModel):
    """Model for creating a new MCP server."""

    id: str = Field(description="Unique server ID")
    name: str = Field(description="Display name")
    command: str = Field(description="Executable command")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    enabled: bool = Field(default=True)
    auto_connect: bool = Field(default=True)
    default_policy: str = Field(default="network")


class MCPServerUpdate(BaseModel):
    """Model for updating an MCP server."""

    name: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None
    auto_connect: Optional[bool] = None
    default_policy: Optional[str] = None


def create_router(templates, MCPClient, emit_event) -> APIRouter:
    """
    Create the MCP API router.

    Args:
        templates: Jinja2 templates instance
        MCPClient: MCPClient module
        emit_event: Event emitter function

    Returns:
        FastAPI router
    """
    router = APIRouter()

    # =========================================================================
    # UI Page
    # =========================================================================

    @router.get("/mcp", response_class=HTMLResponse)
    async def mcp_page(request: Request):
        """Serve the MCP management UI."""
        return templates.TemplateResponse("mcp.html", {"request": request})

    # =========================================================================
    # Server Configuration Endpoints
    # =========================================================================

    @router.get("/api/mcp/servers")
    async def api_list_servers() -> Dict[str, Any]:
        """List all MCP servers with connection status."""
        servers = MCPClient.list_servers()
        connections = MCPClient.list_connections()

        # Build status lookup
        status_lookup = {c.server_id: c for c in connections}

        result = []
        for server in servers:
            conn = status_lookup.get(server.id)
            result.append({
                **server.to_dict(),
                "status": conn.status.value if conn else "disconnected",
                "error_message": conn.error_message if conn else None,
                "connected_at": conn.connected_at.isoformat() if conn and conn.connected_at else None,
                "tool_count": len(conn.tools) if conn else 0,
            })

        return {"servers": result}

    @router.post("/api/mcp/servers")
    async def api_add_server(data: MCPServerCreate) -> Dict[str, Any]:
        """Add a new MCP server."""
        success = MCPClient.add_server(
            id=data.id,
            name=data.name,
            command=data.command,
            args=data.args,
            env=data.env,
            enabled=data.enabled,
            auto_connect=data.auto_connect,
            default_policy=data.default_policy,
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to add server (ID may already exist)")

        await emit_event("mcp", f"Added server: {data.name}", operation="add", server_id=data.id)

        return {"status": "added", "id": data.id}

    @router.get("/api/mcp/servers/{server_id}")
    async def api_get_server(server_id: str) -> Dict[str, Any]:
        """Get a specific MCP server configuration."""
        server = MCPClient.get_server(server_id)
        if not server:
            raise HTTPException(status_code=404, detail="Server not found")

        conn = MCPClient.get_connection_status(server_id)

        return {
            **server.to_dict(),
            "status": conn.status.value if conn else "disconnected",
            "error_message": conn.error_message if conn else None,
            "tool_count": len(conn.tools) if conn else 0,
        }

    @router.put("/api/mcp/servers/{server_id}")
    async def api_update_server(server_id: str, data: MCPServerUpdate) -> Dict[str, Any]:
        """Update an MCP server configuration."""
        success = MCPClient.update_server(
            id=server_id,
            name=data.name,
            command=data.command,
            args=data.args,
            env=data.env,
            enabled=data.enabled,
            auto_connect=data.auto_connect,
            default_policy=data.default_policy,
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to update server")

        await emit_event("mcp", f"Updated server: {server_id}", operation="update", server_id=server_id)

        return {"status": "updated"}

    @router.delete("/api/mcp/servers/{server_id}")
    async def api_delete_server(server_id: str) -> Dict[str, Any]:
        """Delete an MCP server configuration."""
        # Disconnect first if connected
        try:
            await MCPClient.disconnect_server(server_id)
        except Exception:
            pass

        success = MCPClient.remove_server(server_id)
        if not success:
            raise HTTPException(status_code=404, detail="Server not found")

        await emit_event("mcp", f"Removed server: {server_id}", operation="remove", server_id=server_id)

        return {"status": "removed"}

    # =========================================================================
    # Connection Management Endpoints
    # =========================================================================

    @router.post("/api/mcp/servers/{server_id}/connect")
    async def api_connect_server(server_id: str) -> Dict[str, Any]:
        """Connect to an MCP server."""
        server = MCPClient.get_server(server_id)
        if not server:
            raise HTTPException(status_code=404, detail="Server not found")

        success = await MCPClient.connect_server(server_id)

        if success:
            await emit_event("mcp", f"Connected to: {server.name}", operation="connect", server_id=server_id)
            conn = MCPClient.get_connection_status(server_id)
            return {
                "status": "connected",
                "tool_count": len(conn.tools) if conn else 0,
            }
        else:
            conn = MCPClient.get_connection_status(server_id)
            error = conn.error_message if conn else "Unknown error"
            raise HTTPException(status_code=500, detail=f"Connection failed: {error}")

    @router.post("/api/mcp/servers/{server_id}/disconnect")
    async def api_disconnect_server(server_id: str) -> Dict[str, Any]:
        """Disconnect from an MCP server."""
        success = await MCPClient.disconnect_server(server_id)

        if success:
            await emit_event("mcp", f"Disconnected from: {server_id}", operation="disconnect", server_id=server_id)
            return {"status": "disconnected"}
        else:
            raise HTTPException(status_code=400, detail="Failed to disconnect")

    @router.post("/api/mcp/servers/{server_id}/test")
    async def api_test_server(server_id: str) -> Dict[str, Any]:
        """Test connection to an MCP server."""
        server = MCPClient.get_server(server_id)
        if not server:
            raise HTTPException(status_code=404, detail="Server not found")

        # Try to connect
        success = await MCPClient.connect_server(server_id)

        if success:
            conn = MCPClient.get_connection_status(server_id)
            tool_count = len(conn.tools) if conn else 0

            # Disconnect after test
            await MCPClient.disconnect_server(server_id)

            return {
                "status": "success",
                "message": f"Connection successful. Found {tool_count} tools.",
                "tool_count": tool_count,
            }
        else:
            conn = MCPClient.get_connection_status(server_id)
            error = conn.error_message if conn else "Unknown error"
            return {
                "status": "error",
                "message": f"Connection failed: {error}",
            }

    # =========================================================================
    # Tool Endpoints
    # =========================================================================

    @router.get("/api/mcp/tools")
    async def api_list_all_tools() -> Dict[str, Any]:
        """List all tools from all connected MCP servers."""
        tools = MCPClient.list_tools()
        return {"tools": tools}

    @router.get("/api/mcp/servers/{server_id}/tools")
    async def api_list_server_tools(server_id: str) -> Dict[str, Any]:
        """List tools for a specific MCP server."""
        server = MCPClient.get_server(server_id)
        if not server:
            raise HTTPException(status_code=404, detail="Server not found")

        tools = MCPClient.get_server_tools(server_id)
        return {
            "server_id": server_id,
            "tools": [t.to_dict() for t in tools],
        }

    # =========================================================================
    # Status Endpoints
    # =========================================================================

    @router.get("/api/mcp/status")
    async def api_status() -> Dict[str, Any]:
        """Get MCP system status."""
        health = MCPClient.get_health_status()
        return {
            "healthy": health.get("healthy", False),
            "initialized": health.get("initialized", False),
            "details": health.get("details", {}),
        }

    @router.get("/api/mcp/connections")
    async def api_list_connections() -> Dict[str, Any]:
        """List all connection statuses."""
        connections = MCPClient.list_connections()
        return {
            "connections": [c.to_dict() for c in connections],
        }

    return router


__all__ = ["create_router"]
