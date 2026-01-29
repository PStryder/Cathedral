from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse


class ShellExecuteRequest(BaseModel):
    """Model for executing a command."""
    command: str
    working_dir: str | None = None
    timeout: int | None = None


class ShellConfigUpdate(BaseModel):
    """Model for updating shell configuration."""
    default_timeout_seconds: int | None = None
    max_timeout_seconds: int | None = None
    default_working_dir: str | None = None
    allowed_commands: list | None = None
    blocked_commands: list | None = None
    require_unlock: bool | None = None
    log_commands: bool | None = None
    max_concurrent_background: int | None = None


def create_router(templates, ShellGate, emit_event) -> APIRouter:
    router = APIRouter()

    @router.get("/shell", response_class=HTMLResponse)
    async def shell_page(request: Request):
        """Serve the shell execution UI."""
        return templates.TemplateResponse("shell.html", {"request": request})

    @router.post("/api/shell/execute")
    async def api_shell_execute(data: ShellExecuteRequest):
        """Execute a command synchronously."""
        result = ShellGate.execute(data.command, data.working_dir, data.timeout)

        if result.success:
            await emit_event(
                "shell",
                f"Command completed: {data.command[:50]}",
                operation="execute",
                exit_code=result.exit_code,
            )

        return result.to_dict()

    @router.post("/api/shell/stream")
    async def api_shell_stream(data: ShellExecuteRequest):
        """Execute a command with streaming output via SSE."""

        async def generate():
            try:
                async for line in ShellGate.execute_stream(
                    data.command, data.working_dir, data.timeout
                ):
                    yield {"data": json.dumps({"output": line})}
                yield {"data": json.dumps({"done": True})}
            except Exception as e:
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(generate())

    @router.post("/api/shell/background")
    async def api_shell_background(data: ShellExecuteRequest):
        """Execute a command in the background."""
        execution = ShellGate.execute_background(data.command, data.working_dir)

        if execution.status.value == "failed":
            raise HTTPException(status_code=400, detail=execution.stderr)

        await emit_event(
            "shell",
            f"Background command started: {data.command[:50]}",
            operation="background",
            execution_id=execution.id,
        )

        return execution.to_dict()

    @router.get("/api/shell/status/{execution_id}")
    async def api_shell_status(execution_id: str):
        """Get status of a background command."""
        execution = ShellGate.get_status(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution.to_dict()

    @router.post("/api/shell/cancel/{execution_id}")
    async def api_shell_cancel(execution_id: str):
        """Cancel a background command."""
        success = ShellGate.cancel(execution_id)
        if not success:
            raise HTTPException(status_code=400, detail="Could not cancel (not found or already complete)")

        await emit_event(
            "shell",
            f"Command cancelled: {execution_id}",
            operation="cancel",
            execution_id=execution_id,
        )

        return {"status": "cancelled"}

    @router.get("/api/shell/running")
    async def api_shell_running():
        """List running background commands."""
        return {"running": ShellGate.list_running()}

    @router.get("/api/shell/background")
    async def api_shell_list_background():
        """List all background commands."""
        return {"commands": ShellGate.list_background()}

    @router.get("/api/shell/history")
    async def api_shell_history(limit: int = 50, success_only: bool = False):
        """Get command history."""
        return {"history": ShellGate.get_history(limit, success_only)}

    @router.delete("/api/shell/history")
    async def api_shell_clear_history():
        """Clear command history."""
        count = ShellGate.clear_history()
        return {"status": "cleared", "count": count}

    @router.get("/api/shell/validate")
    async def api_shell_validate(command: str):
        """Validate a command without executing."""
        is_valid, error = ShellGate.validate_command(command)
        risk = ShellGate.estimate_risk(command)
        return {
            "valid": is_valid,
            "error": error,
            "risk": risk,
        }

    @router.get("/api/shell/config")
    async def api_shell_config():
        """Get shell configuration."""
        return ShellGate.get_config()

    @router.put("/api/shell/config")
    async def api_shell_config_update(data: ShellConfigUpdate):
        """Update shell configuration."""
        updates = {k: v for k, v in data.model_dump().items() if v is not None}
        ShellGate.update_config(**updates)
        return {"status": "updated"}

    @router.post("/api/shell/blocked")
    async def api_shell_add_blocked(command: str):
        """Add a command to the blocklist."""
        ShellGate.add_blocked_command(command)
        return {"status": "added"}

    @router.delete("/api/shell/blocked")
    async def api_shell_remove_blocked(command: str):
        """Remove a command from the blocklist."""
        success = ShellGate.remove_blocked_command(command)
        if not success:
            raise HTTPException(status_code=404, detail="Command not in blocklist")
        return {"status": "removed"}

    return router


__all__ = ["create_router"]
