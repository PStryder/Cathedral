"""
SubAgent API endpoints for spawning and managing sub-agents.

Provides REST endpoints for SubAgentGate operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


class SpawnAgentRequest(BaseModel):
    """Request model for spawning a sub-agent."""
    task: str
    context: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    personality: Optional[str] = None
    model: Optional[str] = None


def create_router(templates, SubAgentGate, emit_event) -> APIRouter:
    """Create SubAgent API router."""
    router = APIRouter()

    # ==================== UI Page ====================

    @router.get("/agents", response_class=HTMLResponse)
    async def agents_page(request: Request):
        """Serve the sub-agents management UI."""
        return templates.TemplateResponse("agents.html", {"request": request})

    # ==================== Spawn Operations ====================

    @router.post("/api/agents/spawn")
    async def api_spawn_agent(data: SpawnAgentRequest):
        """
        Spawn a new sub-agent to handle a task.

        The agent runs asynchronously and can be polled for status/result.
        """
        agent_id = SubAgentGate.spawn(
            task=data.task,
            context=data.context,
            system_prompt=data.system_prompt,
            max_tokens=data.max_tokens,
            temperature=data.temperature,
            personality=data.personality,
            model=data.model,
        )

        await emit_event(
            "subagent",
            f"Spawned sub-agent: {agent_id}",
            operation="spawn",
            agent_id=agent_id,
            task_preview=data.task[:100],
        )

        # Get initial status
        status = SubAgentGate.status(agent_id)

        return {"agent_id": agent_id, "status": status}

    # ==================== Status & List ====================

    @router.get("/api/agents")
    async def api_list_agents(include_completed: bool = True):
        """List all agents."""
        agents = SubAgentGate.list_agents(include_completed=include_completed)
        return {"agents": agents, "count": len(agents)}

    @router.get("/api/agents/running")
    async def api_list_running_agents():
        """List only running agents."""
        agents = SubAgentGate.list_agents(include_completed=False)
        return {"agents": agents, "count": len(agents)}

    @router.get("/api/agents/{agent_id}")
    async def api_get_agent_status(agent_id: str):
        """Get status of a specific agent."""
        status = SubAgentGate.status(agent_id)

        if status is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        return status

    @router.get("/api/agents/{agent_id}/result")
    async def api_get_agent_result(agent_id: str):
        """Get the result of a completed agent."""
        status = SubAgentGate.status(agent_id)

        if status is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        result = SubAgentGate.result(agent_id)

        return {
            "agent_id": agent_id,
            "status": status.get("status"),
            "result": result,
            "error": status.get("error"),
        }

    # ==================== Control Operations ====================

    @router.post("/api/agents/{agent_id}/cancel")
    async def api_cancel_agent(agent_id: str):
        """Cancel a running agent."""
        success = SubAgentGate.cancel(agent_id)

        if not success:
            # Check if agent exists
            status = SubAgentGate.status(agent_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Agent not found")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel agent in status: {status.get('status')}"
                )

        await emit_event(
            "subagent",
            f"Cancelled sub-agent: {agent_id}",
            operation="cancel",
            agent_id=agent_id,
        )

        return {"status": "cancelled", "agent_id": agent_id}

    @router.post("/api/agents/check")
    async def api_check_completed():
        """
        Check for newly completed agents.

        Returns list of agent IDs that just completed.
        Useful for polling.
        """
        newly_completed = SubAgentGate.check_completed()

        # Emit events for completed agents
        for agent_id in newly_completed:
            status = SubAgentGate.status(agent_id)
            await emit_event(
                "subagent",
                f"Sub-agent completed: {agent_id}",
                operation="completed",
                agent_id=agent_id,
                status=status.get("status") if status else "unknown",
            )

        return {"newly_completed": newly_completed, "count": len(newly_completed)}

    @router.post("/api/agents/cleanup")
    async def api_cleanup_old_agents(max_age_hours: int = 24):
        """
        Clean up old completed agents.

        Removes agents older than max_age_hours from memory and disk.
        """
        manager = SubAgentGate.get_manager()
        removed_count = manager.cleanup_old(max_age_hours=max_age_hours)

        await emit_event(
            "subagent",
            f"Cleaned up {removed_count} old agents",
            operation="cleanup",
            removed_count=removed_count,
        )

        return {"status": "cleaned", "removed_count": removed_count}

    return router


__all__ = ["create_router"]
