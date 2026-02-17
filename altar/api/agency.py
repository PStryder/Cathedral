from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class SpawnRequest(BaseModel):
    session_id: str | None = None
    shell: str | None = None
    cwd: str | None = None


class ExecRequest(BaseModel):
    session_id: str
    command: str
    timeout_ms: int = 30000


def create_router(AgencyGate, emit_event) -> APIRouter:
    router = APIRouter()

    @router.post("/api/agency/spawn")
    async def api_agency_spawn(data: SpawnRequest):
        result = await AgencyGate.spawn(data.session_id, data.shell, data.cwd)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        await emit_event("agency", f"Session spawned: {result.get('session_id')}")
        return result

    @router.post("/api/agency/exec")
    async def api_agency_exec(data: ExecRequest):
        result = await AgencyGate.exec(data.session_id, data.command, data.timeout_ms)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
        return result

    @router.get("/api/agency/sessions")
    async def api_agency_list():
        return {"sessions": AgencyGate.list_sessions()}

    @router.post("/api/agency/close/{session_id}")
    async def api_agency_close(session_id: str):
        result = await AgencyGate.close(session_id)
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result.get("error"))
        await emit_event("agency", f"Session closed: {session_id}")
        return result

    @router.get("/api/agency/health")
    async def api_agency_health():
        return AgencyGate.get_health_status()

    return router


__all__ = ["create_router"]
