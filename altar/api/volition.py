from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel


class ContinueRequest(BaseModel):
    text: str
    reason: str = ""


def create_router(VolitionGate, emit_event) -> APIRouter:
    router = APIRouter()

    @router.post("/api/volition/continue")
    async def api_volition_continue(data: ContinueRequest):
        result = VolitionGate.request_continue(data.text, data.reason)
        if result["status"] == "approved":
            await emit_event("volition", f"Continuation approved (turn {result['turn_count']})")
        return result

    @router.get("/api/volition/status")
    async def api_volition_status():
        return VolitionGate.get_status()

    @router.post("/api/volition/reset")
    async def api_volition_reset():
        VolitionGate.reset()
        await emit_event("volition", "Session reset")
        return {"status": "ok"}

    @router.get("/api/volition/health")
    async def api_volition_health():
        return VolitionGate.get_health_status()

    return router


__all__ = ["create_router"]
