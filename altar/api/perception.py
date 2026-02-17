from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class WatchRequest(BaseModel):
    watcher_id: str
    plugin_type: str
    config: dict[str, Any]
    prompt_template: str = "Event detected: {event.summary}. Review and decide if action needed."


def create_router(PerceptionGate, emit_event) -> APIRouter:
    router = APIRouter()

    @router.post("/api/perception/watch")
    async def api_perception_watch(data: WatchRequest):
        result = await PerceptionGate.watch(
            data.watcher_id, data.plugin_type, data.config, data.prompt_template
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))
        await emit_event("perception", f"Watcher registered: {data.watcher_id}")
        return result

    @router.post("/api/perception/unwatch/{watcher_id}")
    async def api_perception_unwatch(watcher_id: str):
        result = await PerceptionGate.unwatch(watcher_id)
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result.get("error"))
        await emit_event("perception", f"Watcher removed: {watcher_id}")
        return result

    @router.get("/api/perception/watchers")
    async def api_perception_list():
        return {"watchers": PerceptionGate.list_watchers()}

    @router.get("/api/perception/events")
    async def api_perception_events(
        since: str | None = None,
        watcher_id: str | None = None,
        limit: int = 20,
    ):
        return {"events": PerceptionGate.check_events(since, watcher_id, limit)}

    @router.get("/api/perception/health")
    async def api_perception_health():
        return PerceptionGate.get_health_status()

    return router


__all__ = ["create_router"]
