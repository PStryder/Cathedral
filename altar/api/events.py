from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse


def create_router(event_bus, agent_tracker) -> APIRouter:
    router = APIRouter()

    @router.get("/api/events")
    async def api_events():
        """SSE endpoint for real-time system events."""

        async def generate():
            queue = await event_bus.subscribe()
            try:
                # Send initial connection event
                yield {
                    "event": "system",
                    "data": json.dumps({"message": "Connected to event stream"}),
                }

                while True:
                    try:
                        # Wait for events with timeout for keepalive
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield {
                            "event": event["type"],
                            "data": json.dumps(event["data"]),
                        }
                    except asyncio.TimeoutError:
                        # Send keepalive ping
                        yield {"event": "ping", "data": "{}"}

            except asyncio.CancelledError:
                pass
            finally:
                await event_bus.unsubscribe(queue)

        return EventSourceResponse(generate())

    @router.get("/api/agents/status")
    async def api_agents_status(since: float = 0):
        """Get agent status updates since timestamp (for polling fallback)."""
        updates = await agent_tracker.get_updates_since(since)
        return {"updates": updates, "timestamp": time.time()}

    return router


__all__ = ["create_router"]
