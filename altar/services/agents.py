from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Dict, List

from altar.services.events import EventBus


class AgentTracker:
    """Track agent status updates and emit events."""

    def __init__(self, event_bus: EventBus, max_history: int = 50):
        self._event_bus = event_bus
        self._updates: Deque[dict] = deque(maxlen=max_history)
        self._lock = asyncio.Lock()

    async def record_update(self, agent_id: str, message: str, status: str = "running") -> None:
        """Record an agent status update and emit an event."""
        update = {
            "id": agent_id,
            "message": message,
            "status": status,
            "timestamp": time.time(),
        }
        async with self._lock:
            self._updates.append(update)
        await self._event_bus.publish("agent", {"message": message, "id": agent_id, "status": status})

    async def get_updates_since(self, since: float = 0) -> List[Dict]:
        async with self._lock:
            return [u for u in self._updates if u["timestamp"] > since]


__all__ = ["AgentTracker"]
