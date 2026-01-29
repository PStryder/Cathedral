from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Set, Deque, Dict, Any, Callable, Awaitable


class EventBus:
    """Simple pub/sub event bus for system events."""

    def __init__(self, max_history: int = 100):
        self._subscribers: Set[asyncio.Queue] = set()
        self._history: Deque[dict] = deque(maxlen=max_history)
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to events, returns a queue for receiving."""
        queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from events."""
        async with self._lock:
            self._subscribers.discard(queue)

    async def publish(self, event_type: str, data: dict) -> None:
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        self._history.append(event)

        async with self._lock:
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # Skip if queue is full

    def get_recent(self, count: int = 20) -> list:
        """Get recent events from history."""
        return list(self._history)[-count:]


def build_emitter(event_bus: EventBus) -> Callable[..., Awaitable[None]]:
    async def emit_event(event_type: str, message: str, **kwargs) -> None:
        data = {"message": message, **kwargs}
        await event_bus.publish(event_type, data)

    return emit_event


__all__ = ["EventBus", "build_emitter"]
