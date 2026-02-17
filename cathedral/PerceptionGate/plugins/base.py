"""Base plugin contract and Event dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine


@dataclass
class Event:
    """An event emitted by a watcher plugin."""

    plugin_type: str
    watcher_id: str
    event_type: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    dedup_key: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.dedup_key:
            self.dedup_key = f"{self.watcher_id}:{self.event_type}:{self.summary}"


# Type alias for the emit callback
EmitFn = Callable[[Event], Coroutine[Any, Any, None]]


class WatcherPlugin(ABC):
    """Abstract base class for watcher plugins."""

    def __init__(self, watcher_id: str, config: dict[str, Any]):
        self.watcher_id = watcher_id
        self.config = config

    @abstractmethod
    async def setup(self) -> None:
        """Validate config and initialize state."""

    @abstractmethod
    async def watch(self, emit: EmitFn) -> None:
        """Long-running loop. Call emit(Event) when events fire."""

    @abstractmethod
    async def teardown(self) -> None:
        """Cleanup on removal or shutdown."""
