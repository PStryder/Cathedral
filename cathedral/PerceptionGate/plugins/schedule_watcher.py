"""Cron-like time trigger watcher using croniter."""

import asyncio
from datetime import datetime, timezone
from typing import Any

from cathedral.shared.gate import GateLogger
from .base import EmitFn, Event, WatcherPlugin

_log = GateLogger.get("PerceptionGate.ScheduleWatcher")


class ScheduleWatcherPlugin(WatcherPlugin):
    """Fires events on a cron schedule."""

    def __init__(self, watcher_id: str, config: dict[str, Any]):
        super().__init__(watcher_id, config)
        self.cron_expr: str = ""
        self._cron = None
        self._running = False

    async def setup(self) -> None:
        from croniter import croniter

        self.cron_expr = self.config.get("cron", "")
        if not self.cron_expr:
            raise ValueError(f"schedule_watcher '{self.watcher_id}': no cron expression configured")

        if not croniter.is_valid(self.cron_expr):
            raise ValueError(
                f"schedule_watcher '{self.watcher_id}': invalid cron expression: {self.cron_expr}"
            )

        self._cron = croniter(self.cron_expr, datetime.now(timezone.utc))
        _log.info(f"ScheduleWatcher '{self.watcher_id}' initialized: {self.cron_expr}")

    async def watch(self, emit: EmitFn) -> None:
        from croniter import croniter

        self._running = True
        while self._running:
            if self._cron is None:
                break

            next_fire = self._cron.get_next(datetime)
            now = datetime.now(timezone.utc)

            if next_fire.tzinfo is None:
                next_fire = next_fire.replace(tzinfo=timezone.utc)

            wait_seconds = (next_fire - now).total_seconds()
            if wait_seconds > 0:
                try:
                    await asyncio.sleep(wait_seconds)
                except asyncio.CancelledError:
                    break

            if not self._running:
                break

            await emit(Event(
                plugin_type="schedule_watcher",
                watcher_id=self.watcher_id,
                event_type="scheduled",
                summary=f"Cron fired: {self.cron_expr}",
                details={"cron": self.cron_expr, "fire_time": datetime.now(timezone.utc).isoformat()},
                dedup_key=f"{self.watcher_id}:scheduled:{next_fire.isoformat()}",
            ))

    async def teardown(self) -> None:
        self._running = False
