"""File/directory change detection via polling."""

import asyncio
import fnmatch
import os
from pathlib import Path
from typing import Any

from cathedral.shared.gate import GateLogger
from .base import EmitFn, Event, WatcherPlugin

_log = GateLogger.get("PerceptionGate.FileWatcher")


class FileWatcherPlugin(WatcherPlugin):
    """Watches files/directories for changes by polling modification times."""

    def __init__(self, watcher_id: str, config: dict[str, Any]):
        super().__init__(watcher_id, config)
        self.paths: list[Path] = []
        self.patterns: list[str] = []
        self.events: set[str] = set()
        self.debounce_ms: int = 1000
        self.poll_interval_ms: int = 2000
        self._snapshot: dict[str, float] = {}
        self._running = False

    async def setup(self) -> None:
        self.paths = [Path(p) for p in self.config.get("paths", [])]
        self.patterns = self.config.get("patterns", ["*"])
        self.events = set(self.config.get("events", ["modified", "created", "deleted"]))
        self.debounce_ms = self.config.get("debounce_ms", 1000)
        self.poll_interval_ms = self.config.get("poll_interval_ms", 2000)

        if not self.paths:
            raise ValueError(f"file_watcher '{self.watcher_id}': no paths configured")

        self._snapshot = self._scan()
        _log.info(
            f"FileWatcher '{self.watcher_id}' initialized: "
            f"{len(self._snapshot)} files across {len(self.paths)} paths"
        )

    async def watch(self, emit: EmitFn) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(self.poll_interval_ms / 1000.0)
            if not self._running:
                break

            current = self._scan()
            old_keys = set(self._snapshot.keys())
            new_keys = set(current.keys())

            if "created" in self.events:
                for path in new_keys - old_keys:
                    await emit(Event(
                        plugin_type="file_watcher",
                        watcher_id=self.watcher_id,
                        event_type="created",
                        summary=path,
                        details={"mtime": current[path]},
                        dedup_key=f"{self.watcher_id}:created:{path}",
                    ))

            if "deleted" in self.events:
                for path in old_keys - new_keys:
                    await emit(Event(
                        plugin_type="file_watcher",
                        watcher_id=self.watcher_id,
                        event_type="deleted",
                        summary=path,
                        details={},
                        dedup_key=f"{self.watcher_id}:deleted:{path}",
                    ))

            if "modified" in self.events:
                for path in old_keys & new_keys:
                    if current[path] != self._snapshot[path]:
                        await emit(Event(
                            plugin_type="file_watcher",
                            watcher_id=self.watcher_id,
                            event_type="modified",
                            summary=path,
                            details={
                                "old_mtime": self._snapshot[path],
                                "new_mtime": current[path],
                            },
                            dedup_key=f"{self.watcher_id}:modified:{path}",
                        ))

            self._snapshot = current

    async def teardown(self) -> None:
        self._running = False

    def _scan(self) -> dict[str, float]:
        """Scan configured paths and return {filepath: mtime}."""
        result: dict[str, float] = {}
        for base in self.paths:
            if not base.exists():
                continue
            if base.is_file():
                if self._matches(base.name):
                    try:
                        result[str(base)] = os.path.getmtime(base)
                    except OSError:
                        pass
            else:
                try:
                    for entry in os.scandir(base):
                        if entry.is_file() and self._matches(entry.name):
                            try:
                                result[entry.path] = entry.stat().st_mtime
                            except OSError:
                                pass
                except OSError:
                    pass
        return result

    def _matches(self, filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, p) for p in self.patterns)
