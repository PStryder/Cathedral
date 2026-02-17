"""Perception plugins registry."""

from .base import Event, EmitFn, WatcherPlugin
from .file_watcher import FileWatcherPlugin
from .schedule_watcher import ScheduleWatcherPlugin
from .process_watcher import ProcessWatcherPlugin

PLUGIN_REGISTRY = {
    "file_watcher": FileWatcherPlugin,
    "schedule_watcher": ScheduleWatcherPlugin,
    "process_watcher": ProcessWatcherPlugin,
}

__all__ = [
    "Event",
    "EmitFn",
    "WatcherPlugin",
    "FileWatcherPlugin",
    "ScheduleWatcherPlugin",
    "ProcessWatcherPlugin",
    "PLUGIN_REGISTRY",
]
