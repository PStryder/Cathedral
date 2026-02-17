"""
PerceptionGate - Event detection and agent awakening for Cathedral.

Detects file changes, cron schedules, and process events. When events
fire, publishes to Cathedral's EventBus which triggers VolitionGate
continuations.

Usage:
    from cathedral import PerceptionGate

    PerceptionGate.initialize(publish_fn=event_bus.publish)

    # Register a watcher
    await PerceptionGate.watch("src-watcher", "file_watcher", {
        "paths": ["/my/project/src"],
        "patterns": ["*.py"],
    })

    # List active watchers
    watchers = PerceptionGate.list_watchers()

    # Check recent events
    events = PerceptionGate.check_events(limit=10)

    # Remove a watcher
    await PerceptionGate.unwatch("src-watcher")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from cathedral.shared.gate import (
    GateLogger,
    build_health_status,
)

from .dispatcher import Dispatcher
from .plugins import PLUGIN_REGISTRY, Event, WatcherPlugin

_log = GateLogger.get("PerceptionGate")


@dataclass
class WatcherEntry:
    """Tracks a running watcher."""
    plugin: WatcherPlugin
    task: asyncio.Task
    prompt_template: str
    event_count: int = 0
    last_event: str | None = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# Module-level state
_watchers: Dict[str, WatcherEntry] = {}
_dispatcher: Optional[Dispatcher] = None
_initialized: bool = False
_publish_fn: Optional[Callable] = None
MAX_WATCHERS: int = 50

DEFAULT_PROMPT_TEMPLATE = "Event detected: {event.summary}. Review and decide if action needed."


def initialize(
    publish_fn: Callable[[str, dict], Awaitable[None]] | None = None,
    min_interval_ms: int = 5000,
    max_events_per_minute: int = 6,
    burst_size: int = 2,
    dedup_window_ms: int = 10000,
) -> bool:
    """Initialize PerceptionGate.

    Args:
        publish_fn: EventBus publish function for dispatching events
        min_interval_ms: Minimum gap between dispatches
        max_events_per_minute: Sustained rate limit
        burst_size: Max events in burst
        dedup_window_ms: Dedup TTL
    """
    global _dispatcher, _initialized, _publish_fn

    if _initialized:
        return True

    try:
        _publish_fn = publish_fn

        # Create a no-op publish if none provided
        async def _noop_publish(event_type: str, data: dict) -> None:
            _log.warning(f"No publish_fn configured, event dropped: {event_type}")

        _dispatcher = Dispatcher(
            publish_fn=publish_fn or _noop_publish,
            min_interval_ms=min_interval_ms,
            max_events_per_minute=max_events_per_minute,
            burst_size=burst_size,
            dedup_window_ms=dedup_window_ms,
        )

        _initialized = True
        _log.info("PerceptionGate initialized")
        return True

    except Exception as e:
        _log.error(f"PerceptionGate initialization failed: {e}")
        return False


def is_initialized() -> bool:
    return _initialized


def is_healthy() -> bool:
    return _initialized and _dispatcher is not None


def get_health_status() -> Dict[str, Any]:
    active = sum(1 for w in _watchers.values() if not w.task.done())
    return build_health_status(
        gate_name="PerceptionGate",
        initialized=_initialized,
        dependencies=[],
        checks={
            "dispatcher_ready": _dispatcher is not None,
        },
        details={
            "active_watchers": active,
            "total_watchers": len(_watchers),
            "stats": _dispatcher.stats if _dispatcher else {},
        },
    )


def get_dependencies() -> List[str]:
    return []


async def watch(
    watcher_id: str,
    plugin_type: str,
    config: Dict[str, Any],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Dict[str, Any]:
    """Register a new event watcher.

    Args:
        watcher_id: Unique ID for this watcher
        plugin_type: Plugin type (file_watcher, schedule_watcher, process_watcher)
        config: Plugin-specific configuration
        prompt_template: Template for formatting event prompts

    Returns:
        Dict with watcher info
    """
    if not _initialized:
        initialize()

    if watcher_id in _watchers:
        return {"status": "error", "error": f"Watcher '{watcher_id}' already exists"}

    active = sum(1 for w in _watchers.values() if not w.task.done())
    if active >= MAX_WATCHERS:
        return {"status": "error", "error": f"Watcher limit reached ({MAX_WATCHERS}). Remove a watcher first."}

    if plugin_type not in PLUGIN_REGISTRY:
        return {
            "status": "error",
            "error": f"Unknown plugin type: {plugin_type}. "
            f"Available: {', '.join(PLUGIN_REGISTRY.keys())}",
        }

    plugin_class = PLUGIN_REGISTRY[plugin_type]
    plugin = plugin_class(watcher_id, config)

    try:
        await plugin.setup()
    except ValueError as e:
        return {"status": "error", "error": str(e)}

    # Create emit callback that routes through dispatcher
    async def emit(event: Event) -> None:
        entry = _watchers.get(watcher_id)
        if entry:
            entry.event_count += 1
            entry.last_event = datetime.now(timezone.utc).isoformat()
        if _dispatcher:
            await _dispatcher.dispatch(event, prompt_template)

    # Start watcher as background task
    async def _run_watcher():
        try:
            await plugin.watch(emit)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _log.error(f"Watcher '{watcher_id}' error: {e}")
        finally:
            try:
                await plugin.teardown()
            except Exception as e:
                _log.warning(f"Watcher '{watcher_id}' teardown error: {e}")

    task = asyncio.create_task(_run_watcher())
    _watchers[watcher_id] = WatcherEntry(
        plugin=plugin,
        task=task,
        prompt_template=prompt_template,
    )

    _log.info(f"Watcher '{watcher_id}' ({plugin_type}) started")
    return {
        "status": "ok",
        "watcher_id": watcher_id,
        "plugin_type": plugin_type,
    }


async def unwatch(watcher_id: str) -> Dict[str, Any]:
    """Remove an active watcher.

    Args:
        watcher_id: ID of watcher to remove

    Returns:
        Dict with status
    """
    if watcher_id not in _watchers:
        return {"status": "error", "error": f"Watcher '{watcher_id}' not found"}

    entry = _watchers[watcher_id]
    entry.task.cancel()
    try:
        await entry.task
    except asyncio.CancelledError:
        pass

    del _watchers[watcher_id]
    _log.info(f"Watcher '{watcher_id}' removed")
    return {"status": "ok", "watcher_id": watcher_id}


def list_watchers() -> List[Dict[str, Any]]:
    """List all active watchers with stats."""
    result = []
    for wid, entry in _watchers.items():
        result.append({
            "watcher_id": wid,
            "plugin_type": entry.plugin.config.get("_plugin_type", type(entry.plugin).__name__),
            "running": not entry.task.done(),
            "event_count": entry.event_count,
            "last_event": entry.last_event,
            "created_at": entry.created_at,
        })
    return result


def check_events(
    since: str | None = None,
    watcher_id: str | None = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Poll recent events from the ring buffer.

    Args:
        since: ISO timestamp filter
        watcher_id: Filter by watcher
        limit: Max events to return
    """
    if _dispatcher is None:
        return []
    return _dispatcher.get_recent_events(since, watcher_id, limit)


async def shutdown() -> None:
    """Cancel all watchers and clean up."""
    for wid in list(_watchers.keys()):
        await unwatch(wid)
    _log.info("PerceptionGate shut down")


async def _reset() -> None:
    """Reset module state (for testing only)."""
    global _initialized, _dispatcher, _publish_fn
    await shutdown()
    _dispatcher = None
    _publish_fn = None
    _initialized = False


def get_info() -> dict:
    """Get comprehensive documentation for PerceptionGate."""
    return {
        "gate": "PerceptionGate",
        "version": "1.0",
        "purpose": "Detect events in the environment and wake the agent to respond. "
        "Publishes events to Cathedral's EventBus, triggering VolitionGate continuations.",
        "plugin_types": {
            "file_watcher": {
                "purpose": "Detect file/directory changes by polling",
                "config": {
                    "paths": "list of paths to watch",
                    "patterns": "fnmatch patterns (default: ['*'])",
                    "events": "subset of ['modified', 'created', 'deleted']",
                    "poll_interval_ms": "polling interval (default: 2000)",
                    "debounce_ms": "debounce window (default: 1000)",
                },
            },
            "schedule_watcher": {
                "purpose": "Fire events on a cron schedule",
                "config": {
                    "cron": "5-field cron expression (e.g., '*/5 * * * *')",
                },
            },
            "process_watcher": {
                "purpose": "Detect process start/stop (Windows)",
                "config": {
                    "process_names": "list of process names to watch",
                    "poll_interval_ms": "polling interval (default: 5000)",
                },
            },
        },
        "tools": {
            "watch": {"purpose": "Register a new watcher"},
            "unwatch": {"purpose": "Remove a watcher"},
            "list": {"purpose": "List active watchers"},
            "check": {"purpose": "Poll recent events"},
        },
        "rate_limiting": {
            "min_interval_ms": "Minimum gap between dispatches",
            "max_events_per_minute": "Sustained rate limit",
            "burst_size": "Max events in burst",
            "dedup_window_ms": "Dedup TTL",
        },
    }


__all__ = [
    "initialize",
    "is_initialized",
    "is_healthy",
    "get_health_status",
    "get_dependencies",
    "watch",
    "unwatch",
    "list_watchers",
    "check_events",
    "shutdown",
    "get_info",
]
