"""Event dispatcher — rate limiting, dedup, and EventBus dispatch.

Ported from Expergis. Instead of HTTP POST to Velle's sidecar,
events are published to Cathedral's EventBus for the pipeline to pick up.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from cathedral.shared.gate import GateLogger
from .plugins.base import Event

_log = GateLogger.get("PerceptionGate.Dispatcher")

_MAX_AUDIT_BYTES = 5 * 1024 * 1024  # 5 MB
_MAX_AUDIT_BACKUPS = 3


def _rotate_log(path: Path, max_bytes: int = _MAX_AUDIT_BYTES, backups: int = _MAX_AUDIT_BACKUPS) -> None:
    """Rotate a log file if it exceeds max_bytes. Keeps up to `backups` old copies."""
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        for i in range(backups - 1, 0, -1):
            src = path.with_suffix(f".{i}.jsonl")
            dst = path.with_suffix(f".{i + 1}.jsonl")
            if src.exists():
                src.replace(dst)
        path.replace(path.with_suffix(".1.jsonl"))
    except OSError:
        pass


class Dispatcher:
    """Receives events from plugins, applies rate limiting and dedup, publishes to EventBus."""

    def __init__(
        self,
        publish_fn: Callable[[str, dict], Awaitable[None]],
        min_interval_ms: int = 5000,
        max_events_per_minute: int = 6,
        burst_size: int = 2,
        dedup_window_ms: int = 10000,
        audit_path: str = "data/perception_audit.jsonl",
    ):
        self.publish_fn = publish_fn
        self.min_interval_ms = min_interval_ms
        self.max_events_per_minute = max_events_per_minute
        self.burst_size = burst_size
        self.dedup_window_ms = dedup_window_ms
        self.audit_path = Path(audit_path)

        # Dedup state
        self._dedup_cache: dict[str, float] = {}

        # Rate limit state (token bucket)
        self._tokens: float = float(burst_size)
        self._last_refill: float = time.monotonic()
        self._last_dispatch: float = 0.0

        # Ring buffer for perception_check
        self._event_buffer: deque[dict[str, Any]] = deque(maxlen=200)

        # Stats
        self._stats = {
            "total_events": 0,
            "dispatched": 0,
            "deduped": 0,
            "rate_limited": 0,
            "dispatch_errors": 0,
        }

    async def dispatch(self, event: Event, prompt_template: str) -> None:
        """Process an event: dedup, rate limit, format, and publish to EventBus."""
        self._stats["total_events"] += 1

        event_record = {
            "plugin_type": event.plugin_type,
            "watcher_id": event.watcher_id,
            "event_type": event.event_type,
            "summary": event.summary,
            "details": event.details,
            "timestamp": event.timestamp,
            "dispatched": False,
            "reason_skipped": None,
        }

        # Dedup check
        now = time.monotonic()
        self._cleanup_dedup(now)
        if event.dedup_key in self._dedup_cache:
            self._stats["deduped"] += 1
            event_record["reason_skipped"] = "dedup"
            self._event_buffer.append(event_record)
            return
        self._dedup_cache[event.dedup_key] = now

        # Rate limit check
        if not self._try_consume_token():
            self._stats["rate_limited"] += 1
            event_record["reason_skipped"] = "rate_limited"
            self._event_buffer.append(event_record)
            return

        # Min interval check
        if (now - self._last_dispatch) * 1000 < self.min_interval_ms:
            self._stats["rate_limited"] += 1
            event_record["reason_skipped"] = "min_interval"
            self._event_buffer.append(event_record)
            return

        # Format prompt
        prompt = prompt_template.format(event=event)

        # Publish to EventBus
        try:
            await self.publish_fn("perception:event", {
                "watcher_id": event.watcher_id,
                "event_type": event.event_type,
                "summary": event.summary,
                "prompt": prompt,
                "details": event.details,
            })
            self._stats["dispatched"] += 1
            self._last_dispatch = now
            event_record["dispatched"] = True
            _log.info(f"Dispatched: {event.summary}")
        except Exception as e:
            self._stats["dispatch_errors"] += 1
            event_record["reason_skipped"] = f"error:{type(e).__name__}"
            _log.warning(f"Dispatch error: {e}")

        self._event_buffer.append(event_record)

        # Audit
        self._audit({
            "action": "dispatch",
            "watcher_id": event.watcher_id,
            "event_type": event.event_type,
            "summary": event.summary,
            "dispatched": event_record["dispatched"],
            "reason_skipped": event_record["reason_skipped"],
        })

    def get_recent_events(
        self,
        since: str | None = None,
        watcher_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return recent events from the ring buffer."""
        events = list(self._event_buffer)
        if since:
            events = [e for e in events if e["timestamp"] >= since]
        if watcher_id:
            events = [e for e in events if e["watcher_id"] == watcher_id]
        return events[-limit:]

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def _cleanup_dedup(self, now: float) -> None:
        window_sec = self.dedup_window_ms / 1000.0
        expired = [k for k, t in self._dedup_cache.items() if now - t > window_sec]
        for k in expired:
            del self._dedup_cache[k]

    def _try_consume_token(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        refill_rate = self.max_events_per_minute / 60.0
        self._tokens = min(float(self.burst_size), self._tokens + elapsed * refill_rate)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _audit(self, entry: dict) -> None:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_log(self.audit_path)
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            _log.warning(f"Failed to write audit log: {e}")
