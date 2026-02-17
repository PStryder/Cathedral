"""Tests for PerceptionGate event detection."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from cathedral.PerceptionGate import (
    initialize,
    is_initialized,
    watch,
    unwatch,
    list_watchers,
    check_events,
    get_health_status,
    get_info,
)
from cathedral.PerceptionGate.plugins.base import Event
from cathedral.PerceptionGate.dispatcher import Dispatcher


class TestPerceptionGateInitialization:

    def test_initialize(self):
        mock_publish = AsyncMock()
        result = initialize(publish_fn=mock_publish)
        assert result is True
        assert is_initialized() is True

    def test_health_status(self):
        mock_publish = AsyncMock()
        initialize(publish_fn=mock_publish)
        status = get_health_status()
        assert status["gate"] == "PerceptionGate"
        assert status["initialized"] is True

    def test_get_info(self):
        info = get_info()
        assert info["gate"] == "PerceptionGate"
        assert "file_watcher" in info["plugin_types"]


class TestDispatcher:
    """Tests for the event dispatcher."""

    @pytest.mark.asyncio
    async def test_dispatch_publishes_event(self):
        mock_publish = AsyncMock()
        dispatcher = Dispatcher(
            publish_fn=mock_publish,
            min_interval_ms=0,
            dedup_window_ms=100,
        )

        event = Event(
            plugin_type="file_watcher",
            watcher_id="test",
            event_type="modified",
            summary="test.py changed",
        )
        await dispatcher.dispatch(event, "File changed: {event.summary}")

        mock_publish.assert_called_once()
        call_args = mock_publish.call_args
        assert call_args[0][0] == "perception:event"
        assert "test.py changed" in call_args[0][1]["summary"]

    @pytest.mark.asyncio
    async def test_dedup(self):
        mock_publish = AsyncMock()
        dispatcher = Dispatcher(
            publish_fn=mock_publish,
            min_interval_ms=0,
            dedup_window_ms=60000,
        )

        event = Event(
            plugin_type="file_watcher",
            watcher_id="test",
            event_type="modified",
            summary="test.py",
        )

        await dispatcher.dispatch(event, "{event.summary}")
        await dispatcher.dispatch(event, "{event.summary}")

        assert mock_publish.call_count == 1
        assert dispatcher.stats["deduped"] == 1

    @pytest.mark.asyncio
    async def test_rate_limit(self):
        mock_publish = AsyncMock()
        dispatcher = Dispatcher(
            publish_fn=mock_publish,
            min_interval_ms=0,
            max_events_per_minute=60,
            burst_size=1,
            dedup_window_ms=0,
        )

        for i in range(5):
            event = Event(
                plugin_type="file_watcher",
                watcher_id="test",
                event_type="modified",
                summary=f"file_{i}.py",
            )
            await dispatcher.dispatch(event, "{event.summary}")

        assert dispatcher.stats["dispatched"] <= 2
        assert dispatcher.stats["rate_limited"] >= 3

    @pytest.mark.asyncio
    async def test_recent_events(self):
        mock_publish = AsyncMock()
        dispatcher = Dispatcher(
            publish_fn=mock_publish,
            min_interval_ms=0,
            dedup_window_ms=0,
        )

        event = Event(
            plugin_type="file_watcher",
            watcher_id="w1",
            event_type="modified",
            summary="hello.py",
        )
        await dispatcher.dispatch(event, "{event.summary}")

        recent = dispatcher.get_recent_events()
        assert len(recent) == 1
        assert recent[0]["summary"] == "hello.py"


class TestEventDataclass:

    def test_auto_dedup_key(self):
        event = Event(
            plugin_type="file_watcher",
            watcher_id="watch1",
            event_type="modified",
            summary="foo.py",
        )
        assert event.dedup_key == "watch1:modified:foo.py"

    def test_custom_dedup_key(self):
        event = Event(
            plugin_type="file_watcher",
            watcher_id="watch1",
            event_type="modified",
            summary="foo.py",
            dedup_key="custom_key",
        )
        assert event.dedup_key == "custom_key"

    def test_timestamp_auto_set(self):
        event = Event(
            plugin_type="file_watcher",
            watcher_id="watch1",
            event_type="modified",
            summary="foo.py",
        )
        assert event.timestamp is not None
        assert "T" in event.timestamp  # ISO format


class TestWatcherManagement:

    @pytest.mark.asyncio
    async def test_watch_unknown_plugin(self):
        mock_publish = AsyncMock()
        initialize(publish_fn=mock_publish)
        result = await watch("test", "nonexistent_plugin", {})
        assert result["status"] == "error"
        assert "Unknown plugin type" in result["error"]

    @pytest.mark.asyncio
    async def test_unwatch_nonexistent(self):
        mock_publish = AsyncMock()
        initialize(publish_fn=mock_publish)
        result = await unwatch("nonexistent")
        assert result["status"] == "error"


class TestDispatcherAuditRotation:
    """Tests for dispatcher audit log rotation."""

    def test_rotate_log_creates_backup(self, tmp_path):
        from cathedral.PerceptionGate.dispatcher import _rotate_log

        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("x" * 200)

        _rotate_log(log_file, max_bytes=100, backups=2)

        assert not log_file.exists()
        assert (tmp_path / "audit.1.jsonl").exists()
        assert (tmp_path / "audit.1.jsonl").read_text() == "x" * 200

    def test_rotate_log_no_op_under_threshold(self, tmp_path):
        from cathedral.PerceptionGate.dispatcher import _rotate_log

        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("small")

        _rotate_log(log_file, max_bytes=1000, backups=2)

        assert log_file.exists()
        assert log_file.read_text() == "small"

    @pytest.mark.asyncio
    async def test_dispatcher_audit_writes_file(self, tmp_path):
        import json
        audit_file = tmp_path / "test_dispatch_audit.jsonl"
        mock_publish = AsyncMock()
        dispatcher = Dispatcher(
            publish_fn=mock_publish,
            min_interval_ms=0,
            dedup_window_ms=0,
            audit_path=str(audit_file),
        )

        event = Event(
            plugin_type="file_watcher",
            watcher_id="test",
            event_type="modified",
            summary="foo.py",
        )
        await dispatcher.dispatch(event, "{event.summary}")

        assert audit_file.exists()
        line = json.loads(audit_file.read_text().strip().split("\n")[-1])
        assert line["action"] == "dispatch"
        assert line["summary"] == "foo.py"


class TestProcessWatcherAsync:
    """Tests for async process scanning."""

    @pytest.mark.asyncio
    async def test_scan_processes_async_returns_dict(self):
        from cathedral.PerceptionGate.plugins.process_watcher import ProcessWatcherPlugin

        plugin = ProcessWatcherPlugin("test-proc", {
            "process_names": ["nonexistent_process_xyz"],
        })
        await plugin.setup()

        result = await plugin._scan_processes_async()
        assert isinstance(result, dict)
        assert "nonexistent_process_xyz" in result
        assert len(result["nonexistent_process_xyz"]) == 0
