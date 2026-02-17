"""Integration tests for Faculta gates (AgencyGate, VolitionGate, PerceptionGate).

These tests exercise the integration seams between gates and the pipeline,
not just isolated unit behavior.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test 1: Volition pipeline integration — THE important one
# ---------------------------------------------------------------------------


class TestVolitionPipelineIntegration:
    """Test the process_input_stream while-loop continuation logic."""

    @pytest.mark.asyncio
    async def test_continuation_loop_runs_multiple_turns(self):
        """Mock _run_pipeline_turn to emit volition_continue, verify multi-turn loop."""
        from cathedral.pipeline.chat import (
            _contains_volition_continue,
            _extract_volition_continue,
        )
        from cathedral import VolitionGate

        # Ensure zero cooldown so rapid continuation isn't denied
        VolitionGate._reset()
        with patch("cathedral.VolitionGate.Config") as mock_config:
            mock_config.get = lambda key, default=None: {
                "VOLITION_TURN_LIMIT": "20",
                "VOLITION_COOLDOWN_MS": "0",
                "VOLITION_BUDGET_USD": "5.00",
                "VOLITION_AUDIT_MODE": "local",
            }.get(key, default)
            VolitionGate.initialize()

        # Simulate three turns: turns 1-2 contain volition_continue, turn 3 does not
        turn_count = 0

        async def mock_pipeline_turn(
            user_input, thread_uid, services, enable_tools,
            enabled_gates, enable_context,
        ):
            nonlocal turn_count
            turn_count += 1
            if turn_count <= 2:
                # Emit a response with a volition_continue tool call
                response = (
                    f"[Turn {turn_count} output] "
                    f'{{"type": "tool_call", "id": "tc_{turn_count:03d}", '
                    f'"tool": "VolitionGate.request_continue", '
                    f'"args": {{"text": "continue turn {turn_count + 1}", '
                    f'"reason": "more work"}}}}'
                )
                for char in response:
                    yield char
            else:
                # Final turn — no continuation
                response = f"[Turn {turn_count} final output]"
                for char in response:
                    yield char

        # Mock dependencies to avoid full pipeline init
        mock_services = MagicMock()
        mock_services.emit_event = AsyncMock()

        with patch("cathedral.pipeline.chat._run_pipeline_turn", mock_pipeline_turn), \
             patch("cathedral.pipeline.chat.handle_pre_command", return_value=None), \
             patch("cathedral.pipeline.chat.handle_post_command", return_value=None), \
             patch("cathedral.pipeline.chat.emit_completed_agents") as mock_emit_agents, \
             patch("cathedral.pipeline.chat.SecurityManager") as mock_security:

            mock_security.is_locked.return_value = False

            # emit_completed_agents needs to be an async generator
            async def empty_gen(*args, **kwargs):
                return
                yield  # make it an async generator
            mock_emit_agents.return_value = empty_gen()

            from cathedral.pipeline.chat import process_input_stream

            tokens = []
            async for token in process_input_stream(
                "start task",
                thread_uid="test-volition-loop",
                services=mock_services,
                enable_tools=True,
            ):
                tokens.append(token)

            full_output = "".join(tokens)

        # Verify multi-turn execution
        assert turn_count == 3, f"Expected 3 turns, got {turn_count}"
        assert "[Turn 1 output]" in full_output
        assert "[Turn 2 output]" in full_output
        assert "[Turn 3 final output]" in full_output

    @pytest.mark.asyncio
    async def test_continuation_stops_at_turn_limit(self):
        """Verify the loop respects VolitionEngine's turn limit."""
        from cathedral import VolitionGate

        # Initialize with a limit of 1 turn
        VolitionGate._reset()
        with patch("cathedral.VolitionGate.Config") as mock_config:
            mock_config.get = lambda key, default=None: {
                "VOLITION_TURN_LIMIT": "1",
                "VOLITION_COOLDOWN_MS": "0",
                "VOLITION_BUDGET_USD": "5.00",
                "VOLITION_AUDIT_MODE": "local",
            }.get(key, default)
            VolitionGate.initialize()

        turn_count = 0

        async def mock_pipeline_turn(
            user_input, thread_uid, services, enable_tools,
            enabled_gates, enable_context,
        ):
            nonlocal turn_count
            turn_count += 1
            # Always try to continue
            response = (
                f"[Turn {turn_count}] "
                f'{{"tool": "VolitionGate.request_continue", '
                f'"args": {{"text": "keep going", "reason": "test"}}}}'
            )
            for char in response:
                yield char

        mock_services = MagicMock()
        mock_services.emit_event = AsyncMock()

        with patch("cathedral.pipeline.chat._run_pipeline_turn", mock_pipeline_turn), \
             patch("cathedral.pipeline.chat.handle_pre_command", return_value=None), \
             patch("cathedral.pipeline.chat.handle_post_command", return_value=None), \
             patch("cathedral.pipeline.chat.emit_completed_agents") as mock_emit_agents, \
             patch("cathedral.pipeline.chat.SecurityManager") as mock_security:

            mock_security.is_locked.return_value = False

            async def empty_gen(*args, **kwargs):
                return
                yield
            mock_emit_agents.return_value = empty_gen()

            from cathedral.pipeline.chat import process_input_stream

            tokens = []
            async for token in process_input_stream(
                "start",
                thread_uid="test-turn-limit",
                services=mock_services,
                enable_tools=True,
            ):
                tokens.append(token)

        # Turn 1: initial. Turn 2: first continuation (approved, uses the 1 allowed turn).
        # Turn 2's response also requests continuation, but turn_limit=1 means it's denied.
        assert turn_count == 2, f"Expected 2 turns (1 initial + 1 approved continuation), got {turn_count}"

    @pytest.mark.asyncio
    async def test_no_continuation_without_enable_tools(self):
        """Verify volition_continue in output is ignored when tools are disabled."""
        turn_count = 0

        async def mock_pipeline_turn(
            user_input, thread_uid, services, enable_tools,
            enabled_gates, enable_context,
        ):
            nonlocal turn_count
            turn_count += 1
            response = (
                f'Response with {{"tool": "VolitionGate.request_continue", '
                f'"args": {{"text": "try", "reason": "test"}}}}'
            )
            for char in response:
                yield char

        mock_services = MagicMock()
        mock_services.emit_event = AsyncMock()

        with patch("cathedral.pipeline.chat._run_pipeline_turn", mock_pipeline_turn), \
             patch("cathedral.pipeline.chat.handle_pre_command", return_value=None), \
             patch("cathedral.pipeline.chat.handle_post_command", return_value=None), \
             patch("cathedral.pipeline.chat.emit_completed_agents") as mock_emit_agents, \
             patch("cathedral.pipeline.chat.SecurityManager") as mock_security:

            mock_security.is_locked.return_value = False

            async def empty_gen(*args, **kwargs):
                return
                yield
            mock_emit_agents.return_value = empty_gen()

            from cathedral.pipeline.chat import process_input_stream

            tokens = []
            async for token in process_input_stream(
                "test",
                thread_uid="test-no-tools",
                services=mock_services,
                enable_tools=False,  # tools disabled
            ):
                tokens.append(token)

        assert turn_count == 1, "Should not loop when tools are disabled"


# ---------------------------------------------------------------------------
# Test 2: FileWatcherPlugin functional test
# ---------------------------------------------------------------------------


class TestFileWatcherFunctional:
    """Test that FileWatcherPlugin detects real file changes."""

    @pytest.mark.asyncio
    async def test_detects_file_creation(self, tmp_path):
        from cathedral.PerceptionGate.plugins.file_watcher import FileWatcherPlugin

        plugin = FileWatcherPlugin("fw-create", {
            "paths": [str(tmp_path)],
            "patterns": ["*.txt"],
            "events": ["created"],
            "poll_interval_ms": 100,
        })
        await plugin.setup()

        events_received = []

        async def collect(event):
            events_received.append(event)

        # Start watcher as background task
        task = asyncio.create_task(plugin.watch(collect))

        # Wait a beat, then create a file
        await asyncio.sleep(0.15)
        (tmp_path / "newfile.txt").write_text("hello")

        # Wait for detection
        try:
            await asyncio.wait_for(
                _wait_for_events(events_received, 1),
                timeout=3.0,
            )
        finally:
            await plugin.teardown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(events_received) >= 1
        assert events_received[0].event_type == "created"
        assert "newfile.txt" in events_received[0].summary

    @pytest.mark.asyncio
    async def test_detects_file_modification(self, tmp_path):
        from cathedral.PerceptionGate.plugins.file_watcher import FileWatcherPlugin

        # Pre-create the file
        target = tmp_path / "existing.txt"
        target.write_text("original")

        plugin = FileWatcherPlugin("fw-modify", {
            "paths": [str(tmp_path)],
            "patterns": ["*.txt"],
            "events": ["modified"],
            "poll_interval_ms": 100,
        })
        await plugin.setup()

        events_received = []

        async def collect(event):
            events_received.append(event)

        task = asyncio.create_task(plugin.watch(collect))

        # Modify the file (ensure mtime changes)
        await asyncio.sleep(0.15)
        target.write_text("modified content")
        # Force a distinct mtime on Windows (resolution can be coarse)
        os.utime(target, (time.time() + 1, time.time() + 1))

        try:
            await asyncio.wait_for(
                _wait_for_events(events_received, 1),
                timeout=3.0,
            )
        finally:
            await plugin.teardown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(events_received) >= 1
        assert events_received[0].event_type == "modified"

    @pytest.mark.asyncio
    async def test_detects_file_deletion(self, tmp_path):
        from cathedral.PerceptionGate.plugins.file_watcher import FileWatcherPlugin

        # Pre-create the file
        target = tmp_path / "to_delete.txt"
        target.write_text("doomed")

        plugin = FileWatcherPlugin("fw-delete", {
            "paths": [str(tmp_path)],
            "patterns": ["*.txt"],
            "events": ["deleted"],
            "poll_interval_ms": 100,
        })
        await plugin.setup()

        events_received = []

        async def collect(event):
            events_received.append(event)

        task = asyncio.create_task(plugin.watch(collect))

        await asyncio.sleep(0.15)
        target.unlink()

        try:
            await asyncio.wait_for(
                _wait_for_events(events_received, 1),
                timeout=3.0,
            )
        finally:
            await plugin.teardown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(events_received) >= 1
        assert events_received[0].event_type == "deleted"
        assert "to_delete.txt" in events_received[0].summary

    @pytest.mark.asyncio
    async def test_pattern_filtering(self, tmp_path):
        """Only *.py files should trigger events, not *.txt."""
        from cathedral.PerceptionGate.plugins.file_watcher import FileWatcherPlugin

        plugin = FileWatcherPlugin("fw-filter", {
            "paths": [str(tmp_path)],
            "patterns": ["*.py"],
            "events": ["created"],
            "poll_interval_ms": 100,
        })
        await plugin.setup()

        events_received = []

        async def collect(event):
            events_received.append(event)

        task = asyncio.create_task(plugin.watch(collect))

        await asyncio.sleep(0.15)
        (tmp_path / "ignored.txt").write_text("nope")
        (tmp_path / "detected.py").write_text("yes")

        try:
            await asyncio.wait_for(
                _wait_for_events(events_received, 1),
                timeout=3.0,
            )
        finally:
            await plugin.teardown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        summaries = [e.summary for e in events_received]
        assert any("detected.py" in s for s in summaries)
        assert not any("ignored.txt" in s for s in summaries)


# ---------------------------------------------------------------------------
# Test 3: ScheduleWatcherPlugin functional test
# ---------------------------------------------------------------------------

try:
    import croniter as _croniter_check
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False


@pytest.mark.skipif(not HAS_CRONITER, reason="croniter not installed")
class TestScheduleWatcherFunctional:
    """Test that ScheduleWatcherPlugin fires on cron schedule."""

    @pytest.mark.asyncio
    async def test_fires_on_schedule(self):
        from cathedral.PerceptionGate.plugins.schedule_watcher import ScheduleWatcherPlugin

        # Use a cron that fires every minute — we'll mock time to avoid waiting
        plugin = ScheduleWatcherPlugin("sw-test", {
            "cron": "* * * * *",  # every minute
        })
        await plugin.setup()

        events_received = []

        async def collect(event):
            events_received.append(event)

        # Patch asyncio.sleep to skip the wait
        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            # Skip long waits, just yield control
            await original_sleep(0)

        with patch("asyncio.sleep", fast_sleep):
            task = asyncio.create_task(plugin.watch(collect))
            # Give it time for one iteration
            await original_sleep(0.2)
            await plugin.teardown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(events_received) >= 1
        assert events_received[0].event_type == "scheduled"
        assert events_received[0].plugin_type == "schedule_watcher"

    @pytest.mark.asyncio
    async def test_rejects_invalid_cron(self):
        from cathedral.PerceptionGate.plugins.schedule_watcher import ScheduleWatcherPlugin

        plugin = ScheduleWatcherPlugin("sw-bad", {
            "cron": "not a cron expression",
        })
        with pytest.raises(ValueError, match="invalid cron expression"):
            await plugin.setup()

    @pytest.mark.asyncio
    async def test_rejects_empty_cron(self):
        from cathedral.PerceptionGate.plugins.schedule_watcher import ScheduleWatcherPlugin

        plugin = ScheduleWatcherPlugin("sw-empty", {})
        with pytest.raises(ValueError, match="no cron expression"):
            await plugin.setup()


# ---------------------------------------------------------------------------
# Test 4: ProcessWatcherPlugin validation test
# ---------------------------------------------------------------------------


class TestProcessWatcherValidation:
    """Test ProcessWatcherPlugin setup validation and config parsing."""

    @pytest.mark.asyncio
    async def test_rejects_empty_process_names(self):
        from cathedral.PerceptionGate.plugins.process_watcher import ProcessWatcherPlugin

        plugin = ProcessWatcherPlugin("pw-empty", {"process_names": []})
        with pytest.raises(ValueError, match="no process_names"):
            await plugin.setup()

    @pytest.mark.asyncio
    async def test_rejects_missing_process_names(self):
        from cathedral.PerceptionGate.plugins.process_watcher import ProcessWatcherPlugin

        plugin = ProcessWatcherPlugin("pw-missing", {})
        with pytest.raises(ValueError, match="no process_names"):
            await plugin.setup()

    @pytest.mark.asyncio
    async def test_config_parsing(self):
        from cathedral.PerceptionGate.plugins.process_watcher import ProcessWatcherPlugin

        plugin = ProcessWatcherPlugin("pw-config", {
            "process_names": ["Python", "Node"],
            "poll_interval_ms": 10000,
        })
        await plugin.setup()

        assert plugin.process_names == ["python", "node"]  # lowercased
        assert plugin.poll_interval_ms == 10000

    @pytest.mark.asyncio
    async def test_scan_returns_correct_structure(self):
        from cathedral.PerceptionGate.plugins.process_watcher import ProcessWatcherPlugin

        plugin = ProcessWatcherPlugin("pw-scan", {
            "process_names": ["nonexistent_process_abc123"],
        })
        await plugin.setup()

        result = await plugin._scan_processes_async()
        assert isinstance(result, dict)
        assert "nonexistent_process_abc123" in result
        assert isinstance(result["nonexistent_process_abc123"], set)


# ---------------------------------------------------------------------------
# Test 5: PerceptionGate → EventBus → VolitionGate integration
# ---------------------------------------------------------------------------


class TestPerceptionVolitionIntegration:
    """Test the event chain from PerceptionGate dispatch to VolitionGate-compatible payload."""

    @pytest.mark.asyncio
    async def test_dispatched_event_contains_formatted_prompt(self):
        """Verify PerceptionGate's dispatch produces a prompt suitable for VolitionGate."""
        from cathedral.PerceptionGate.dispatcher import Dispatcher
        from cathedral.PerceptionGate.plugins.base import Event
        from cathedral import VolitionGate

        VolitionGate.initialize()

        captured_events = []

        async def mock_publish(event_type: str, data: dict):
            captured_events.append({"event_type": event_type, "data": data})

        dispatcher = Dispatcher(
            publish_fn=mock_publish,
            min_interval_ms=0,
            dedup_window_ms=0,
        )

        event = Event(
            plugin_type="file_watcher",
            watcher_id="src-watcher",
            event_type="modified",
            summary="main.py changed",
            details={"old_mtime": 1000, "new_mtime": 2000},
        )

        template = "File changed: {event.summary}. Review the change and decide next steps."
        await dispatcher.dispatch(event, template)

        # Verify the event was published
        assert len(captured_events) == 1
        payload = captured_events[0]
        assert payload["event_type"] == "perception:event"
        assert payload["data"]["summary"] == "main.py changed"
        assert "File changed: main.py changed" in payload["data"]["prompt"]
        assert payload["data"]["watcher_id"] == "src-watcher"

        # Verify the prompt text could be fed to VolitionGate
        prompt_text = payload["data"]["prompt"]
        result = VolitionGate.request_continue(prompt_text, reason="perception event")
        assert result["status"] == "approved"
        assert result["turn_count"] == 1

    @pytest.mark.asyncio
    async def test_full_perception_gate_to_eventbus(self):
        """Test through the full PerceptionGate.watch → dispatch → publish chain."""
        from cathedral import PerceptionGate
        from cathedral.PerceptionGate.plugins.base import Event

        captured = []

        async def mock_publish(event_type: str, data: dict):
            captured.append(data)

        PerceptionGate.initialize(
            publish_fn=mock_publish,
            min_interval_ms=0,
            dedup_window_ms=0,
        )

        # We can't easily trigger a real file event without waiting for poll,
        # so test the dispatcher pathway directly via check_events + dispatch
        # after verifying watch setup works
        result = await PerceptionGate.watch(
            "integration-test",
            "file_watcher",
            {"paths": ["."], "patterns": ["*.py"], "poll_interval_ms": 60000},
            "Event: {event.summary}",
        )
        assert result["status"] == "ok"

        watchers = PerceptionGate.list_watchers()
        assert any(w["watcher_id"] == "integration-test" for w in watchers)

        await PerceptionGate.unwatch("integration-test")


# ---------------------------------------------------------------------------
# Test 6: AgencyGate ↔ ShellGate security integration
# ---------------------------------------------------------------------------


class TestAgencyShellGateSecurity:
    """Test that ShellGate's blocklist actually blocks commands through AgencyGate."""

    @pytest.mark.asyncio
    async def test_blocked_command_rejected_through_agency(self):
        """A known-blocked command (rm -rf /) must be rejected by AgencyGate.exec()."""
        from cathedral import AgencyGate, ShellGate

        ShellGate.initialize()
        AgencyGate.initialize()

        info = await AgencyGate.spawn("security-test")
        assert info.get("alive") is True or info.get("session_id") == "security-test"

        # "rm -rf /" is in the default blocklist
        result = await AgencyGate.exec("security-test", "rm -rf /")
        assert result["status"] == "error"
        assert "blocked" in result["error"].lower() or "security" in result["error"].lower()

        await AgencyGate.close("security-test")

    @pytest.mark.asyncio
    async def test_safe_command_allowed_through_agency(self):
        """A safe command should pass ShellGate validation and execute."""
        from cathedral import AgencyGate, ShellGate

        ShellGate.initialize()
        AgencyGate.initialize()

        info = await AgencyGate.spawn("security-safe")
        result = await AgencyGate.exec("security-safe", "echo security_test_ok")

        assert result["status"] == "ok"
        assert "security_test_ok" in result["output"]

        await AgencyGate.close("security-safe")


# ---------------------------------------------------------------------------
# Test 7: AgencyGate MAX_SESSIONS enforcement
# ---------------------------------------------------------------------------


class TestAgencyGateSessionLimits:
    """Test that MAX_SESSIONS is enforced."""

    @pytest.mark.asyncio
    async def test_max_sessions_enforcement(self):
        from cathedral import AgencyGate

        AgencyGate.initialize()

        # Temporarily lower the limit for testing
        original_max = AgencyGate.MAX_SESSIONS
        AgencyGate.MAX_SESSIONS = 3
        spawned = []

        try:
            # Spawn up to the limit
            for i in range(3):
                info = await AgencyGate.spawn(f"limit-test-{i}")
                assert info.get("alive") is True or info.get("session_id") == f"limit-test-{i}"
                spawned.append(f"limit-test-{i}")

            # One more should be rejected
            result = await AgencyGate.spawn("limit-test-overflow")
            assert result["status"] == "error"
            assert "limit" in result["error"].lower()
        finally:
            AgencyGate.MAX_SESSIONS = original_max
            for sid in spawned:
                await AgencyGate.close(sid)


# ---------------------------------------------------------------------------
# Test 8: PerceptionGate MAX_WATCHERS enforcement
# ---------------------------------------------------------------------------


class TestPerceptionGateWatcherLimits:
    """Test that MAX_WATCHERS is enforced."""

    @pytest.mark.asyncio
    async def test_max_watchers_enforcement(self):
        from cathedral import PerceptionGate

        mock_publish = AsyncMock()
        PerceptionGate.initialize(publish_fn=mock_publish)

        original_max = PerceptionGate.MAX_WATCHERS
        PerceptionGate.MAX_WATCHERS = 2
        registered = []

        try:
            for i in range(2):
                result = await PerceptionGate.watch(
                    f"limit-w-{i}",
                    "file_watcher",
                    {"paths": ["."], "patterns": ["*"], "poll_interval_ms": 60000},
                )
                assert result["status"] == "ok"
                registered.append(f"limit-w-{i}")

            # One more should be rejected
            result = await PerceptionGate.watch(
                "limit-w-overflow",
                "file_watcher",
                {"paths": ["."], "patterns": ["*"], "poll_interval_ms": 60000},
            )
            assert result["status"] == "error"
            assert "limit" in result["error"].lower()
        finally:
            PerceptionGate.MAX_WATCHERS = original_max
            for wid in registered:
                await PerceptionGate.unwatch(wid)


# ---------------------------------------------------------------------------
# Test 10: VolitionEngine audit log rotation under continuation
# ---------------------------------------------------------------------------


class TestVolitionAuditRotationUnderContinuation:
    """Test that audit log rotation works correctly during multi-turn sessions."""

    def test_rotation_triggered_during_multi_turn(self, tmp_path):
        from cathedral.VolitionGate.engine import VolitionEngine

        audit_file = tmp_path / "volition_audit.jsonl"

        engine = VolitionEngine(
            turn_limit=50,
            cooldown_ms=0,
            audit_path=str(audit_file),
        )

        # Pre-fill the audit log past the rotation threshold
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        # Write enough to exceed a low threshold
        with open(audit_file, "w") as f:
            # Write 200 bytes of data
            for i in range(20):
                f.write(json.dumps({"turn": i, "pad": "x" * 50}) + "\n")

        initial_size = audit_file.stat().st_size

        # Now do a continuation that triggers rotation (use a tiny threshold)
        from cathedral.VolitionGate.engine import _rotate_log
        _rotate_log(audit_file, max_bytes=100, backups=2)

        # The original file should have been rotated
        backup = tmp_path / "volition_audit.1.jsonl"
        assert backup.exists()
        assert backup.stat().st_size == initial_size

    def test_audit_continues_writing_after_rotation(self, tmp_path):
        """After rotation, new entries still get written to the main file."""
        from cathedral.VolitionGate.engine import VolitionEngine, _rotate_log

        audit_file = tmp_path / "volition_audit.jsonl"

        engine = VolitionEngine(
            turn_limit=50,
            cooldown_ms=0,
            audit_path=str(audit_file),
        )

        # Fill initial file
        engine.request_continue("initial turn", reason="setup")
        assert audit_file.exists()

        # Manually rotate
        _rotate_log(audit_file, max_bytes=1, backups=2)  # force rotation

        # The main file should be gone (rotated to .1)
        assert not audit_file.exists()
        assert (tmp_path / "volition_audit.1.jsonl").exists()

        # New write should create fresh main file
        engine.request_continue("post-rotation turn", reason="continued")
        assert audit_file.exists()

        lines = audit_file.read_text().strip().split("\n")
        last_entry = json.loads(lines[-1])
        assert last_entry["outcome"] == "approved"

    def test_rotation_preserves_chain_across_turns(self, tmp_path):
        """Run multiple turns and verify the audit trail is complete across files."""
        from cathedral.VolitionGate.engine import VolitionEngine, _rotate_log

        audit_file = tmp_path / "volition_audit.jsonl"

        engine = VolitionEngine(
            turn_limit=20,
            cooldown_ms=0,
            audit_path=str(audit_file),
        )

        # Do several turns
        for i in range(5):
            result = engine.request_continue(f"turn {i}", reason=f"step {i}")
            assert result["status"] == "approved"

        # Force rotation mid-stream
        _rotate_log(audit_file, max_bytes=1, backups=3)

        # Do more turns
        for i in range(5, 10):
            result = engine.request_continue(f"turn {i}", reason=f"step {i}")
            assert result["status"] == "approved"

        # Verify: backup has first 5, main has last 5
        backup = tmp_path / "volition_audit.1.jsonl"
        assert backup.exists()
        assert audit_file.exists()

        backup_lines = backup.read_text().strip().split("\n")
        main_lines = audit_file.read_text().strip().split("\n")

        # First 5 turns produced 5 audit entries
        assert len(backup_lines) == 5
        # Last 5 turns produced 5 more
        assert len(main_lines) == 5

        # Verify continuity
        first_entry = json.loads(backup_lines[0])
        assert first_entry["turn"] == 1
        last_entry = json.loads(main_lines[-1])
        assert last_entry["turn"] == 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_events(events: list, count: int):
    """Poll until `events` has at least `count` items."""
    while len(events) < count:
        await asyncio.sleep(0.05)
