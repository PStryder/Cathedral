"""Tests for VolitionGate agent self-continuation."""

import pytest
from unittest.mock import patch

from cathedral.VolitionGate.engine import VolitionEngine


class TestVolitionEngine:
    """Tests for the VolitionEngine guardrails."""

    def test_first_continue_approved(self):
        engine = VolitionEngine(turn_limit=5, cooldown_ms=0)
        result = engine.request_continue("do something", reason="test")
        assert result["status"] == "approved"
        assert result["turn_count"] == 1
        assert result["remaining"] == 4

    def test_turn_limit_enforced(self):
        engine = VolitionEngine(turn_limit=2, cooldown_ms=0)
        engine.request_continue("turn 1")
        engine.request_continue("turn 2")
        result = engine.request_continue("turn 3")
        assert result["status"] == "denied"
        assert result["error_code"] == "TURN_LIMIT_REACHED"

    def test_cooldown_enforced(self):
        engine = VolitionEngine(turn_limit=10, cooldown_ms=60000)
        engine.request_continue("turn 1")
        result = engine.request_continue("turn 2")
        assert result["status"] == "denied"
        assert result["error_code"] == "COOLDOWN_ACTIVE"

    def test_get_status(self):
        engine = VolitionEngine(turn_limit=20, cooldown_ms=0)
        status = engine.get_status()
        assert status["active"] is False
        assert status["turn_count"] == 0
        assert status["turn_limit"] == 20

        engine.request_continue("test")
        status = engine.get_status()
        assert status["active"] is True
        assert status["turn_count"] == 1

    def test_reset(self):
        engine = VolitionEngine(turn_limit=5, cooldown_ms=0)
        engine.request_continue("turn 1")
        engine.request_continue("turn 2")

        engine.reset()
        status = engine.get_status()
        assert status["active"] is False
        assert status["turn_count"] == 0

    def test_prompts_logged(self):
        engine = VolitionEngine(turn_limit=5, cooldown_ms=0)
        engine.request_continue("do task A", reason="first step")
        engine.request_continue("do task B", reason="second step")

        status = engine.get_status()
        prompts = status["recent_prompts"]
        assert len(prompts) == 2
        assert prompts[0]["reason"] == "first step"
        assert prompts[1]["reason"] == "second step"

    def test_session_start_set_on_first_call(self):
        engine = VolitionEngine(turn_limit=5, cooldown_ms=0)
        assert engine.session_start is None
        engine.request_continue("start")
        assert engine.session_start is not None


class TestAuditLogRotation:
    """Tests for audit log rotation."""

    def test_rotate_creates_backup(self, tmp_path):
        from cathedral.VolitionGate.engine import _rotate_log

        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("x" * 100)

        _rotate_log(log_file, max_bytes=50, backups=3)

        assert not log_file.exists()
        assert (tmp_path / "audit.1.jsonl").exists()
        assert (tmp_path / "audit.1.jsonl").read_text() == "x" * 100

    def test_rotate_shifts_existing_backups(self, tmp_path):
        from cathedral.VolitionGate.engine import _rotate_log

        log_file = tmp_path / "audit.jsonl"
        (tmp_path / "audit.1.jsonl").write_text("old-1")

        log_file.write_text("x" * 100)
        _rotate_log(log_file, max_bytes=50, backups=3)

        assert (tmp_path / "audit.2.jsonl").read_text() == "old-1"
        assert (tmp_path / "audit.1.jsonl").read_text() == "x" * 100

    def test_rotate_drops_beyond_max_backups(self, tmp_path):
        from cathedral.VolitionGate.engine import _rotate_log

        log_file = tmp_path / "audit.jsonl"
        (tmp_path / "audit.1.jsonl").write_text("old-1")
        (tmp_path / "audit.2.jsonl").write_text("old-2")
        (tmp_path / "audit.3.jsonl").write_text("old-3")

        log_file.write_text("x" * 100)
        _rotate_log(log_file, max_bytes=50, backups=3)

        # old-3 should be overwritten by old-2 shifting
        assert (tmp_path / "audit.3.jsonl").read_text() == "old-2"
        assert (tmp_path / "audit.2.jsonl").read_text() == "old-1"
        assert (tmp_path / "audit.1.jsonl").read_text() == "x" * 100

    def test_no_rotate_under_threshold(self, tmp_path):
        from cathedral.VolitionGate.engine import _rotate_log

        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("small")

        _rotate_log(log_file, max_bytes=1000, backups=3)

        assert log_file.exists()
        assert not (tmp_path / "audit.1.jsonl").exists()

    def test_engine_audit_writes_to_custom_path(self, tmp_path):
        audit_file = tmp_path / "test_audit.jsonl"
        engine = VolitionEngine(
            turn_limit=5, cooldown_ms=0, audit_path=str(audit_file)
        )
        engine.request_continue("test action", reason="testing")

        assert audit_file.exists()
        import json
        line = json.loads(audit_file.read_text().strip().split("\n")[-1])
        assert line["outcome"] == "approved"


class TestVolitionPipelineHelpers:
    """Tests for the pipeline integration helpers."""

    def test_contains_volition_continue(self):
        from cathedral.pipeline.chat import _contains_volition_continue
        assert _contains_volition_continue('{"tool": "VolitionGate.request_continue"}') is True
        assert _contains_volition_continue('just normal text') is False

    def test_extract_volition_continue(self):
        from cathedral.pipeline.chat import _extract_volition_continue

        response = (
            'Some text before\n'
            '{"type": "tool_call", "id": "tc_001", '
            '"tool": "VolitionGate.request_continue", '
            '"args": {"text": "analyze results", "reason": "follow-up"}}\n'
            'Some text after'
        )
        result = _extract_volition_continue(response)
        assert result is not None
        assert result["text"] == "analyze results"
        assert result["reason"] == "follow-up"

    def test_extract_volition_continue_no_match(self):
        from cathedral.pipeline.chat import _extract_volition_continue
        result = _extract_volition_continue("no tool calls here")
        assert result is None
