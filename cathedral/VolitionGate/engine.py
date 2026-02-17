"""VolitionEngine - manages autonomous turn state and guardrails.

Ported from Velle's session state management. Instead of console injection,
Cathedral's pipeline feeds continuation text back into process_input_stream().
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("VolitionGate")


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


class VolitionEngine:
    """Manages autonomous continuation state and guardrails."""

    def __init__(
        self,
        turn_limit: int = 20,
        cooldown_ms: int = 1000,
        budget_usd: float = 5.00,
        audit_mode: str = "local",
        audit_path: str = "data/volition_audit.jsonl",
    ):
        self.turn_limit = turn_limit
        self.cooldown_ms = cooldown_ms
        self.budget_usd = budget_usd
        self.audit_mode = audit_mode
        self.audit_path = Path(audit_path)

        # Session state
        self.turn_count: int = 0
        self.session_start: str | None = None
        self.last_prompt_time: float | None = None  # monotonic clock
        self.prompts_log: deque[dict] = deque(maxlen=50)

    def request_continue(
        self,
        text: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Request an autonomous continuation turn.

        Validates guardrails and returns approval/denial.

        Args:
            text: The continuation instruction
            reason: Why this continuation is needed (audit trail)

        Returns:
            Dict with status and details
        """
        now = datetime.now(timezone.utc)

        # Initialize session on first call
        if self.session_start is None:
            self.session_start = now.isoformat()

        # Check turn limit
        if self.turn_count >= self.turn_limit:
            result = {
                "status": "denied",
                "error_code": "TURN_LIMIT_REACHED",
                "message": f"Turn limit reached ({self.turn_limit}). "
                "Cannot continue autonomously.",
                "turn_count": self.turn_count,
                "turn_limit": self.turn_limit,
                "timestamp": now.isoformat(),
            }
            self._audit({
                "tool": "volition_continue",
                "text": text[:200],
                "reason": reason,
                "outcome": "turn_limit_reached",
            })
            return result

        # Check cooldown (monotonic clock — immune to wall-clock drift)
        mono_now = time.monotonic()
        if self.last_prompt_time is not None:
            elapsed_ms = (mono_now - self.last_prompt_time) * 1000
            if elapsed_ms < self.cooldown_ms:
                return {
                    "status": "denied",
                    "error_code": "COOLDOWN_ACTIVE",
                    "message": f"Cooldown active ({self.cooldown_ms}ms between turns).",
                    "timestamp": now.isoformat(),
                }

        # Approved — update state
        self.turn_count += 1
        self.last_prompt_time = mono_now

        log_entry = {
            "turn": self.turn_count,
            "text_preview": text[:100],
            "reason": reason,
            "timestamp": now.isoformat(),
        }
        self.prompts_log.append(log_entry)

        self._audit({
            "tool": "volition_continue",
            "turn": self.turn_count,
            "text": text[:500],
            "reason": reason,
            "outcome": "approved",
        })

        return {
            "status": "approved",
            "turn_count": self.turn_count,
            "turn_limit": self.turn_limit,
            "remaining": self.turn_limit - self.turn_count,
            "timestamp": now.isoformat(),
        }

    def get_status(self) -> dict[str, Any]:
        """Get current session state."""
        return {
            "active": self.session_start is not None,
            "turn_count": self.turn_count,
            "turn_limit": self.turn_limit,
            "remaining": self.turn_limit - self.turn_count,
            "cooldown_ms": self.cooldown_ms,
            "budget_usd": self.budget_usd,
            "audit_mode": self.audit_mode,
            "session_start": self.session_start,
            "recent_prompts": list(self.prompts_log)[-10:],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def reset(self) -> None:
        """Reset session state (new autonomous session)."""
        self.turn_count = 0
        self.session_start = None
        self.last_prompt_time = None
        self.prompts_log.clear()

    def _audit(self, entry: dict) -> None:
        """Write audit log entry."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        entry["session_start"] = self.session_start

        if self.audit_mode in ("local", "both"):
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                _rotate_log(self.audit_path)
                with open(self.audit_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError as e:
                _log.warning(f"Failed to write audit log: {e}")
