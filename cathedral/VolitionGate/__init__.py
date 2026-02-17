"""
VolitionGate - Agent self-continuation for Cathedral.

Enables the agent to autonomously continue processing without waiting
for user input. The agent calls volition_continue as a tool call, and
the pipeline intercepts it to feed continuation text back as a new turn.

Usage:
    from cathedral import VolitionGate

    VolitionGate.initialize()

    # Agent requests continuation (called via ToolGate)
    result = VolitionGate.request_continue("Analyze the results", reason="follow-up needed")

    # Check status
    status = VolitionGate.get_status()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cathedral.shared.gate import (
    GateLogger,
    build_health_status,
)
from cathedral import Config

from .engine import VolitionEngine

_log = GateLogger.get("VolitionGate")

# Module-level state
_engine: Optional[VolitionEngine] = None
_initialized: bool = False


def _get_engine() -> VolitionEngine:
    global _engine
    if _engine is None:
        initialize()
    return _engine


def initialize() -> bool:
    """Initialize VolitionGate with config from Cathedral's ConfigManager."""
    global _engine, _initialized

    if _initialized:
        return True

    try:
        turn_limit = int(Config.get("VOLITION_TURN_LIMIT", 20))
        cooldown_ms = int(Config.get("VOLITION_COOLDOWN_MS", 1000))
        budget_usd = float(Config.get("VOLITION_BUDGET_USD", 5.00))
        audit_mode = Config.get("VOLITION_AUDIT_MODE", "local")

        _engine = VolitionEngine(
            turn_limit=turn_limit,
            cooldown_ms=cooldown_ms,
            budget_usd=budget_usd,
            audit_mode=audit_mode,
        )

        _initialized = True
        _log.info(f"VolitionGate initialized (turn_limit={turn_limit}, cooldown={cooldown_ms}ms)")
        return True

    except Exception as e:
        _log.error(f"VolitionGate initialization failed: {e}")
        return False


def is_initialized() -> bool:
    return _initialized


def is_healthy() -> bool:
    return _initialized and _engine is not None


def get_health_status() -> Dict[str, Any]:
    engine = _engine
    details = {}
    if engine:
        details = {
            "turn_count": engine.turn_count,
            "turn_limit": engine.turn_limit,
            "active": engine.session_start is not None,
        }

    return build_health_status(
        gate_name="VolitionGate",
        initialized=_initialized,
        dependencies=[],
        checks={
            "engine_ready": _engine is not None,
        },
        details=details,
    )


def get_dependencies() -> List[str]:
    return []


def request_continue(text: str, reason: str = "") -> Dict[str, Any]:
    """Request an autonomous continuation.

    Called by the agent via ToolGate. The pipeline checks the result
    and feeds the text back as a new turn if approved.

    Args:
        text: Continuation instruction
        reason: Audit trail explanation

    Returns:
        Dict with status (approved/denied) and details
    """
    return _get_engine().request_continue(text, reason)


def get_status() -> Dict[str, Any]:
    """Get current volition session state."""
    return _get_engine().get_status()


def reset() -> None:
    """Reset the session (start fresh)."""
    if _engine:
        _engine.reset()


def _reset() -> None:
    """Reset module state (for testing only)."""
    global _initialized, _engine
    if _engine:
        _engine.reset()
    _engine = None
    _initialized = False


def get_info() -> dict:
    """Get comprehensive documentation for VolitionGate."""
    return {
        "gate": "VolitionGate",
        "version": "1.0",
        "purpose": "Enable the agent to autonomously continue processing without "
        "waiting for user input. The agent calls volition_continue with a "
        "continuation instruction, and the pipeline feeds it back as a new turn.",
        "tools": {
            "continue": {
                "purpose": "Request an autonomous continuation turn",
                "call_format": {
                    "text": {"type": "string", "required": True, "description": "Continuation instruction"},
                    "reason": {"type": "string", "required": False, "description": "Why this continuation is needed (audit trail)"},
                },
                "response": {
                    "status": "string - 'approved' or 'denied'",
                    "turn_count": "integer - current turn number",
                    "remaining": "integer - turns remaining",
                },
            },
            "status": {
                "purpose": "Get current volition session state",
                "response": {
                    "active": "boolean - whether a session is active",
                    "turn_count": "integer - turns used",
                    "turn_limit": "integer - max turns",
                    "remaining": "integer - turns left",
                },
            },
        },
        "guardrails": {
            "turn_limit": "Max autonomous turns per session (default 20, user-configurable)",
            "cooldown": "Minimum delay between turns (default 1000ms)",
            "budget": "Cost ceiling if token tracking available",
            "audit": "All continuations logged to audit file",
        },
        "note": "The agent CANNOT modify turn_limit or budget. Only the user can change these via config.",
    }


__all__ = [
    "initialize",
    "is_initialized",
    "is_healthy",
    "get_health_status",
    "get_dependencies",
    "request_continue",
    "get_status",
    "reset",
    "get_info",
]
