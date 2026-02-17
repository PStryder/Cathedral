"""
AgencyGate - Persistent shell sessions for Cathedral.

Provides persistent shell sessions that maintain state (env vars, cwd, aliases)
across tool calls. Built on top of ShellGate's security validation layer.

Usage:
    from cathedral import AgencyGate

    AgencyGate.initialize()

    # Spawn a persistent session
    info = await AgencyGate.spawn("my-session")

    # Execute commands (state persists between calls)
    result = await AgencyGate.exec("my-session", "export FOO=bar")
    result = await AgencyGate.exec("my-session", "echo $FOO")  # -> "bar"

    # List active sessions
    sessions = AgencyGate.list_sessions()

    # Close a session
    await AgencyGate.close("my-session")
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cathedral.shared.gate import (
    GateLogger,
    PathUtils,
    build_health_status,
)

from .session import ShellSession, detect_shell

_log = GateLogger.get("AgencyGate")

# Module-level state
_sessions: Dict[str, ShellSession] = {}
_initialized: bool = False
_log_dir: Path = Path("data/shell_history")
MAX_SESSIONS: int = 20


def initialize() -> bool:
    """Initialize AgencyGate."""
    global _initialized

    if _initialized:
        return True

    try:
        PathUtils.ensure_dirs(_log_dir)
        _initialized = True
        _log.info("AgencyGate initialized")
        return True
    except Exception as e:
        _log.error(f"AgencyGate initialization failed: {e}")
        return False


def is_initialized() -> bool:
    return _initialized


def is_healthy() -> bool:
    if not _initialized:
        return False
    return True


def get_health_status() -> Dict[str, Any]:
    active = sum(1 for s in _sessions.values() if s.alive)
    return build_health_status(
        gate_name="AgencyGate",
        initialized=_initialized,
        dependencies=["shell"],
        checks={
            "log_dir_exists": _log_dir.exists(),
        },
        details={
            "active_sessions": active,
            "total_sessions": len(_sessions),
        },
    )


def get_dependencies() -> List[str]:
    return ["shell"]


async def spawn(
    session_id: str | None = None,
    shell: str | None = None,
    cwd: str | None = None,
) -> Dict[str, Any]:
    """Spawn a new persistent shell session.

    Args:
        session_id: Unique ID (auto-generated if omitted)
        shell: Shell binary (auto-detected if omitted)
        cwd: Working directory (defaults to current)

    Returns:
        Session metadata dict
    """
    if not _initialized:
        initialize()

    sid = session_id or f"ses_{uuid.uuid4().hex[:8]}"

    # Enforce session limit
    active_count = sum(1 for s in _sessions.values() if s.alive)
    if active_count >= MAX_SESSIONS:
        return {
            "status": "error",
            "error": f"Session limit reached ({MAX_SESSIONS}). Close a session first.",
        }

    if sid in _sessions:
        existing = _sessions[sid]
        if existing.alive:
            return existing.info()
        # Dead session — close log file before removing
        await existing.close()
        del _sessions[sid]

    try:
        session = ShellSession(
            session_id=sid,
            shell=shell,
            cwd=cwd,
            log_dir=_log_dir,
        )
    except ValueError as e:
        return {"status": "error", "error": str(e)}

    await session.start()
    _sessions[sid] = session

    _log.info(f"Session '{sid}' spawned")
    return session.info()


async def exec(
    session_id: str,
    command: str,
    timeout_ms: int = 30000,
) -> Dict[str, Any]:
    """Execute a command in a persistent session.

    All commands pass through ShellGate.validate_command() before execution.

    Args:
        session_id: Session to execute in
        command: Command string
        timeout_ms: Timeout in milliseconds

    Returns:
        Dict with status, output, exit_code, command
    """
    if session_id not in _sessions:
        return {"status": "error", "error": f"Session '{session_id}' not found"}

    session = _sessions[session_id]
    if not session.alive:
        return {"status": "error", "error": f"Session '{session_id}' is not running"}

    # Route through ShellGate security validation (FAIL CLOSED)
    try:
        from cathedral import ShellGate
        is_valid, error_msg = ShellGate.validate_command(command)
        if not is_valid:
            return {
                "status": "error",
                "error": f"Command blocked by security policy: {error_msg}",
                "command": command,
            }
    except ImportError:
        _log.error("ShellGate unavailable — command blocked (fail-closed)")
        return {
            "status": "error",
            "error": "Security validation unavailable (ShellGate not loaded). Command blocked.",
            "command": command,
        }
    except Exception as e:
        _log.error(f"Security validation error: {e} — command blocked (fail-closed)")
        return {
            "status": "error",
            "error": f"Security validation failed: {e}",
            "command": command,
        }

    return await session.execute(command, timeout_ms)


def list_sessions() -> List[Dict[str, Any]]:
    """List all sessions with metadata."""
    return [s.info() for s in _sessions.values()]


async def close(session_id: str) -> Dict[str, Any]:
    """Close a persistent shell session.

    Args:
        session_id: Session to close

    Returns:
        Dict with status information
    """
    if session_id not in _sessions:
        return {"status": "error", "error": f"Session '{session_id}' not found"}

    session = _sessions[session_id]
    cmd_count = session.command_count
    await session.close()
    del _sessions[session_id]

    _log.info(f"Session '{session_id}' closed")
    return {
        "status": "closed",
        "session_id": session_id,
        "commands_executed": cmd_count,
    }


async def close_all() -> int:
    """Close all sessions. Returns count closed."""
    count = 0
    for sid in list(_sessions.keys()):
        await close(sid)
        count += 1
    return count


async def _reset() -> None:
    """Reset module state (for testing only)."""
    global _initialized
    await close_all()
    _initialized = False


def get_info() -> dict:
    """Get comprehensive documentation for AgencyGate."""
    return {
        "gate": "AgencyGate",
        "version": "1.0",
        "purpose": "Persistent shell sessions that maintain state across tool calls. "
        "Environment variables, working directory, and aliases persist between commands.",
        "tools": {
            "spawn": {
                "purpose": "Create a new persistent shell session",
                "call_format": {
                    "session_id": {"type": "string", "required": False, "description": "Unique ID (auto-generated if omitted)"},
                    "shell": {"type": "string", "required": False, "description": "Shell binary (auto-detected if omitted)"},
                    "cwd": {"type": "string", "required": False, "description": "Working directory"},
                },
            },
            "exec": {
                "purpose": "Execute a command in a persistent session",
                "call_format": {
                    "session_id": {"type": "string", "required": True, "description": "Session to execute in"},
                    "command": {"type": "string", "required": True, "description": "Command to execute"},
                    "timeout_ms": {"type": "integer", "required": False, "default": 30000, "description": "Timeout in ms"},
                },
            },
            "list": {
                "purpose": "List all active sessions",
            },
            "close": {
                "purpose": "Close a session",
                "call_format": {
                    "session_id": {"type": "string", "required": True, "description": "Session to close"},
                },
            },
        },
    }


__all__ = [
    "initialize",
    "is_initialized",
    "is_healthy",
    "get_health_status",
    "get_dependencies",
    "spawn",
    "exec",
    "list_sessions",
    "close",
    "close_all",
    "get_info",
]
