"""
ShellGate - Secure shell command execution for Cathedral.

Provides:
- Command execution with security validation
- Blocklist/allowlist for commands
- Environment variable sanitization
- Background command execution
- Streaming output
- Command history

Usage:
    from cathedral import ShellGate

    # Initialize (call on startup)
    ShellGate.initialize()

    # Execute a command
    result = ShellGate.execute("ls -la")

    # Execute in background
    execution = ShellGate.execute_background("long-running-task")
    status = ShellGate.get_status(execution.id)

    # Cancel background command
    ShellGate.cancel(execution.id)
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator
import uuid

from .models import (
    CommandConfig,
    CommandExecution,
    CommandResult,
    CommandStatus,
    ShellConfig,
    HistoryEntry,
)
from .security import (
    validate_command,
    sanitize_environment,
    check_dangerous_constructs,
    estimate_command_risk,
    CommandSecurityError,
)
from .executor import (
    execute_sync,
    execute_async,
    execute_streaming,
    start_background,
    get_background_status,
    cancel_background,
    list_background,
    cleanup_completed,
)


# Module-level state
_config: Optional[ShellConfig] = None
_history: List[HistoryEntry] = []
_initialized: bool = False

# Default paths
DEFAULT_CONFIG_PATH = "data/config/shell.json"
DEFAULT_HISTORY_PATH = "data/shell_history/history.json"


class ShellGate:
    """
    Main interface for Cathedral's shell command execution.

    All methods are class methods for easy access throughout the application.
    """

    @classmethod
    def initialize(
        cls,
        config_path: Optional[str] = None,
        history_path: Optional[str] = None
    ) -> bool:
        """
        Initialize the shell gate.

        Args:
            config_path: Path to config file
            history_path: Path to history file

        Returns:
            True if initialization successful
        """
        global _config, _history, _initialized

        try:
            # Resolve paths
            if config_path is None:
                config_path = DEFAULT_CONFIG_PATH
            if history_path is None:
                history_path = DEFAULT_HISTORY_PATH

            # Ensure directories exist
            for path in [config_path, history_path]:
                dir_path = os.path.dirname(path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)

            # Load or create config
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _config = ShellConfig.from_dict(data)
            else:
                _config = ShellConfig()
                cls._save_config(config_path)

            # Load history
            if os.path.exists(history_path):
                try:
                    with open(history_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        _history = [HistoryEntry.model_validate(item) for item in data]
                except (json.JSONDecodeError, Exception):
                    _history = []
            else:
                _history = []

            _initialized = True
            return True

        except Exception as e:
            print(f"[ShellGate] Initialization failed: {e}")
            return False

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the gate is initialized."""
        return _initialized

    @classmethod
    def _save_config(cls, config_path: Optional[str] = None):
        """Save configuration to disk."""
        global _config
        if _config is None:
            return

        path = config_path or DEFAULT_CONFIG_PATH
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(_config.to_dict(), f, indent=2)

    @classmethod
    def _save_history(cls, history_path: Optional[str] = None):
        """Save history to disk."""
        global _history
        path = history_path or DEFAULT_HISTORY_PATH
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump([h.to_dict() for h in _history], f, indent=2, default=str)

    @classmethod
    def _get_config(cls) -> ShellConfig:
        """Get current config, initializing if needed."""
        global _config
        if _config is None:
            cls.initialize()
        return _config

    @classmethod
    def _add_to_history(cls, execution: CommandExecution):
        """Add an execution to history."""
        global _history
        config = cls._get_config()

        if not config.command_config.log_commands:
            return

        entry = HistoryEntry.from_execution(execution)
        _history.append(entry)

        # Trim history if needed
        if len(_history) > config.history_max_entries:
            _history = _history[-config.history_max_entries:]

        cls._save_history()

    # ==================== Command Execution ====================

    @classmethod
    def execute(
        cls,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None
    ) -> CommandResult:
        """
        Execute a command synchronously.

        Args:
            command: Command string to execute
            working_dir: Working directory
            timeout: Timeout in seconds
            env: Additional environment variables

        Returns:
            CommandResult
        """
        config = cls._get_config().command_config
        result = execute_sync(command, config, working_dir, timeout, env)

        # Add to history
        execution = CommandExecution(
            id=result.execution_id or str(uuid.uuid4())[:8],
            command=command,
            working_dir=working_dir or config.default_working_dir or os.getcwd(),
            status=result.status,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            started_at=datetime.utcnow() - timedelta(seconds=result.duration_seconds),
            completed_at=datetime.utcnow()
        )
        cls._add_to_history(execution)

        return result

    @classmethod
    async def execute_async(
        cls,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None
    ) -> CommandResult:
        """
        Execute a command asynchronously.

        Args:
            command: Command string to execute
            working_dir: Working directory
            timeout: Timeout in seconds
            env: Additional environment variables

        Returns:
            CommandResult
        """
        config = cls._get_config().command_config
        result = await execute_async(command, config, working_dir, timeout, env)

        # Add to history
        execution = CommandExecution(
            id=result.execution_id or str(uuid.uuid4())[:8],
            command=command,
            working_dir=working_dir or config.default_working_dir or os.getcwd(),
            status=result.status,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            started_at=datetime.utcnow() - timedelta(seconds=result.duration_seconds),
            completed_at=datetime.utcnow()
        )
        cls._add_to_history(execution)

        return result

    @classmethod
    async def execute_stream(
        cls,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Execute a command with streaming output.

        Args:
            command: Command string
            working_dir: Working directory
            timeout: Timeout in seconds
            env: Additional environment variables

        Yields:
            Output lines as they become available
        """
        config = cls._get_config().command_config
        async for line in execute_streaming(command, config, working_dir, timeout, env):
            yield line

    @classmethod
    def execute_background(
        cls,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> CommandExecution:
        """
        Execute a command in the background.

        Args:
            command: Command string
            working_dir: Working directory
            env: Additional environment variables

        Returns:
            CommandExecution record (check status with get_status)
        """
        config = cls._get_config().command_config

        def on_complete(execution: CommandExecution):
            cls._add_to_history(execution)

        return start_background(command, config, working_dir, env, on_complete)

    @classmethod
    def get_status(cls, execution_id: str) -> Optional[CommandExecution]:
        """Get status of a background command."""
        return get_background_status(execution_id)

    @classmethod
    def cancel(cls, execution_id: str) -> bool:
        """Cancel a background command."""
        return cancel_background(execution_id)

    @classmethod
    def list_running(cls) -> List[Dict[str, Any]]:
        """List running background commands."""
        executions = list_background()
        return [e.to_dict() for e in executions if e.status == CommandStatus.RUNNING]

    @classmethod
    def list_background(cls) -> List[Dict[str, Any]]:
        """List all background commands (running and completed)."""
        return [e.to_dict() for e in list_background()]

    # ==================== Security ====================

    @classmethod
    def validate_command(cls, command: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a command is allowed.

        Args:
            command: Command to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        config = cls._get_config().command_config
        return validate_command(command, config)

    @classmethod
    def check_warnings(cls, command: str) -> List[str]:
        """
        Check for warnings about a command.

        Args:
            command: Command to check

        Returns:
            List of warning strings
        """
        return check_dangerous_constructs(command)

    @classmethod
    def estimate_risk(cls, command: str) -> Dict[str, Any]:
        """
        Estimate the risk level of a command.

        Args:
            command: Command to analyze

        Returns:
            Dict with risk_level, is_blocked, warnings
        """
        config = cls._get_config().command_config
        return estimate_command_risk(command, config)

    # ==================== History ====================

    @classmethod
    def get_history(
        cls,
        limit: int = 50,
        success_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get command history.

        Args:
            limit: Maximum entries to return
            success_only: Only return successful commands

        Returns:
            List of history entries (newest first)
        """
        global _history
        entries = _history.copy()

        if success_only:
            entries = [e for e in entries if e.success]

        # Sort by started_at descending
        entries.sort(key=lambda e: e.started_at, reverse=True)

        return [e.to_dict() for e in entries[:limit]]

    @classmethod
    def clear_history(cls) -> int:
        """Clear command history. Returns number of entries cleared."""
        global _history
        count = len(_history)
        _history = []
        cls._save_history()
        return count

    @classmethod
    def cleanup_old_history(cls) -> int:
        """Remove history entries older than retention period."""
        global _history
        config = cls._get_config()
        cutoff = datetime.utcnow() - timedelta(days=config.history_retention_days)

        original_count = len(_history)
        _history = [e for e in _history if e.started_at >= cutoff]
        removed = original_count - len(_history)

        if removed > 0:
            cls._save_history()

        return removed

    # ==================== Configuration ====================

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get full configuration."""
        return cls._get_config().to_dict()

    @classmethod
    def update_config(
        cls,
        default_timeout_seconds: Optional[int] = None,
        max_timeout_seconds: Optional[int] = None,
        default_working_dir: Optional[str] = None,
        allowed_commands: Optional[List[str]] = None,
        blocked_commands: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
        require_unlock: Optional[bool] = None,
        log_commands: Optional[bool] = None,
        max_concurrent_background: Optional[int] = None
    ):
        """Update command configuration."""
        config = cls._get_config()
        cmd_config = config.command_config

        if default_timeout_seconds is not None:
            cmd_config.default_timeout_seconds = default_timeout_seconds
        if max_timeout_seconds is not None:
            cmd_config.max_timeout_seconds = max_timeout_seconds
        if default_working_dir is not None:
            cmd_config.default_working_dir = default_working_dir
        if allowed_commands is not None:
            cmd_config.allowed_commands = allowed_commands
        if blocked_commands is not None:
            cmd_config.blocked_commands = blocked_commands
        if blocked_patterns is not None:
            cmd_config.blocked_patterns = blocked_patterns
        if require_unlock is not None:
            cmd_config.require_unlock = require_unlock
        if log_commands is not None:
            cmd_config.log_commands = log_commands
        if max_concurrent_background is not None:
            cmd_config.max_concurrent_background = max_concurrent_background

        cls._save_config()

    @classmethod
    def add_blocked_command(cls, command: str):
        """Add a command to the blocklist."""
        config = cls._get_config().command_config
        if command not in config.blocked_commands:
            config.blocked_commands.append(command)
            cls._save_config()

    @classmethod
    def remove_blocked_command(cls, command: str) -> bool:
        """Remove a command from the blocklist."""
        config = cls._get_config().command_config
        if command in config.blocked_commands:
            config.blocked_commands.remove(command)
            cls._save_config()
            return True
        return False

    @classmethod
    def add_allowed_command(cls, command: str):
        """Add a command to the allowlist."""
        config = cls._get_config().command_config
        if command not in config.allowed_commands:
            config.allowed_commands.append(command)
            cls._save_config()

    @classmethod
    def cleanup(cls):
        """Clean up completed background commands."""
        cleanup_completed()


# ==================== Convenience Functions ====================

def initialize(config_path: Optional[str] = None, history_path: Optional[str] = None) -> bool:
    """Initialize ShellGate."""
    return ShellGate.initialize(config_path, history_path)


def is_initialized() -> bool:
    """Check if initialized."""
    return ShellGate.is_initialized()


def execute(
    command: str,
    working_dir: Optional[str] = None,
    timeout: Optional[int] = None
) -> CommandResult:
    """Execute a command synchronously."""
    return ShellGate.execute(command, working_dir, timeout)


async def execute_async(
    command: str,
    working_dir: Optional[str] = None,
    timeout: Optional[int] = None
) -> CommandResult:
    """Execute a command asynchronously."""
    return await ShellGate.execute_async(command, working_dir, timeout)


def execute_background(
    command: str,
    working_dir: Optional[str] = None
) -> CommandExecution:
    """Execute in background."""
    return ShellGate.execute_background(command, working_dir)


def get_status(execution_id: str) -> Optional[CommandExecution]:
    """Get background command status."""
    return ShellGate.get_status(execution_id)


def cancel(execution_id: str) -> bool:
    """Cancel a background command."""
    return ShellGate.cancel(execution_id)


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get command history."""
    return ShellGate.get_history(limit)


def validate_command(command: str) -> Tuple[bool, Optional[str]]:
    """Validate a command."""
    return ShellGate.validate_command(command)
