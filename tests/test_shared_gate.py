"""
Tests for shared Gate utilities.
"""

import json
import logging
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from cathedral.shared.gate import (
    GateLogger,
    GateErrorHandler,
    GateHealth,
    build_health_status,
    ConfigLoader,
    PathUtils,
    GateOperationResult,
    deep_update,
    get_logger,
)


class TestGateLogger:
    """Tests for GateLogger."""

    def test_get_returns_logger(self):
        """Should return a logger instance."""
        logger = GateLogger.get("TestGate")

        assert isinstance(logger, logging.Logger)
        assert "cathedral.TestGate" in logger.name

    def test_get_same_logger_for_same_name(self):
        """Should return same logger for same gate name."""
        logger1 = GateLogger.get("SameGate")
        logger2 = GateLogger.get("SameGate")

        assert logger1 is logger2

    def test_get_different_loggers_for_different_names(self):
        """Should return different loggers for different gate names."""
        logger1 = GateLogger.get("Gate1")
        logger2 = GateLogger.get("Gate2")

        assert logger1 is not logger2

    def test_set_level_specific_gate(self):
        """Should set level for specific gate."""
        logger = GateLogger.get("LevelTestGate")
        GateLogger.set_level(logging.DEBUG, "LevelTestGate")

        assert logger.level == logging.DEBUG

    def test_get_logger_shortcut(self):
        """get_logger() should be equivalent to GateLogger.get()."""
        logger1 = GateLogger.get("ShortcutGate")
        logger2 = get_logger("ShortcutGate")

        assert logger1 is logger2


class TestGateErrorHandler:
    """Tests for GateErrorHandler."""

    def test_handle_logs_error(self, caplog):
        """handle() should log the error."""
        with caplog.at_level(logging.ERROR):
            result = GateErrorHandler.handle(
                gate_name="TestGate",
                operation="test_op",
                exception=ValueError("test error"),
                default_return="default",
            )

        assert result == "default"
        assert "test_op failed" in caplog.text
        assert "test error" in caplog.text

    def test_handle_custom_log_level(self, caplog):
        """handle() should use custom log level."""
        with caplog.at_level(logging.WARNING):
            GateErrorHandler.handle(
                gate_name="TestGate",
                operation="test_op",
                exception=ValueError("warning"),
                log_level=logging.WARNING,
            )

        assert "test_op failed" in caplog.text

    def test_wrap_decorator_catches_exception(self, caplog):
        """wrap() decorator should catch exceptions."""
        @GateErrorHandler.wrap("TestGate", "wrapped_op", default_return="failed")
        def failing_function():
            raise ValueError("intentional failure")

        with caplog.at_level(logging.ERROR):
            result = failing_function()

        assert result == "failed"
        assert "wrapped_op failed" in caplog.text

    def test_wrap_decorator_passes_through_success(self):
        """wrap() decorator should pass through successful results."""
        @GateErrorHandler.wrap("TestGate", "wrapped_op", default_return="failed")
        def succeeding_function():
            return "success"

        result = succeeding_function()
        assert result == "success"

    def test_wrap_decorator_reraise_option(self):
        """wrap() decorator should re-raise when reraise=True."""
        @GateErrorHandler.wrap("TestGate", "wrapped_op", reraise=True)
        def failing_function():
            raise ValueError("should propagate")

        with pytest.raises(ValueError, match="should propagate"):
            failing_function()

    def test_wrap_async_catches_exception(self, caplog):
        """wrap_async() decorator should catch async exceptions."""
        import asyncio

        @GateErrorHandler.wrap_async("TestGate", "async_op", default_return="async_failed")
        async def async_failing():
            raise ValueError("async failure")

        with caplog.at_level(logging.ERROR):
            result = asyncio.get_event_loop().run_until_complete(async_failing())

        assert result == "async_failed"
        assert "async_op failed" in caplog.text

    def test_wrap_async_passes_through_success(self):
        """wrap_async() decorator should pass through successful results."""
        import asyncio

        @GateErrorHandler.wrap_async("TestGate", "async_op", default_return="failed")
        async def async_succeeding():
            return "async_success"

        result = asyncio.get_event_loop().run_until_complete(async_succeeding())
        assert result == "async_success"

    def test_wrap_async_reraise_option(self):
        """wrap_async() decorator should re-raise when reraise=True."""
        import asyncio

        @GateErrorHandler.wrap_async("TestGate", "async_op", reraise=True)
        async def async_failing():
            raise ValueError("should propagate async")

        with pytest.raises(ValueError, match="should propagate async"):
            asyncio.get_event_loop().run_until_complete(async_failing())


class TestGateHealth:
    """Tests for GateHealth protocol and build_health_status."""

    def test_build_health_status_healthy(self):
        """Should build healthy status dict."""
        status = build_health_status(
            gate_name="TestGate",
            initialized=True,
            dependencies=["database", "api"],
            checks={"db_connected": True, "api_available": True},
            details={"version": "1.0"},
        )

        assert status["gate"] == "TestGate"
        assert status["healthy"] is True
        assert status["initialized"] is True
        assert "database" in status["dependencies"]
        assert status["checks"]["db_connected"] is True
        assert status["details"]["version"] == "1.0"

    def test_build_health_status_unhealthy_not_initialized(self):
        """Should be unhealthy if not initialized."""
        status = build_health_status(
            gate_name="TestGate",
            initialized=False,
            dependencies=[],
            checks={"check1": True},
        )

        assert status["healthy"] is False

    def test_build_health_status_unhealthy_check_failed(self):
        """Should be unhealthy if any check fails."""
        status = build_health_status(
            gate_name="TestGate",
            initialized=True,
            dependencies=[],
            checks={"check1": True, "check2": False},
        )

        assert status["healthy"] is False

    def test_build_health_status_empty_checks(self):
        """Should be healthy with empty checks if initialized."""
        status = build_health_status(
            gate_name="TestGate",
            initialized=True,
            dependencies=[],
            checks={},
        )

        assert status["healthy"] is True

    def test_gate_health_protocol(self):
        """Class implementing GateHealth should pass protocol check."""
        class HealthyGate:
            @classmethod
            def is_healthy(cls) -> bool:
                return True

            @classmethod
            def get_health_status(cls) -> Dict[str, Any]:
                return {"healthy": True}

            @classmethod
            def get_dependencies(cls) -> List[str]:
                return []

        assert isinstance(HealthyGate, type)
        # Runtime check for protocol
        assert hasattr(HealthyGate, "is_healthy")
        assert hasattr(HealthyGate, "get_health_status")
        assert hasattr(HealthyGate, "get_dependencies")


@dataclass
class SampleConfig:
    """Sample config class for testing ConfigLoader."""
    name: str = "default"
    value: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "SampleConfig":
        return cls(**data)

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value}


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_creates_default_when_file_missing(self, tmp_path):
        """Should create default config when file doesn't exist."""
        config_path = tmp_path / "missing.json"

        result = ConfigLoader.load(config_path, SampleConfig, create_default=True)

        assert result is not None
        assert result.name == "default"
        assert result.value == 0

    def test_load_returns_none_when_file_missing_and_no_default(self, tmp_path):
        """Should return None when file doesn't exist and create_default=False."""
        config_path = tmp_path / "missing.json"

        result = ConfigLoader.load(config_path, SampleConfig, create_default=False)

        assert result is None

    def test_load_parses_json_file(self, tmp_path):
        """Should load and parse JSON file."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"name": "loaded", "value": 42}')

        result = ConfigLoader.load(config_path, SampleConfig)

        assert result.name == "loaded"
        assert result.value == 42

    def test_load_handles_invalid_json(self, tmp_path, caplog):
        """Should handle invalid JSON gracefully."""
        config_path = tmp_path / "invalid.json"
        config_path.write_text("not valid json {{{")

        with caplog.at_level(logging.ERROR):
            result = ConfigLoader.load(config_path, SampleConfig, create_default=False)

        assert result is None
        assert "Failed to load config" in caplog.text

    def test_save_creates_file(self, tmp_path):
        """Should save config to JSON file."""
        config_path = tmp_path / "output.json"
        config = SampleConfig(name="saved", value=100)

        result = ConfigLoader.save(config_path, config)

        assert result is True
        assert config_path.exists()

        data = json.loads(config_path.read_text())
        assert data["name"] == "saved"
        assert data["value"] == 100

    def test_save_creates_directories(self, tmp_path):
        """Should create parent directories if needed."""
        config_path = tmp_path / "nested" / "deep" / "config.json"
        config = SampleConfig(name="nested", value=1)

        result = ConfigLoader.save(config_path, config)

        assert result is True
        assert config_path.exists()

    def test_load_list_loads_all_configs(self, tmp_path):
        """Should load all config files from directory."""
        (tmp_path / "config1.json").write_text('{"name": "one", "value": 1}')
        (tmp_path / "config2.json").write_text('{"name": "two", "value": 2}')
        (tmp_path / "not_json.txt").write_text("ignored")

        results = ConfigLoader.load_list(tmp_path, SampleConfig)

        assert len(results) == 2
        names = {r.name for r in results}
        assert "one" in names
        assert "two" in names

    def test_load_list_empty_directory(self, tmp_path):
        """Should return empty list for empty directory."""
        results = ConfigLoader.load_list(tmp_path, SampleConfig)
        assert results == []

    def test_load_list_missing_directory(self, tmp_path):
        """Should return empty list for missing directory."""
        results = ConfigLoader.load_list(tmp_path / "nonexistent", SampleConfig)
        assert results == []


class TestPathUtils:
    """Tests for PathUtils."""

    def test_ensure_dirs_creates_directory(self, tmp_path):
        """Should create directory."""
        new_dir = tmp_path / "new_dir"
        assert not new_dir.exists()

        PathUtils.ensure_dirs(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_dirs_creates_parent_for_file(self, tmp_path):
        """Should create parent directory for file path."""
        file_path = tmp_path / "nested" / "dir" / "file.txt"
        assert not file_path.parent.exists()

        PathUtils.ensure_dirs(file_path)

        assert file_path.parent.exists()
        assert not file_path.exists()  # File itself not created

    def test_ensure_dirs_multiple_paths(self, tmp_path):
        """Should handle multiple paths."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"

        PathUtils.ensure_dirs(dir1, dir2)

        assert dir1.exists()
        assert dir2.exists()

    def test_find_module_success(self, tmp_path):
        """Should find module in search paths."""
        module_dir = tmp_path / "mymodule"
        module_dir.mkdir()
        (module_dir / "__init__.py").touch()

        result = PathUtils.find_module("mymodule", [tmp_path])

        assert result == module_dir

    def test_find_module_not_found(self, tmp_path):
        """Should return None if module not found."""
        result = PathUtils.find_module("nonexistent", [tmp_path])
        assert result is None

    def test_find_module_no_marker(self, tmp_path):
        """Should return None if marker file missing."""
        module_dir = tmp_path / "mymodule"
        module_dir.mkdir()
        # No __init__.py

        result = PathUtils.find_module("mymodule", [tmp_path])
        assert result is None

    def test_find_module_priority_order(self, tmp_path):
        """Should return first match in priority order."""
        path1 = tmp_path / "first"
        path2 = tmp_path / "second"
        path1.mkdir()
        path2.mkdir()

        module1 = path1 / "module"
        module2 = path2 / "module"
        module1.mkdir()
        module2.mkdir()
        (module1 / "__init__.py").touch()
        (module2 / "__init__.py").touch()

        result = PathUtils.find_module("module", [path1, path2])
        assert result == module1

    def test_get_project_root_from_subdir(self, tmp_path):
        """Should find project root from subdirectory."""
        # Create anchor file
        (tmp_path / ".env").touch()
        subdir = tmp_path / "src" / "deep" / "nested"
        subdir.mkdir(parents=True)

        with patch.object(Path, "cwd", return_value=subdir):
            result = PathUtils.get_project_root(".env")

        assert result == tmp_path

    def test_get_project_root_not_found(self, tmp_path):
        """Should return None if anchor not found."""
        subdir = tmp_path / "isolated"
        subdir.mkdir()

        with patch.object(Path, "cwd", return_value=subdir):
            result = PathUtils.get_project_root(".nonexistent_anchor")

        assert result is None


class TestGateOperationResult:
    """Tests for GateOperationResult."""

    def test_ok_creates_success_result(self):
        """ok() should create successful result."""
        result = GateOperationResult.ok(
            operation="test_op",
            message="Success message",
            data={"key": "value"},
        )

        assert result.success is True
        assert result.operation == "test_op"
        assert result.message == "Success message"
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_fail_creates_failure_result(self):
        """fail() should create failed result."""
        result = GateOperationResult.fail(
            operation="test_op",
            error="Something went wrong",
            data={"partial": "data"},
        )

        assert result.success is False
        assert result.operation == "test_op"
        assert result.error == "Something went wrong"
        assert result.data == {"partial": "data"}

    def test_bool_conversion(self):
        """Should convert to bool based on success."""
        success = GateOperationResult.ok("op")
        failure = GateOperationResult.fail("op", "error")

        assert bool(success) is True
        assert bool(failure) is False

        # Should work in conditionals
        if success:
            passed = True
        else:
            passed = False
        assert passed is True

    def test_to_tuple(self):
        """to_tuple() should return legacy format."""
        success = GateOperationResult.ok("op", message="done")
        failure = GateOperationResult.fail("op", error="failed")

        assert success.to_tuple() == (True, "done")
        assert failure.to_tuple() == (False, "failed")

    def test_to_dict(self):
        """to_dict() should return serializable dict."""
        result = GateOperationResult.ok(
            operation="test_op",
            message="done",
            data=123,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["operation"] == "test_op"
        assert d["message"] == "done"
        assert d["data"] == 123
        assert d["error"] is None

        # Should be JSON serializable
        json.dumps(d)


class TestDeepUpdate:
    """Tests for deep_update utility."""

    def test_simple_update(self):
        """Should update simple keys."""
        base = {"a": 1, "b": 2}
        updates = {"b": 3, "c": 4}

        result = deep_update(base, updates)

        assert result == {"a": 1, "b": 3, "c": 4}
        assert result is base  # In-place update

    def test_nested_update(self):
        """Should deeply merge nested dicts."""
        base = {"outer": {"inner": {"a": 1, "b": 2}}}
        updates = {"outer": {"inner": {"b": 3, "c": 4}}}

        result = deep_update(base, updates)

        assert result["outer"]["inner"] == {"a": 1, "b": 3, "c": 4}

    def test_replace_non_dict_with_dict(self):
        """Should replace non-dict with dict."""
        base = {"key": "string_value"}
        updates = {"key": {"nested": True}}

        result = deep_update(base, updates)

        assert result["key"] == {"nested": True}

    def test_replace_dict_with_non_dict(self):
        """Should replace dict with non-dict."""
        base = {"key": {"nested": True}}
        updates = {"key": "string_value"}

        result = deep_update(base, updates)

        assert result["key"] == "string_value"

    def test_empty_updates(self):
        """Should handle empty updates."""
        base = {"a": 1}
        updates = {}

        result = deep_update(base, updates)

        assert result == {"a": 1}

    def test_deeply_nested_update(self):
        """Should handle deeply nested structures."""
        base = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "original"
                    }
                }
            }
        }
        updates = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "updated",
                        "new_key": "added"
                    }
                }
            }
        }

        result = deep_update(base, updates)

        assert result["level1"]["level2"]["level3"]["value"] == "updated"
        assert result["level1"]["level2"]["level3"]["new_key"] == "added"

    def test_list_replacement(self):
        """Lists should be replaced, not merged."""
        base = {"items": [1, 2, 3]}
        updates = {"items": [4, 5]}

        result = deep_update(base, updates)

        assert result["items"] == [4, 5]
