"""
Shared Gate utilities for Cathedral.

Provides consolidated patterns for all Gate implementations:
- GateLogger: Unified logging with Python's logging module
- GateInitializer: Base class for initialization patterns
- GateErrorHandler: Decorator for error handling
- GateHealth: Protocol for health checks
- ConfigLoader: Unified JSON config loading/saving
- PathUtils: Common path operations
- GateOperationResult: Unified result type
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)


# =============================================================================
# GateLogger - Unified logging for all Gates
# =============================================================================


class GateLogger:
    """
    Unified logging for all Gates.

    Replaces print() statements with proper Python logging.
    Each gate gets its own namespaced logger.
    """

    _loggers: Dict[str, logging.Logger] = {}
    _configured = False

    @classmethod
    def _ensure_configured(cls):
        """Ensure basic logging is configured."""
        if cls._configured:
            return

        # Configure root cathedral logger if not already configured
        root_logger = logging.getLogger("cathedral")
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)

        cls._configured = True

    @classmethod
    def get(cls, gate_name: str) -> logging.Logger:
        """
        Get a logger for a specific gate.

        Args:
            gate_name: Name of the gate (e.g., "MemoryGate", "FileSystemGate")

        Returns:
            Logger instance for the gate
        """
        cls._ensure_configured()

        logger_name = f"cathedral.{gate_name}"
        if logger_name not in cls._loggers:
            cls._loggers[logger_name] = logging.getLogger(logger_name)

        return cls._loggers[logger_name]

    @classmethod
    def set_level(cls, level: int, gate_name: Optional[str] = None):
        """
        Set logging level.

        Args:
            level: Logging level (e.g., logging.DEBUG)
            gate_name: Specific gate to set level for, or None for all
        """
        if gate_name:
            cls.get(gate_name).setLevel(level)
        else:
            logging.getLogger("cathedral").setLevel(level)


# =============================================================================
# GateErrorHandler - Unified error handling
# =============================================================================


class GateErrorHandler:
    """
    Unified error handling for Gate operations.

    Provides decorators and utility methods for consistent error handling
    across all Gate implementations.
    """

    @staticmethod
    def handle(
        gate_name: str,
        operation: str,
        exception: Exception,
        default_return: Any = None,
        log_level: int = logging.ERROR,
    ) -> Any:
        """
        Handle and log a gate operation error.

        Args:
            gate_name: Name of the gate
            operation: Operation that failed
            exception: The exception that occurred
            default_return: Value to return on error
            log_level: Logging level to use

        Returns:
            The default_return value
        """
        logger = GateLogger.get(gate_name)
        logger.log(log_level, f"{operation} failed: {exception}")
        return default_return

    @staticmethod
    def wrap(
        gate_name: str,
        operation: str,
        default_return: Any = None,
        log_level: int = logging.ERROR,
        reraise: bool = False,
    ):
        """
        Decorator for wrapping gate operations with error handling.

        Args:
            gate_name: Name of the gate
            operation: Operation name for logging
            default_return: Value to return on error
            log_level: Logging level to use
            reraise: Whether to re-raise the exception after logging

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    GateErrorHandler.handle(
                        gate_name, operation, e, default_return, log_level
                    )
                    if reraise:
                        raise
                    return default_return
            return wrapper
        return decorator

    @staticmethod
    def wrap_async(
        gate_name: str,
        operation: str,
        default_return: Any = None,
        log_level: int = logging.ERROR,
        reraise: bool = False,
    ):
        """
        Decorator for wrapping async gate operations with error handling.

        Args:
            gate_name: Name of the gate
            operation: Operation name for logging
            default_return: Value to return on error
            log_level: Logging level to use
            reraise: Whether to re-raise the exception after logging

        Returns:
            Decorator function for async functions
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    GateErrorHandler.handle(
                        gate_name, operation, e, default_return, log_level
                    )
                    if reraise:
                        raise
                    return default_return
            return wrapper
        return decorator


# =============================================================================
# GateHealth - Protocol for health checks
# =============================================================================


@runtime_checkable
class GateHealth(Protocol):
    """
    Protocol for gate health checking.

    All Gates should implement this protocol to provide
    consistent health monitoring capabilities.
    """

    @classmethod
    def is_healthy(cls) -> bool:
        """Check if the gate is operational."""
        ...

    @classmethod
    def get_health_status(cls) -> Dict[str, Any]:
        """
        Get detailed health information.

        Returns:
            Dict with health details including:
            - healthy: bool
            - initialized: bool
            - dependencies: List[str]
            - details: Dict[str, Any]
        """
        ...

    @classmethod
    def get_dependencies(cls) -> List[str]:
        """List external dependencies (e.g., database, API keys)."""
        ...


def build_health_status(
    gate_name: str,
    initialized: bool,
    dependencies: List[str],
    checks: Dict[str, bool],
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a standardized health status dict.

    Args:
        gate_name: Name of the gate
        initialized: Whether the gate is initialized
        dependencies: List of dependency names
        checks: Dict of check name -> passed
        details: Additional details

    Returns:
        Standardized health status dict
    """
    all_checks_passed = all(checks.values()) if checks else True

    return {
        "gate": gate_name,
        "healthy": initialized and all_checks_passed,
        "initialized": initialized,
        "dependencies": dependencies,
        "checks": checks,
        "details": details or {},
    }


# =============================================================================
# ConfigLoader - Unified JSON config loading
# =============================================================================


ConfigT = TypeVar("ConfigT")


class ConfigLoader:
    """
    Unified configuration loading and saving.

    Provides consistent config file handling across all Gates.
    """

    @staticmethod
    def load(
        path: Union[str, Path],
        model_class: Type[ConfigT],
        create_default: bool = True,
    ) -> Optional[ConfigT]:
        """
        Load a JSON config file into a model class.

        Args:
            path: Path to the config file
            model_class: Class with from_dict() method
            create_default: If True and file doesn't exist, return model_class()

        Returns:
            Instance of model_class or None if file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            if create_default:
                return model_class()
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if hasattr(model_class, "from_dict"):
                return model_class.from_dict(data)
            elif hasattr(model_class, "model_validate"):
                return model_class.model_validate(data)
            else:
                return model_class(**data)

        except (json.JSONDecodeError, Exception) as e:
            logger = GateLogger.get("ConfigLoader")
            logger.error(f"Failed to load config from {path}: {e}")
            return None

    @staticmethod
    def save(
        path: Union[str, Path],
        config: Any,
        create_dirs: bool = True,
    ) -> bool:
        """
        Save a config object to a JSON file.

        Args:
            path: Path to save the config
            config: Config object with to_dict() or model_dump() method
            create_dirs: Create parent directories if needed

        Returns:
            True if successful
        """
        path = Path(path)

        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            if hasattr(config, "to_dict"):
                data = config.to_dict()
            elif hasattr(config, "model_dump"):
                data = config.model_dump(mode="json")
            else:
                data = dict(config)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            return True

        except Exception as e:
            logger = GateLogger.get("ConfigLoader")
            logger.error(f"Failed to save config to {path}: {e}")
            return False

    @staticmethod
    def load_list(
        directory: Union[str, Path],
        model_class: Type[ConfigT],
        pattern: str = "*.json",
    ) -> List[ConfigT]:
        """
        Load all JSON configs from a directory.

        Args:
            directory: Directory to scan
            model_class: Class to instantiate for each file
            pattern: Glob pattern for files

        Returns:
            List of loaded config objects
        """
        directory = Path(directory)
        results = []

        if not directory.exists():
            return results

        for file_path in directory.glob(pattern):
            config = ConfigLoader.load(file_path, model_class, create_default=False)
            if config:
                results.append(config)

        return results


# =============================================================================
# PathUtils - Common path operations
# =============================================================================


class PathUtils:
    """Common path utilities for Gates."""

    @staticmethod
    def ensure_dirs(*paths: Union[str, Path]) -> None:
        """
        Ensure directories exist for the given paths.

        For file paths, creates the parent directory.
        For directory paths, creates the directory.
        """
        for path in paths:
            path = Path(path)
            if path.suffix:
                # Looks like a file path, create parent
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Looks like a directory, create it
                path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def find_module(
        module_name: str,
        search_paths: List[Path],
        marker_file: str = "__init__.py",
    ) -> Optional[Path]:
        """
        Find an external module by searching candidate paths.

        Args:
            module_name: Name of the module to find
            search_paths: Paths to search (in priority order)
            marker_file: File or directory that must exist to confirm module

        Returns:
            Path to the module or None if not found
        """
        for base_path in search_paths:
            if base_path is None:
                continue

            candidate = base_path / module_name if module_name else base_path

            if not candidate.exists():
                continue

            marker = candidate / marker_file
            if marker.exists():
                return candidate

        return None

    @staticmethod
    def get_project_root(anchor_file: str = ".env") -> Optional[Path]:
        """
        Find project root by searching up for an anchor file.

        Args:
            anchor_file: File that marks the project root

        Returns:
            Path to project root or None
        """
        current = Path.cwd()

        for parent in [current, *current.parents]:
            if (parent / anchor_file).exists():
                return parent

        return None


# =============================================================================
# GateOperationResult - Unified result type
# =============================================================================


@dataclass
class GateOperationResult:
    """
    Unified operation result for all Gates.

    Provides consistent result handling across Gate operations.
    """

    success: bool
    operation: str
    message: str = ""
    data: Any = None
    error: Optional[str] = None

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success

    def to_tuple(self) -> tuple[bool, str]:
        """Convert to legacy (success, message) tuple."""
        return (self.success, self.error or self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "success": self.success,
            "operation": self.operation,
            "message": self.message,
            "data": self.data,
            "error": self.error,
        }

    @classmethod
    def ok(
        cls,
        operation: str,
        message: str = "",
        data: Any = None,
    ) -> "GateOperationResult":
        """Create a successful result."""
        return cls(
            success=True,
            operation=operation,
            message=message,
            data=data,
        )

    @classmethod
    def fail(
        cls,
        operation: str,
        error: str,
        data: Any = None,
    ) -> "GateOperationResult":
        """Create a failed result."""
        return cls(
            success=False,
            operation=operation,
            message="",
            data=data,
            error=error,
        )


# =============================================================================
# Deep update utility
# =============================================================================


def deep_update(base: dict, updates: dict) -> dict:
    """
    Deep merge updates into base dict (in-place).

    Args:
        base: Base dictionary to update
        updates: Updates to apply

    Returns:
        The updated base dict
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


# =============================================================================
# Convenience exports
# =============================================================================


def get_logger(gate_name: str) -> logging.Logger:
    """Shortcut for GateLogger.get()."""
    return GateLogger.get(gate_name)
