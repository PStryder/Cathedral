"""
ShellGate security module.

Provides command validation, blocklist checking, and environment sanitization.
"""

import os
import re
import shlex
from typing import Tuple, Optional, List, Dict, Any

from .models import CommandConfig


class CommandSecurityError(Exception):
    """Raised when a command fails security validation."""
    pass


def validate_command(
    command: str,
    config: CommandConfig
) -> Tuple[bool, Optional[str]]:
    """
    Validate a command against security rules.

    Args:
        command: Command string to validate
        config: Command configuration with security rules

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command or not command.strip():
        return False, "Empty command"

    command = command.strip()

    # Check against blocked commands (exact match)
    for blocked in config.blocked_commands:
        if command == blocked or command.startswith(blocked + " "):
            return False, f"Command blocked: {blocked}"

    # Check against blocked patterns (regex)
    for pattern in config.blocked_patterns:
        try:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command matches blocked pattern"
        except re.error:
            # Invalid regex, skip
            continue

    # Check allowlist if configured
    if config.allowed_commands:
        # Extract base command
        try:
            parts = shlex.split(command)
            base_cmd = parts[0] if parts else ""
        except ValueError:
            base_cmd = command.split()[0] if command.split() else ""

        if base_cmd not in config.allowed_commands:
            return False, f"Command not in allowlist: {base_cmd}"

    return True, None


def sanitize_environment(
    env: Optional[Dict[str, str]] = None,
    blocklist: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create a sanitized copy of environment variables.

    Args:
        env: Environment dict (default: os.environ)
        blocklist: Keys to remove

    Returns:
        Sanitized environment dict
    """
    if env is None:
        env = dict(os.environ)
    else:
        env = dict(env)

    if blocklist is None:
        blocklist = []

    # Remove blocked keys
    for key in blocklist:
        env.pop(key, None)

    # Also remove any key containing common secret patterns
    secret_patterns = [
        "SECRET",
        "PASSWORD",
        "PRIVATE_KEY",
        "API_KEY",
        "ACCESS_TOKEN",
        "AUTH_TOKEN",
        "CREDENTIALS",
    ]

    keys_to_remove = []
    for key in env:
        key_upper = key.upper()
        for pattern in secret_patterns:
            if pattern in key_upper:
                keys_to_remove.append(key)
                break

    for key in keys_to_remove:
        env.pop(key, None)

    return env


def validate_working_directory(
    working_dir: str,
    allowed_dirs: Optional[List[str]] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate a working directory.

    Args:
        working_dir: Directory path
        allowed_dirs: Optional list of allowed base directories

    Returns:
        Tuple of (is_valid, resolved_path, error_message)
    """
    # Expand and normalize
    resolved = os.path.abspath(os.path.expanduser(working_dir))

    # Check exists
    if not os.path.exists(resolved):
        return False, resolved, f"Directory does not exist: {resolved}"

    # Check is directory
    if not os.path.isdir(resolved):
        return False, resolved, f"Path is not a directory: {resolved}"

    # Check allowed directories if specified
    if allowed_dirs:
        is_allowed = False
        for allowed in allowed_dirs:
            allowed_resolved = os.path.abspath(os.path.expanduser(allowed))
            if resolved.startswith(allowed_resolved):
                is_allowed = True
                break

        if not is_allowed:
            return False, resolved, "Working directory not in allowed paths"

    return True, resolved, None


def check_dangerous_constructs(command: str) -> List[str]:
    """
    Check for potentially dangerous shell constructs.

    Args:
        command: Command string

    Returns:
        List of warnings (empty if none)
    """
    warnings = []

    # Check for shell operators that could be dangerous
    dangerous_patterns = [
        (r"\|.*rm\b", "Piping to rm command"),
        (r">\s*/etc/", "Writing to /etc"),
        (r">\s*/var/", "Writing to /var"),
        (r">\s*/usr/", "Writing to /usr"),
        (r">\s*/bin/", "Writing to /bin"),
        (r">\s*/sbin/", "Writing to /sbin"),
        (r"curl.*\|\s*(bash|sh)", "Piping curl to shell"),
        (r"wget.*\|\s*(bash|sh)", "Piping wget to shell"),
        (r"eval\s+", "Using eval"),
        (r"\$\(.*\)", "Command substitution"),
        (r"`.*`", "Backtick command substitution"),
    ]

    for pattern, description in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            warnings.append(description)

    return warnings


def parse_command_for_display(command: str) -> Dict[str, str]:
    """
    Parse command for safe display.

    Args:
        command: Raw command string

    Returns:
        Dict with base_command, args, and full (sanitized)
    """
    try:
        parts = shlex.split(command)
        base = parts[0] if parts else ""
        args = " ".join(parts[1:]) if len(parts) > 1 else ""
    except ValueError:
        # Failed to parse, fall back to simple split
        parts = command.split()
        base = parts[0] if parts else ""
        args = " ".join(parts[1:]) if len(parts) > 1 else ""

    return {
        "base_command": base,
        "args": args,
        "full": command
    }


def estimate_command_risk(command: str, config: CommandConfig) -> Dict[str, Any]:
    """
    Estimate the risk level of a command.

    Args:
        command: Command string
        config: Command configuration

    Returns:
        Dict with risk_level (low/medium/high), warnings, and is_blocked
    """
    # Check if blocked
    is_valid, error = validate_command(command, config)
    if not is_valid:
        return {
            "risk_level": "blocked",
            "is_blocked": True,
            "block_reason": error,
            "warnings": []
        }

    # Get warnings
    warnings = check_dangerous_constructs(command)

    # Determine risk level
    if len(warnings) >= 3:
        risk_level = "high"
    elif len(warnings) >= 1:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Specific high-risk commands
    high_risk_bases = ["sudo", "su", "chmod", "chown", "kill", "pkill", "killall"]
    parsed = parse_command_for_display(command)
    if parsed["base_command"] in high_risk_bases:
        if risk_level == "low":
            risk_level = "medium"

    return {
        "risk_level": risk_level,
        "is_blocked": False,
        "block_reason": None,
        "warnings": warnings
    }
