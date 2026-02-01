"""
ToolGate Policy Manager.

Enforces security policies for tool execution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from cathedral.shared.gate import GateLogger
from cathedral.ToolGate.models import PolicyClass, ToolDefinition

_log = GateLogger.get("ToolGate")


class PolicyManager:
    """
    Manages security policies for tool execution.

    Integrates with Cathedral's SecurityManager and Gate-specific security.
    """

    def __init__(self):
        # Default: only read-only operations allowed
        self._enabled_policies: Set[PolicyClass] = {PolicyClass.READ_ONLY}

    def enable_policy(self, policy: PolicyClass) -> Tuple[bool, Optional[str]]:
        """
        Enable a policy class.

        Args:
            policy: Policy class to enable

        Returns:
            Tuple of (success, error_message)
        """
        # Check if we can enable this policy
        if policy in (PolicyClass.WRITE, PolicyClass.DESTRUCTIVE):
            if not self._check_write_access():
                return False, "Session must be unlocked for write operations"

        if policy == PolicyClass.PRIVILEGED:
            if not self._check_privileged_access():
                return False, "Privileged access requires explicit unlock"

        if policy == PolicyClass.NETWORK:
            # Network access is generally allowed, but could be restricted
            pass

        self._enabled_policies.add(policy)
        _log.info(f"Policy enabled: {policy.value}")
        return True, None

    def disable_policy(self, policy: PolicyClass) -> None:
        """Disable a policy class."""
        if policy == PolicyClass.READ_ONLY:
            # Can't disable read-only
            return
        self._enabled_policies.discard(policy)
        _log.info(f"Policy disabled: {policy.value}")

    def get_enabled_policies(self) -> Set[PolicyClass]:
        """Get currently enabled policies."""
        return self._enabled_policies.copy()

    def is_policy_enabled(self, policy: PolicyClass) -> bool:
        """Check if a policy is enabled."""
        return policy in self._enabled_policies

    def check(
        self,
        tool: ToolDefinition,
        args: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if tool execution is allowed under current policy.

        Args:
            tool: Tool definition
            args: Tool arguments

        Returns:
            Tuple of (allowed, reason)
        """
        # Check policy class
        if tool.policy_class not in self._enabled_policies:
            return False, f"Policy class '{tool.policy_class.value}' not enabled"

        # Gate-specific security checks
        if tool.gate == "ShellGate":
            allowed, reason = self._check_shell_policy(tool, args)
            if not allowed:
                return False, reason

        if tool.gate == "FileSystemGate":
            allowed, reason = self._check_filesystem_policy(tool, args)
            if not allowed:
                return False, reason

        if tool.gate == "BrowserGate":
            allowed, reason = self._check_network_policy(tool, args)
            if not allowed:
                return False, reason

        return True, None

    def _check_write_access(self) -> bool:
        """Check if write access is available."""
        try:
            from cathedral import SecurityManager
            # If security is initialized and session is unlocked, allow writes
            if SecurityManager.is_initialized():
                return not SecurityManager.is_locked()
            # If security not configured, allow by default
            return True
        except ImportError:
            return True

    def _check_privileged_access(self) -> bool:
        """Check if privileged access is available."""
        try:
            from cathedral import SecurityManager
            if SecurityManager.is_initialized():
                # Privileged access requires explicit unlock
                return not SecurityManager.is_locked()
            return True
        except ImportError:
            return True

    def _check_shell_policy(
        self,
        tool: ToolDefinition,
        args: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Check ShellGate-specific security."""
        if "command" not in args:
            return True, None

        command = args["command"]

        try:
            from cathedral import ShellGate

            # Use ShellGate's built-in validation
            valid, error = ShellGate.validate_command(command)
            if not valid:
                return False, f"Command blocked: {error}"

            # Check for high-risk patterns
            risk = ShellGate.estimate_risk(command)
            if risk.get("level") == "critical":
                return False, f"Command too risky: {risk.get('reason', 'unknown')}"

        except ImportError:
            _log.warning("ShellGate not available for policy check")

        return True, None

    def _check_filesystem_policy(
        self,
        tool: ToolDefinition,
        args: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Check FileSystemGate-specific security."""
        folder_id = args.get("folder_id")

        if not folder_id:
            return True, None

        try:
            from cathedral import FileSystemGate

            folder = FileSystemGate.get_folder(folder_id)
            if not folder:
                return False, f"Folder not found: {folder_id}"

            # Check folder permission matches operation
            if tool.method in ("write_file", "mkdir", "delete", "copy", "move"):
                from cathedral.FileSystemGate.models import FolderPermission
                if folder.permission == FolderPermission.READ_ONLY:
                    return False, f"Folder {folder_id} is read-only"

        except ImportError:
            _log.warning("FileSystemGate not available for policy check")

        return True, None

    def _check_network_policy(
        self,
        tool: ToolDefinition,
        args: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Check network access policy."""
        url = args.get("url")

        if not url:
            return True, None

        # Parse URL to extract hostname (not substring match on full URL)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            hostname_lower = hostname.lower()
        except Exception:
            # If we can't parse, allow (fail open for weird URLs)
            return True, None

        # Block localhost/internal networks for security
        # Check hostname only, not full URL (avoids false positives on query params)
        blocked_hostnames = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        if hostname_lower in blocked_hostnames:
            return False, f"Access to internal networks blocked: {hostname}"

        # Check private IP ranges by prefix
        private_prefixes = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                           "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                           "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                           "172.29.", "172.30.", "172.31.")
        if hostname_lower.startswith(private_prefixes):
            return False, f"Access to internal networks blocked: {hostname}"

        return True, None

    def reset(self) -> None:
        """Reset to default policies."""
        self._enabled_policies = {PolicyClass.READ_ONLY}


# Global policy manager instance
_policy_manager: Optional[PolicyManager] = None


def get_policy_manager() -> PolicyManager:
    """Get or create the global policy manager."""
    global _policy_manager
    if _policy_manager is None:
        _policy_manager = PolicyManager()
    return _policy_manager


def reset_policy_manager() -> None:
    """Reset the global policy manager."""
    global _policy_manager
    _policy_manager = None


__all__ = [
    "PolicyManager",
    "get_policy_manager",
    "reset_policy_manager",
]
