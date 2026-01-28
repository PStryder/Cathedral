"""
FileSystemGate security module.

Provides path validation, traversal prevention, and extension checking.
"""

import os
import re
from typing import Tuple, Optional, List

from .models import FolderConfig, FolderPermission


class PathSecurityError(Exception):
    """Raised when a path fails security validation."""
    pass


class PermissionError(Exception):
    """Raised when an operation is not permitted."""
    pass


def normalize_path(path: str) -> str:
    """
    Normalize a path to prevent traversal attacks.

    Args:
        path: Raw path string

    Returns:
        Normalized absolute path
    """
    # Expand user home directory if present
    path = os.path.expanduser(path)
    # Normalize the path (resolve . and ..)
    path = os.path.normpath(path)
    # Convert to absolute
    path = os.path.abspath(path)
    return path


def validate_path_within_folder(
    target_path: str,
    folder: FolderConfig
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate that a target path is within the configured folder.

    Args:
        target_path: Path to validate
        folder: Folder configuration

    Returns:
        Tuple of (is_valid, resolved_path, error_message)
    """
    folder_path = normalize_path(folder.path)
    target = normalize_path(target_path)

    # Check if target is within folder
    try:
        # Use os.path.commonpath to check containment
        common = os.path.commonpath([folder_path, target])
        if common != folder_path:
            return False, target, f"Path escapes folder boundary: {target}"
    except ValueError:
        # Different drives on Windows
        return False, target, f"Path is on different drive: {target}"

    # Check for recursive access
    if not folder.recursive:
        # Only allow files directly in the folder, not subdirectories
        rel_path = os.path.relpath(target, folder_path)
        if os.sep in rel_path or "/" in rel_path:
            return False, target, "Subdirectory access not allowed for this folder"

    return True, target, None


def check_extension_allowed(
    filename: str,
    folder: FolderConfig
) -> Tuple[bool, Optional[str]]:
    """
    Check if a file extension is allowed for the folder.

    Args:
        filename: Filename to check
        folder: Folder configuration

    Returns:
        Tuple of (is_allowed, error_message)
    """
    # Get extension (including the dot)
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if not ext:
        # No extension - generally allowed unless blocked
        return True, None

    # Check blocklist first
    if ext in [e.lower() for e in folder.blocked_extensions]:
        return False, f"Extension {ext} is blocked for security"

    # Check allowlist if set
    if folder.allowed_extensions is not None:
        allowed = [e.lower() for e in folder.allowed_extensions]
        if ext not in allowed:
            return False, f"Extension {ext} is not in allowed list"

    return True, None


def check_file_size(
    size_bytes: int,
    folder: FolderConfig
) -> Tuple[bool, Optional[str]]:
    """
    Check if a file size is within limits.

    Args:
        size_bytes: File size in bytes
        folder: Folder configuration

    Returns:
        Tuple of (is_allowed, error_message)
    """
    max_bytes = folder.max_file_size_mb * 1024 * 1024

    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        return False, f"File size ({size_mb:.2f}MB) exceeds limit ({folder.max_file_size_mb}MB)"

    return True, None


def validate_read_access(
    target_path: str,
    folder: FolderConfig
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate read access to a path.

    Args:
        target_path: Path to read
        folder: Folder configuration

    Returns:
        Tuple of (is_valid, resolved_path, error_message)
    """
    # Check permission
    if not folder.can_read():
        return False, "", f"Folder '{folder.id}' does not allow read access"

    # Validate path is within folder
    is_valid, resolved, error = validate_path_within_folder(target_path, folder)
    if not is_valid:
        return False, resolved, error

    # Check if path exists
    if not os.path.exists(resolved):
        return False, resolved, f"Path does not exist: {resolved}"

    return True, resolved, None


def validate_write_access(
    target_path: str,
    folder: FolderConfig,
    content_size: int = 0
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate write access to a path.

    Args:
        target_path: Path to write
        folder: Folder configuration
        content_size: Size of content to write

    Returns:
        Tuple of (is_valid, resolved_path, error_message)
    """
    # Check permission
    if not folder.can_write():
        return False, "", f"Folder '{folder.id}' does not allow write access"

    # Validate path is within folder
    is_valid, resolved, error = validate_path_within_folder(target_path, folder)
    if not is_valid:
        return False, resolved, error

    # Check extension
    if os.path.isfile(resolved) or not os.path.exists(resolved):
        basename = os.path.basename(resolved)
        ext_ok, ext_error = check_extension_allowed(basename, folder)
        if not ext_ok:
            return False, resolved, ext_error

    # Check file size
    if content_size > 0:
        size_ok, size_error = check_file_size(content_size, folder)
        if not size_ok:
            return False, resolved, size_error

    return True, resolved, None


def validate_delete_access(
    target_path: str,
    folder: FolderConfig
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate delete access to a path.

    Args:
        target_path: Path to delete
        folder: Folder configuration

    Returns:
        Tuple of (is_valid, resolved_path, error_message)
    """
    # Check permission (requires write access)
    if not folder.can_write():
        return False, "", f"Folder '{folder.id}' does not allow delete operations"

    # Validate path is within folder
    is_valid, resolved, error = validate_path_within_folder(target_path, folder)
    if not is_valid:
        return False, resolved, error

    # Cannot delete the folder root itself
    folder_path = normalize_path(folder.path)
    if resolved == folder_path:
        return False, resolved, "Cannot delete the folder root"

    # Check if path exists
    if not os.path.exists(resolved):
        return False, resolved, f"Path does not exist: {resolved}"

    return True, resolved, None


def resolve_folder_path(
    folder: FolderConfig,
    relative_path: str
) -> str:
    """
    Resolve a path relative to a folder.

    Args:
        folder: Folder configuration
        relative_path: Path relative to folder root

    Returns:
        Absolute path
    """
    folder_path = normalize_path(folder.path)

    # Handle empty or root path
    if not relative_path or relative_path in (".", "/", "\\"):
        return folder_path

    # Strip leading slashes
    relative_path = relative_path.lstrip("/\\")

    # Join and normalize
    return normalize_path(os.path.join(folder_path, relative_path))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to remove dangerous characters.

    Args:
        filename: Raw filename

    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = os.path.basename(filename)

    # Remove null bytes and other control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)

    # Remove other dangerous characters
    filename = re.sub(r'[<>:"|?*]', '_', filename)

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        max_name_len = 255 - len(ext)
        filename = name[:max_name_len] + ext

    return filename
