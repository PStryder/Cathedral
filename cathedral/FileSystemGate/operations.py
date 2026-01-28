"""
FileSystemGate file operations.

Provides read, write, list, mkdir, and delete operations with security checks.
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional, Tuple

from .models import FolderConfig, FileInfo, OperationResult
from .security import (
    validate_read_access,
    validate_write_access,
    validate_delete_access,
    resolve_folder_path,
    normalize_path,
    sanitize_filename,
)


def list_directory(
    folder: FolderConfig,
    relative_path: str = "",
    show_hidden: bool = False
) -> OperationResult:
    """
    List contents of a directory.

    Args:
        folder: Folder configuration
        relative_path: Path relative to folder root
        show_hidden: Include hidden files (starting with .)

    Returns:
        OperationResult with list of FileInfo in data
    """
    target_path = resolve_folder_path(folder, relative_path)

    # Validate access
    is_valid, resolved, error = validate_read_access(target_path, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="list",
            path=target_path,
            error=error
        )

    # Check if it's a directory
    if not os.path.isdir(resolved):
        return OperationResult(
            success=False,
            operation="list",
            path=resolved,
            error="Path is not a directory"
        )

    # List directory contents
    try:
        entries = os.listdir(resolved)
        files: List[FileInfo] = []

        for entry in entries:
            # Skip hidden files if not requested
            if not show_hidden and entry.startswith('.'):
                continue

            entry_path = os.path.join(resolved, entry)
            rel_path = os.path.relpath(entry_path, normalize_path(folder.path))

            try:
                stat = os.stat(entry_path)
                is_dir = os.path.isdir(entry_path)

                file_info = FileInfo(
                    name=entry,
                    path=rel_path,
                    absolute_path=entry_path,
                    is_directory=is_dir,
                    size_bytes=0 if is_dir else stat.st_size,
                    extension=None if is_dir else os.path.splitext(entry)[1],
                    modified_at=datetime.fromtimestamp(stat.st_mtime),
                    created_at=datetime.fromtimestamp(stat.st_ctime),
                    is_readable=folder.can_read(),
                    is_writable=folder.can_write()
                )
                files.append(file_info)
            except OSError:
                # Skip files we can't stat
                continue

        # Sort: directories first, then by name
        files.sort(key=lambda f: (not f.is_directory, f.name.lower()))

        return OperationResult(
            success=True,
            operation="list",
            path=resolved,
            message=f"Listed {len(files)} items",
            data=[f.to_dict() for f in files]
        )

    except OSError as e:
        return OperationResult(
            success=False,
            operation="list",
            path=resolved,
            error=f"Failed to list directory: {e}"
        )


def read_file(
    folder: FolderConfig,
    relative_path: str,
    encoding: str = "utf-8",
    binary: bool = False
) -> OperationResult:
    """
    Read a file's contents.

    Args:
        folder: Folder configuration
        relative_path: Path relative to folder root
        encoding: Text encoding (ignored if binary=True)
        binary: Read as binary data

    Returns:
        OperationResult with file contents in data
    """
    target_path = resolve_folder_path(folder, relative_path)

    # Validate access
    is_valid, resolved, error = validate_read_access(target_path, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="read",
            path=target_path,
            error=error
        )

    # Check if it's a file
    if os.path.isdir(resolved):
        return OperationResult(
            success=False,
            operation="read",
            path=resolved,
            error="Path is a directory, not a file"
        )

    # Check file size
    try:
        size = os.path.getsize(resolved)
        max_bytes = folder.max_file_size_mb * 1024 * 1024
        if size > max_bytes:
            return OperationResult(
                success=False,
                operation="read",
                path=resolved,
                error=f"File too large ({size / (1024*1024):.2f}MB > {folder.max_file_size_mb}MB)"
            )
    except OSError as e:
        return OperationResult(
            success=False,
            operation="read",
            path=resolved,
            error=f"Cannot check file size: {e}"
        )

    # Read the file
    try:
        mode = "rb" if binary else "r"
        with open(resolved, mode, encoding=None if binary else encoding) as f:
            content = f.read()

        return OperationResult(
            success=True,
            operation="read",
            path=resolved,
            message=f"Read {len(content)} {'bytes' if binary else 'characters'}",
            data=content if binary else content
        )

    except UnicodeDecodeError:
        return OperationResult(
            success=False,
            operation="read",
            path=resolved,
            error=f"Cannot decode file with {encoding} encoding. Try binary mode."
        )
    except OSError as e:
        return OperationResult(
            success=False,
            operation="read",
            path=resolved,
            error=f"Failed to read file: {e}"
        )


def write_file(
    folder: FolderConfig,
    relative_path: str,
    content: str | bytes,
    encoding: str = "utf-8",
    create_dirs: bool = False
) -> Tuple[OperationResult, Optional[str]]:
    """
    Write content to a file.

    Args:
        folder: Folder configuration
        relative_path: Path relative to folder root
        content: Content to write (str or bytes)
        encoding: Text encoding (ignored if content is bytes)
        create_dirs: Create parent directories if they don't exist

    Returns:
        Tuple of (OperationResult, backup_path if backup was created)
    """
    target_path = resolve_folder_path(folder, relative_path)

    # Calculate content size
    if isinstance(content, str):
        content_bytes = content.encode(encoding)
        content_size = len(content_bytes)
    else:
        content_size = len(content)

    # Validate access
    is_valid, resolved, error = validate_write_access(target_path, folder, content_size)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="write",
            path=target_path,
            error=error
        ), None

    # Create parent directories if requested
    parent_dir = os.path.dirname(resolved)
    if create_dirs and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except OSError as e:
            return OperationResult(
                success=False,
                operation="write",
                path=resolved,
                error=f"Failed to create parent directories: {e}"
            ), None

    # Check if file exists (for backup tracking)
    file_existed = os.path.exists(resolved)
    backup_path = None

    # Write the file
    try:
        if isinstance(content, bytes):
            mode = "wb"
            with open(resolved, mode) as f:
                f.write(content)
        else:
            mode = "w"
            with open(resolved, mode, encoding=encoding) as f:
                f.write(content)

        return OperationResult(
            success=True,
            operation="write",
            path=resolved,
            message=f"{'Updated' if file_existed else 'Created'} file ({content_size} bytes)"
        ), backup_path

    except OSError as e:
        return OperationResult(
            success=False,
            operation="write",
            path=resolved,
            error=f"Failed to write file: {e}"
        ), None


def make_directory(
    folder: FolderConfig,
    relative_path: str,
    parents: bool = False
) -> OperationResult:
    """
    Create a directory.

    Args:
        folder: Folder configuration
        relative_path: Path relative to folder root
        parents: Create parent directories if needed

    Returns:
        OperationResult
    """
    target_path = resolve_folder_path(folder, relative_path)

    # Validate access (use write validation)
    is_valid, resolved, error = validate_write_access(target_path, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="mkdir",
            path=target_path,
            error=error
        )

    # Check if already exists
    if os.path.exists(resolved):
        if os.path.isdir(resolved):
            return OperationResult(
                success=True,
                operation="mkdir",
                path=resolved,
                message="Directory already exists"
            )
        else:
            return OperationResult(
                success=False,
                operation="mkdir",
                path=resolved,
                error="A file with that name already exists"
            )

    # Create the directory
    try:
        if parents:
            os.makedirs(resolved, exist_ok=True)
        else:
            os.mkdir(resolved)

        return OperationResult(
            success=True,
            operation="mkdir",
            path=resolved,
            message="Directory created"
        )

    except OSError as e:
        return OperationResult(
            success=False,
            operation="mkdir",
            path=resolved,
            error=f"Failed to create directory: {e}"
        )


def delete_path(
    folder: FolderConfig,
    relative_path: str,
    recursive: bool = False
) -> Tuple[OperationResult, Optional[str]]:
    """
    Delete a file or directory.

    Args:
        folder: Folder configuration
        relative_path: Path relative to folder root
        recursive: If True, delete directories recursively

    Returns:
        Tuple of (OperationResult, backup_path if backup was created)
    """
    target_path = resolve_folder_path(folder, relative_path)

    # Validate access
    is_valid, resolved, error = validate_delete_access(target_path, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="delete",
            path=target_path,
            error=error
        ), None

    backup_path = None
    is_dir = os.path.isdir(resolved)

    try:
        if is_dir:
            if recursive:
                shutil.rmtree(resolved)
            else:
                os.rmdir(resolved)
            message = "Directory deleted"
        else:
            os.remove(resolved)
            message = "File deleted"

        return OperationResult(
            success=True,
            operation="delete",
            path=resolved,
            message=message
        ), backup_path

    except OSError as e:
        error_msg = str(e)
        if "not empty" in error_msg.lower() or "directory not empty" in error_msg.lower():
            error_msg = "Directory is not empty. Use recursive=True to delete."

        return OperationResult(
            success=False,
            operation="delete",
            path=resolved,
            error=f"Failed to delete: {error_msg}"
        ), None


def get_file_info(
    folder: FolderConfig,
    relative_path: str
) -> OperationResult:
    """
    Get detailed information about a file or directory.

    Args:
        folder: Folder configuration
        relative_path: Path relative to folder root

    Returns:
        OperationResult with FileInfo in data
    """
    target_path = resolve_folder_path(folder, relative_path)

    # Validate access
    is_valid, resolved, error = validate_read_access(target_path, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="info",
            path=target_path,
            error=error
        )

    try:
        stat = os.stat(resolved)
        is_dir = os.path.isdir(resolved)
        name = os.path.basename(resolved)

        file_info = FileInfo(
            name=name,
            path=os.path.relpath(resolved, normalize_path(folder.path)),
            absolute_path=resolved,
            is_directory=is_dir,
            size_bytes=0 if is_dir else stat.st_size,
            extension=None if is_dir else os.path.splitext(name)[1],
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            is_readable=folder.can_read(),
            is_writable=folder.can_write()
        )

        return OperationResult(
            success=True,
            operation="info",
            path=resolved,
            data=file_info.to_dict()
        )

    except OSError as e:
        return OperationResult(
            success=False,
            operation="info",
            path=resolved,
            error=f"Failed to get file info: {e}"
        )


def copy_file(
    folder: FolderConfig,
    source_path: str,
    dest_path: str,
    overwrite: bool = False
) -> OperationResult:
    """
    Copy a file within a folder.

    Args:
        folder: Folder configuration
        source_path: Source path relative to folder root
        dest_path: Destination path relative to folder root
        overwrite: Overwrite if destination exists

    Returns:
        OperationResult
    """
    source_full = resolve_folder_path(folder, source_path)
    dest_full = resolve_folder_path(folder, dest_path)

    # Validate read access on source
    is_valid, source_resolved, error = validate_read_access(source_full, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="copy",
            path=source_full,
            error=f"Source: {error}"
        )

    # Validate write access on destination
    is_valid, dest_resolved, error = validate_write_access(dest_full, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="copy",
            path=dest_full,
            error=f"Destination: {error}"
        )

    # Check source is a file
    if not os.path.isfile(source_resolved):
        return OperationResult(
            success=False,
            operation="copy",
            path=source_resolved,
            error="Source is not a file"
        )

    # Check destination
    if os.path.exists(dest_resolved) and not overwrite:
        return OperationResult(
            success=False,
            operation="copy",
            path=dest_resolved,
            error="Destination already exists. Use overwrite=True to replace."
        )

    try:
        shutil.copy2(source_resolved, dest_resolved)

        return OperationResult(
            success=True,
            operation="copy",
            path=dest_resolved,
            message=f"Copied from {source_path} to {dest_path}"
        )

    except OSError as e:
        return OperationResult(
            success=False,
            operation="copy",
            path=dest_resolved,
            error=f"Failed to copy: {e}"
        )


def move_file(
    folder: FolderConfig,
    source_path: str,
    dest_path: str,
    overwrite: bool = False
) -> OperationResult:
    """
    Move a file within a folder.

    Args:
        folder: Folder configuration
        source_path: Source path relative to folder root
        dest_path: Destination path relative to folder root
        overwrite: Overwrite if destination exists

    Returns:
        OperationResult
    """
    source_full = resolve_folder_path(folder, source_path)
    dest_full = resolve_folder_path(folder, dest_path)

    # Validate delete access on source (need write to move/delete)
    is_valid, source_resolved, error = validate_delete_access(source_full, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="move",
            path=source_full,
            error=f"Source: {error}"
        )

    # Validate write access on destination
    is_valid, dest_resolved, error = validate_write_access(dest_full, folder)
    if not is_valid:
        return OperationResult(
            success=False,
            operation="move",
            path=dest_full,
            error=f"Destination: {error}"
        )

    # Check destination
    if os.path.exists(dest_resolved) and not overwrite:
        return OperationResult(
            success=False,
            operation="move",
            path=dest_resolved,
            error="Destination already exists. Use overwrite=True to replace."
        )

    try:
        shutil.move(source_resolved, dest_resolved)

        return OperationResult(
            success=True,
            operation="move",
            path=dest_resolved,
            message=f"Moved from {source_path} to {dest_path}"
        )

    except OSError as e:
        return OperationResult(
            success=False,
            operation="move",
            path=dest_resolved,
            error=f"Failed to move: {e}"
        )
