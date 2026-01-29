"""
FileSystemGate - Secure file system access for Cathedral.

Provides:
- Folder-based access control with read/write/no-access permissions
- Path traversal prevention
- Extension blocking for security
- Automatic backup on modify/delete
- Audit logging via event bus

Usage:
    from cathedral import FileSystemGate

    # Initialize (call on startup)
    FileSystemGate.initialize()

    # Add a managed folder
    FileSystemGate.add_folder("projects", "/path/to/projects", "Projects", "read_write")

    # List files
    result = FileSystemGate.list_dir("projects", "subfolder")

    # Read a file
    result = FileSystemGate.read_file("projects", "README.md")

    # Write a file (with automatic backup if policy set)
    result = FileSystemGate.write_file("projects", "notes.txt", "Hello world")
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from .models import (
    FolderConfig,
    FolderPermission,
    BackupPolicy,
    FileSystemConfig,
    OperationResult,
    FileInfo,
    BackupRecord,
)
from .security import (
    resolve_folder_path,
    normalize_path,
    PathSecurityError,
)
from .operations import (
    list_directory as op_list_directory,
    read_file as op_read_file,
    write_file as op_write_file,
    make_directory as op_make_directory,
    delete_path as op_delete_path,
    get_file_info as op_get_file_info,
    copy_file as op_copy_file,
    move_file as op_move_file,
)
from .backup import BackupManager


# Module-level state
_config: Optional[FileSystemConfig] = None
_backup_manager: Optional[BackupManager] = None
_initialized: bool = False

# Default paths
DEFAULT_CONFIG_PATH = "data/config/filesystem.json"
DEFAULT_BACKUP_PATH = "data/backups"


class FileSystemGate:
    """
    Main interface for Cathedral's file system access.

    All methods are class methods for easy access throughout the application.
    """

    @classmethod
    def initialize(cls, config_path: Optional[str] = None, backup_path: Optional[str] = None) -> bool:
        """
        Initialize the file system gate.

        Args:
            config_path: Path to config file (default: data/config/filesystem.json)
            backup_path: Path to backup directory (default: data/backups)

        Returns:
            True if initialization successful
        """
        global _config, _backup_manager, _initialized

        try:
            # Resolve paths relative to project root
            if config_path is None:
                config_path = DEFAULT_CONFIG_PATH
            if backup_path is None:
                backup_path = DEFAULT_BACKUP_PATH

            # Ensure directories exist
            config_dir = os.path.dirname(config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            os.makedirs(backup_path, exist_ok=True)

            # Load or create config
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _config = FileSystemConfig.from_dict(data)
            else:
                _config = FileSystemConfig()
                cls._save_config(config_path)

            # Initialize backup manager
            _backup_manager = BackupManager(backup_path)

            _initialized = True
            return True

        except Exception as e:
            print(f"[FileSystemGate] Initialization failed: {e}")
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
        config_dir = os.path.dirname(path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(_config.to_dict(), f, indent=2, default=str)

    @classmethod
    def _get_config(cls) -> FileSystemConfig:
        """Get current config, initializing if needed."""
        global _config
        if _config is None:
            cls.initialize()
        return _config

    @classmethod
    def _get_backup_manager(cls) -> BackupManager:
        """Get backup manager, initializing if needed."""
        global _backup_manager
        if _backup_manager is None:
            cls.initialize()
        return _backup_manager

    # ==================== Folder Management ====================

    @classmethod
    def add_folder(
        cls,
        folder_id: str,
        path: str,
        name: str,
        permission: str = "read_only",
        backup_policy: str = "on_modify",
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Add a managed folder.

        Args:
            folder_id: Unique identifier/shortcut for the folder
            path: Absolute filesystem path
            name: Display name
            permission: read_only, read_write, or no_access
            backup_policy: never, always, on_modify, or on_delete
            **kwargs: Additional FolderConfig fields

        Returns:
            Tuple of (success, message)
        """
        config = cls._get_config()

        # Validate path exists
        abs_path = normalize_path(path)
        if not os.path.exists(abs_path):
            return False, f"Path does not exist: {abs_path}"
        if not os.path.isdir(abs_path):
            return False, f"Path is not a directory: {abs_path}"

        # Check for duplicate ID
        if config.get_folder(folder_id) is not None:
            return False, f"Folder ID already exists: {folder_id}"

        # Create folder config
        try:
            folder = FolderConfig(
                id=folder_id,
                path=abs_path,
                name=name,
                permission=FolderPermission(permission),
                backup_policy=BackupPolicy(backup_policy),
                **kwargs
            )
        except ValueError as e:
            return False, f"Invalid configuration: {e}"

        # Add to config
        config.add_folder(folder)
        cls._save_config()

        return True, f"Added folder '{name}' ({folder_id}) with {permission} access"

    @classmethod
    def remove_folder(cls, folder_id: str) -> Tuple[bool, str]:
        """
        Remove a managed folder.

        Args:
            folder_id: Folder ID to remove

        Returns:
            Tuple of (success, message)
        """
        config = cls._get_config()

        if config.get_folder(folder_id) is None:
            return False, f"Folder not found: {folder_id}"

        config.remove_folder(folder_id)
        cls._save_config()

        return True, f"Removed folder: {folder_id}"

    @classmethod
    def update_folder(cls, folder_id: str, **updates) -> Tuple[bool, str]:
        """
        Update a folder's configuration.

        Args:
            folder_id: Folder ID to update
            **updates: Fields to update (permission, backup_policy, etc.)

        Returns:
            Tuple of (success, message)
        """
        config = cls._get_config()

        folder = config.get_folder(folder_id)
        if folder is None:
            return False, f"Folder not found: {folder_id}"

        # Convert string enums if needed
        if "permission" in updates:
            updates["permission"] = FolderPermission(updates["permission"])
        if "backup_policy" in updates:
            updates["backup_policy"] = BackupPolicy(updates["backup_policy"])

        config.update_folder(folder_id, updates)
        cls._save_config()

        return True, f"Updated folder: {folder_id}"

    @classmethod
    def get_folder(cls, folder_id: str) -> Optional[FolderConfig]:
        """Get a folder configuration by ID."""
        return cls._get_config().get_folder(folder_id)

    @classmethod
    def list_folders(cls) -> List[Dict[str, Any]]:
        """List all configured folders."""
        config = cls._get_config()
        return [f.to_dict() for f in config.folders]

    # ==================== File Operations ====================

    @classmethod
    def _parse_path(cls, path_spec: str) -> Tuple[Optional[FolderConfig], str]:
        """
        Parse a path specification like 'folder:path' or just 'path'.

        Args:
            path_spec: Path specification

        Returns:
            Tuple of (folder_config or None, relative_path)
        """
        config = cls._get_config()

        if ":" in path_spec:
            folder_id, rel_path = path_spec.split(":", 1)
            folder = config.get_folder(folder_id)
            return folder, rel_path
        else:
            # Try to find a matching folder
            for folder in config.folders:
                if path_spec.startswith(folder.path):
                    rel_path = os.path.relpath(path_spec, folder.path)
                    return folder, rel_path
            return None, path_spec

    @classmethod
    def list_dir(
        cls,
        folder_id: str,
        relative_path: str = "",
        show_hidden: bool = False
    ) -> OperationResult:
        """
        List directory contents.

        Args:
            folder_id: Folder ID
            relative_path: Path relative to folder root
            show_hidden: Include hidden files

        Returns:
            OperationResult with list of FileInfo
        """
        folder = cls.get_folder(folder_id)
        if folder is None:
            return OperationResult(
                success=False,
                operation="list",
                path=relative_path,
                error=f"Folder not found: {folder_id}"
            )

        return op_list_directory(folder, relative_path, show_hidden)

    @classmethod
    def read_file(
        cls,
        folder_id: str,
        relative_path: str,
        encoding: str = "utf-8",
        binary: bool = False
    ) -> OperationResult:
        """
        Read a file's contents.

        Args:
            folder_id: Folder ID
            relative_path: Path relative to folder root
            encoding: Text encoding
            binary: Read as binary

        Returns:
            OperationResult with file contents in data
        """
        folder = cls.get_folder(folder_id)
        if folder is None:
            return OperationResult(
                success=False,
                operation="read",
                path=relative_path,
                error=f"Folder not found: {folder_id}"
            )

        return op_read_file(folder, relative_path, encoding, binary)

    @classmethod
    def write_file(
        cls,
        folder_id: str,
        relative_path: str,
        content: str | bytes,
        encoding: str = "utf-8",
        create_dirs: bool = False
    ) -> OperationResult:
        """
        Write content to a file.

        Creates automatic backup if folder policy requires it.

        Args:
            folder_id: Folder ID
            relative_path: Path relative to folder root
            content: Content to write
            encoding: Text encoding
            create_dirs: Create parent directories

        Returns:
            OperationResult
        """
        folder = cls.get_folder(folder_id)
        if folder is None:
            return OperationResult(
                success=False,
                operation="write",
                path=relative_path,
                error=f"Folder not found: {folder_id}"
            )

        # Create backup if file exists and policy requires
        backup_manager = cls._get_backup_manager()
        target_path = resolve_folder_path(folder, relative_path)

        backup_record = None
        if os.path.exists(target_path):
            backup_record = backup_manager.create_backup(folder, target_path, "modify")

        # Perform write
        result, _ = op_write_file(folder, relative_path, content, encoding, create_dirs)

        # Add backup ID to result if backup was created
        if backup_record:
            result.backup_id = backup_record.id
            result.message += f" (backup: {backup_record.id})"

        return result

    @classmethod
    def mkdir(
        cls,
        folder_id: str,
        relative_path: str,
        parents: bool = False
    ) -> OperationResult:
        """
        Create a directory.

        Args:
            folder_id: Folder ID
            relative_path: Path relative to folder root
            parents: Create parent directories

        Returns:
            OperationResult
        """
        folder = cls.get_folder(folder_id)
        if folder is None:
            return OperationResult(
                success=False,
                operation="mkdir",
                path=relative_path,
                error=f"Folder not found: {folder_id}"
            )

        return op_make_directory(folder, relative_path, parents)

    @classmethod
    def delete(
        cls,
        folder_id: str,
        relative_path: str,
        recursive: bool = False
    ) -> OperationResult:
        """
        Delete a file or directory.

        Creates automatic backup if folder policy requires it.

        Args:
            folder_id: Folder ID
            relative_path: Path relative to folder root
            recursive: Delete directories recursively

        Returns:
            OperationResult
        """
        folder = cls.get_folder(folder_id)
        if folder is None:
            return OperationResult(
                success=False,
                operation="delete",
                path=relative_path,
                error=f"Folder not found: {folder_id}"
            )

        # Create backup before delete if policy requires
        backup_manager = cls._get_backup_manager()
        target_path = resolve_folder_path(folder, relative_path)

        backup_record = None
        if os.path.exists(target_path):
            backup_record = backup_manager.create_backup(folder, target_path, "delete")

        # Perform delete
        result, _ = op_delete_path(folder, relative_path, recursive)

        # Add backup ID to result
        if backup_record:
            result.backup_id = backup_record.id
            result.message += f" (backup: {backup_record.id})"

        return result

    @classmethod
    def info(cls, folder_id: str, relative_path: str) -> OperationResult:
        """
        Get file or directory information.

        Args:
            folder_id: Folder ID
            relative_path: Path relative to folder root

        Returns:
            OperationResult with FileInfo
        """
        folder = cls.get_folder(folder_id)
        if folder is None:
            return OperationResult(
                success=False,
                operation="info",
                path=relative_path,
                error=f"Folder not found: {folder_id}"
            )

        return op_get_file_info(folder, relative_path)

    @classmethod
    def copy(
        cls,
        folder_id: str,
        source_path: str,
        dest_path: str,
        overwrite: bool = False
    ) -> OperationResult:
        """
        Copy a file within a folder.

        Args:
            folder_id: Folder ID
            source_path: Source path relative to folder
            dest_path: Destination path relative to folder
            overwrite: Overwrite if destination exists

        Returns:
            OperationResult
        """
        folder = cls.get_folder(folder_id)
        if folder is None:
            return OperationResult(
                success=False,
                operation="copy",
                path=source_path,
                error=f"Folder not found: {folder_id}"
            )

        return op_copy_file(folder, source_path, dest_path, overwrite)

    @classmethod
    def move(
        cls,
        folder_id: str,
        source_path: str,
        dest_path: str,
        overwrite: bool = False
    ) -> OperationResult:
        """
        Move a file within a folder.

        Args:
            folder_id: Folder ID
            source_path: Source path relative to folder
            dest_path: Destination path relative to folder
            overwrite: Overwrite if destination exists

        Returns:
            OperationResult
        """
        folder = cls.get_folder(folder_id)
        if folder is None:
            return OperationResult(
                success=False,
                operation="move",
                path=source_path,
                error=f"Folder not found: {folder_id}"
            )

        return op_move_file(folder, source_path, dest_path, overwrite)

    # ==================== Backup Operations ====================

    @classmethod
    def list_backups(
        cls,
        folder_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List backups.

        Args:
            folder_id: Filter by folder ID (None for all)
            limit: Maximum records

        Returns:
            List of backup records
        """
        backup_manager = cls._get_backup_manager()
        records = backup_manager.list_backups(folder_id, limit)
        return [r.to_dict() for r in records]

    @classmethod
    def restore_backup(
        cls,
        backup_id: str,
        overwrite: bool = True
    ) -> Tuple[bool, str]:
        """
        Restore a file from backup.

        Args:
            backup_id: Backup ID
            overwrite: Overwrite existing file

        Returns:
            Tuple of (success, message)
        """
        return cls._get_backup_manager().restore_backup(backup_id, overwrite)

    @classmethod
    def delete_backup(cls, backup_id: str) -> Tuple[bool, str]:
        """Delete a backup."""
        return cls._get_backup_manager().delete_backup(backup_id)

    @classmethod
    def cleanup_backups(
        cls,
        max_age_days: int = 30,
        max_count: int = 100
    ) -> Tuple[int, str]:
        """
        Clean up old backups.

        Args:
            max_age_days: Delete backups older than this
            max_count: Keep at most this many

        Returns:
            Tuple of (deleted_count, message)
        """
        return cls._get_backup_manager().cleanup_old_backups(max_age_days, max_count)

    @classmethod
    def get_backup_stats(cls) -> Dict[str, Any]:
        """Get backup statistics."""
        return cls._get_backup_manager().get_stats()

    # ==================== Configuration ====================

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get full configuration."""
        return cls._get_config().to_dict()

    @classmethod
    def update_config(
        cls,
        require_unlock_for_write: Optional[bool] = None,
        require_unlock_for_read: Optional[bool] = None,
        max_backup_age_days: Optional[int] = None,
        max_backup_count: Optional[int] = None
    ):
        """Update global configuration settings."""
        config = cls._get_config()

        if require_unlock_for_write is not None:
            config.require_unlock_for_write = require_unlock_for_write
        if require_unlock_for_read is not None:
            config.require_unlock_for_read = require_unlock_for_read
        if max_backup_age_days is not None:
            config.max_backup_age_days = max_backup_age_days
        if max_backup_count is not None:
            config.max_backup_count = max_backup_count

        cls._save_config()


# ==================== Convenience Functions ====================

def initialize(config_path: Optional[str] = None, backup_path: Optional[str] = None) -> bool:
    """Initialize FileSystemGate."""
    return FileSystemGate.initialize(config_path, backup_path)


def is_initialized() -> bool:
    """Check if initialized."""
    return FileSystemGate.is_initialized()


def add_folder(
    folder_id: str,
    path: str,
    name: str,
    permission: str = "read_only",
    **kwargs
) -> Tuple[bool, str]:
    """Add a managed folder."""
    return FileSystemGate.add_folder(folder_id, path, name, permission, **kwargs)


def remove_folder(folder_id: str) -> Tuple[bool, str]:
    """Remove a managed folder."""
    return FileSystemGate.remove_folder(folder_id)


def list_folders() -> List[Dict[str, Any]]:
    """List all configured folders."""
    return FileSystemGate.list_folders()


def get_folder(folder_id: str) -> Optional[FolderConfig]:
    """Get a folder configuration."""
    return FileSystemGate.get_folder(folder_id)


def list_dir(folder_id: str, relative_path: str = "", show_hidden: bool = False) -> OperationResult:
    """List directory contents."""
    return FileSystemGate.list_dir(folder_id, relative_path, show_hidden)


def read_file(folder_id: str, relative_path: str, encoding: str = "utf-8", binary: bool = False) -> OperationResult:
    """Read a file."""
    return FileSystemGate.read_file(folder_id, relative_path, encoding, binary)


def write_file(folder_id: str, relative_path: str, content: str | bytes, **kwargs) -> OperationResult:
    """Write a file."""
    return FileSystemGate.write_file(folder_id, relative_path, content, **kwargs)


def mkdir(folder_id: str, relative_path: str, parents: bool = False) -> OperationResult:
    """Create a directory."""
    return FileSystemGate.mkdir(folder_id, relative_path, parents)


def delete(folder_id: str, relative_path: str, recursive: bool = False) -> OperationResult:
    """Delete a file or directory."""
    return FileSystemGate.delete(folder_id, relative_path, recursive)


def list_backups(folder_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """List backups."""
    return FileSystemGate.list_backups(folder_id, limit)


def restore_backup(backup_id: str, overwrite: bool = True) -> Tuple[bool, str]:
    """Restore from backup."""
    return FileSystemGate.restore_backup(backup_id, overwrite)
