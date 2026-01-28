"""
FileSystemGate backup management.

Provides backup creation, restoration, and listing for file operations.
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from typing import List, Optional
import uuid

from .models import BackupRecord, FolderConfig, BackupPolicy


# Default backup directory relative to data/
DEFAULT_BACKUP_DIR = "backups"


class BackupManager:
    """Manages file backups for FileSystemGate."""

    def __init__(self, backup_root: str):
        """
        Initialize the backup manager.

        Args:
            backup_root: Root directory for backups (e.g., data/backups)
        """
        self.backup_root = backup_root
        self.index_file = os.path.join(backup_root, "backup_index.json")
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        """Ensure backup directory exists."""
        os.makedirs(self.backup_root, exist_ok=True)

    def _load_index(self) -> List[BackupRecord]:
        """Load backup index from disk."""
        if not os.path.exists(self.index_file):
            return []

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [BackupRecord.model_validate(item) for item in data]
        except (json.JSONDecodeError, OSError):
            return []

    def _save_index(self, records: List[BackupRecord]):
        """Save backup index to disk."""
        data = [record.to_dict() for record in records]
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def should_backup(
        self,
        folder: FolderConfig,
        operation: str
    ) -> bool:
        """
        Check if a backup should be created based on folder policy.

        Args:
            folder: Folder configuration
            operation: Operation type (modify/delete)

        Returns:
            True if backup should be created
        """
        policy = folder.backup_policy

        if policy == BackupPolicy.NEVER:
            return False
        if policy == BackupPolicy.ALWAYS:
            return True
        if policy == BackupPolicy.ON_MODIFY and operation == "modify":
            return True
        if policy == BackupPolicy.ON_DELETE and operation == "delete":
            return True

        return False

    def create_backup(
        self,
        folder: FolderConfig,
        file_path: str,
        operation: str
    ) -> Optional[BackupRecord]:
        """
        Create a backup of a file before modification/deletion.

        Args:
            folder: Folder configuration
            file_path: Absolute path to file to backup
            operation: Operation triggering backup (modify/delete)

        Returns:
            BackupRecord if backup was created, None otherwise
        """
        if not self.should_backup(folder, operation):
            return None

        if not os.path.exists(file_path):
            return None

        # Generate backup ID and path
        backup_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(file_path)

        # Create timestamped backup directory
        backup_dir = os.path.join(self.backup_root, timestamp)
        os.makedirs(backup_dir, exist_ok=True)

        # Create backup filename with ID prefix
        backup_filename = f"{backup_id}_{filename}"
        backup_path = os.path.join(backup_dir, backup_filename)

        try:
            # Copy the file/directory
            if os.path.isdir(file_path):
                shutil.copytree(file_path, backup_path)
            else:
                shutil.copy2(file_path, backup_path)

            # Get file size
            if os.path.isfile(backup_path):
                size = os.path.getsize(backup_path)
            else:
                size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for dirpath, _, filenames in os.walk(backup_path)
                    for f in filenames
                )

            # Create record
            record = BackupRecord(
                id=backup_id,
                folder_id=folder.id,
                original_path=file_path,
                backup_path=backup_path,
                operation=operation,
                file_size_bytes=size,
                created_at=datetime.utcnow()
            )

            # Update index
            records = self._load_index()
            records.append(record)
            self._save_index(records)

            return record

        except OSError as e:
            print(f"[BackupManager] Failed to create backup: {e}")
            return None

    def restore_backup(
        self,
        backup_id: str,
        overwrite: bool = True
    ) -> tuple[bool, str]:
        """
        Restore a file from backup.

        Args:
            backup_id: Backup ID to restore
            overwrite: Overwrite existing file if it exists

        Returns:
            Tuple of (success, message)
        """
        records = self._load_index()
        record = None

        for r in records:
            if r.id == backup_id:
                record = r
                break

        if record is None:
            return False, f"Backup not found: {backup_id}"

        if not os.path.exists(record.backup_path):
            return False, f"Backup file missing: {record.backup_path}"

        # Check if destination exists
        if os.path.exists(record.original_path) and not overwrite:
            return False, f"Original path exists and overwrite=False: {record.original_path}"

        try:
            # Remove existing file/directory if present
            if os.path.exists(record.original_path):
                if os.path.isdir(record.original_path):
                    shutil.rmtree(record.original_path)
                else:
                    os.remove(record.original_path)

            # Restore from backup
            if os.path.isdir(record.backup_path):
                shutil.copytree(record.backup_path, record.original_path)
            else:
                # Ensure parent directory exists
                parent = os.path.dirname(record.original_path)
                os.makedirs(parent, exist_ok=True)
                shutil.copy2(record.backup_path, record.original_path)

            return True, f"Restored {record.original_path} from backup {backup_id}"

        except OSError as e:
            return False, f"Failed to restore: {e}"

    def list_backups(
        self,
        folder_id: Optional[str] = None,
        limit: int = 50
    ) -> List[BackupRecord]:
        """
        List backups, optionally filtered by folder.

        Args:
            folder_id: Filter by folder ID (None for all)
            limit: Maximum records to return

        Returns:
            List of BackupRecord
        """
        records = self._load_index()

        if folder_id:
            records = [r for r in records if r.folder_id == folder_id]

        # Sort by created_at descending (newest first)
        records.sort(key=lambda r: r.created_at, reverse=True)

        return records[:limit]

    def get_backup(self, backup_id: str) -> Optional[BackupRecord]:
        """Get a specific backup by ID."""
        records = self._load_index()
        for r in records:
            if r.id == backup_id:
                return r
        return None

    def delete_backup(self, backup_id: str) -> tuple[bool, str]:
        """
        Delete a backup.

        Args:
            backup_id: Backup ID to delete

        Returns:
            Tuple of (success, message)
        """
        records = self._load_index()
        record = None
        record_index = -1

        for i, r in enumerate(records):
            if r.id == backup_id:
                record = r
                record_index = i
                break

        if record is None:
            return False, f"Backup not found: {backup_id}"

        # Delete the backup file/directory
        try:
            if os.path.exists(record.backup_path):
                if os.path.isdir(record.backup_path):
                    shutil.rmtree(record.backup_path)
                else:
                    os.remove(record.backup_path)

            # Remove from index
            records.pop(record_index)
            self._save_index(records)

            return True, f"Deleted backup {backup_id}"

        except OSError as e:
            return False, f"Failed to delete backup: {e}"

    def cleanup_old_backups(
        self,
        max_age_days: int = 30,
        max_count: int = 100
    ) -> tuple[int, str]:
        """
        Clean up old backups based on age and count limits.

        Args:
            max_age_days: Delete backups older than this
            max_count: Keep at most this many backups

        Returns:
            Tuple of (deleted_count, message)
        """
        records = self._load_index()
        now = datetime.utcnow()
        cutoff = now - timedelta(days=max_age_days)

        # Identify old backups
        old_records = [r for r in records if r.created_at < cutoff]

        # Sort by age (oldest first) to delete oldest beyond max_count
        records.sort(key=lambda r: r.created_at)
        excess_records = records[max_count:] if len(records) > max_count else []

        # Combine records to delete (unique by ID)
        to_delete = {r.id: r for r in old_records + excess_records}

        deleted = 0
        for record in to_delete.values():
            success, _ = self.delete_backup(record.id)
            if success:
                deleted += 1

        # Clean up empty backup directories
        self._cleanup_empty_dirs()

        return deleted, f"Cleaned up {deleted} old backups"

    def _cleanup_empty_dirs(self):
        """Remove empty backup directories."""
        try:
            for entry in os.listdir(self.backup_root):
                path = os.path.join(self.backup_root, entry)
                if os.path.isdir(path) and not os.listdir(path):
                    os.rmdir(path)
        except OSError:
            pass

    def get_stats(self) -> dict:
        """Get backup statistics."""
        records = self._load_index()
        total_size = sum(r.file_size_bytes for r in records)

        return {
            "total_backups": len(records),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "folders_with_backups": len(set(r.folder_id for r in records)),
            "oldest_backup": min((r.created_at for r in records), default=None),
            "newest_backup": max((r.created_at for r in records), default=None)
        }
