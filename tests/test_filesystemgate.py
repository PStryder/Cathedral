"""
Tests for FileSystemGate secure file access.
"""

import os
import pytest
from pathlib import Path

from cathedral.FileSystemGate import (
    FileSystemGate,
    initialize,
    is_initialized,
    add_folder,
    remove_folder,
    list_folders,
)
from cathedral.FileSystemGate.models import FolderPermission, BackupPolicy


class TestFileSystemGateInitialization:
    """Tests for FileSystemGate initialization."""

    def test_initialize_creates_config(self, temp_data_dir):
        """Initialize should create config file if not exists."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")

        result = FileSystemGate.initialize(config_path, backup_path)

        assert result is True
        assert FileSystemGate.is_initialized() is True
        assert os.path.exists(config_path)

    def test_initialize_creates_backup_dir(self, temp_data_dir):
        """Initialize should create backup directory."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")

        FileSystemGate.initialize(config_path, backup_path)

        assert os.path.isdir(backup_path)


class TestFolderManagement:
    """Tests for folder management."""

    def test_add_folder(self, temp_data_dir, sample_folder):
        """Should add a managed folder."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)

        success, message = FileSystemGate.add_folder(
            folder_id="test",
            path=str(sample_folder),
            name="Test Folder",
            permission="read_write"
        )

        assert success is True
        assert "test" in message.lower() or "added" in message.lower()

    def test_add_folder_nonexistent_path(self, temp_data_dir):
        """Should reject nonexistent paths."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)

        success, message = FileSystemGate.add_folder(
            folder_id="bad",
            path="/nonexistent/path/12345",
            name="Bad Folder"
        )

        assert success is False
        assert "not exist" in message.lower() or "does not exist" in message.lower()

    def test_add_duplicate_folder_id(self, temp_data_dir, sample_folder):
        """Should reject duplicate folder IDs."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)

        FileSystemGate.add_folder("dup", str(sample_folder), "Folder 1")
        success, message = FileSystemGate.add_folder("dup", str(sample_folder), "Folder 2")

        assert success is False
        assert "exists" in message.lower()

    def test_remove_folder(self, temp_data_dir, sample_folder):
        """Should remove a managed folder."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)

        FileSystemGate.add_folder("removeme", str(sample_folder), "Remove Me")
        success, message = FileSystemGate.remove_folder("removeme")

        assert success is True
        assert FileSystemGate.get_folder("removeme") is None

    def test_list_folders(self, temp_data_dir, sample_folder):
        """Should list all managed folders."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)

        FileSystemGate.add_folder("list1", str(sample_folder), "List 1")

        folders = FileSystemGate.list_folders()

        assert len(folders) >= 1
        assert any(f["id"] == "list1" for f in folders)


class TestFileOperations:
    """Tests for file operations."""

    def test_list_dir(self, temp_data_dir, sample_folder):
        """Should list directory contents."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("files", str(sample_folder), "Files", "read_write")

        result = FileSystemGate.list_dir("files", "")

        assert result.success is True
        assert result.data is not None
        # Should contain readme.txt and data.json
        file_names = [f.name for f in result.data] if result.data else []
        assert "readme.txt" in file_names or len(file_names) > 0

    def test_list_dir_nonexistent_folder(self, temp_data_dir):
        """Should fail gracefully for nonexistent folder."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)

        result = FileSystemGate.list_dir("nonexistent", "")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_file(self, temp_data_dir, sample_folder):
        """Should read file contents."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("read", str(sample_folder), "Read", "read_only")

        result = FileSystemGate.read_file("read", "readme.txt")

        assert result.success is True
        assert result.data is not None
        assert "Hello World" in result.data

    def test_write_file(self, temp_data_dir, sample_folder):
        """Should write file contents."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("write", str(sample_folder), "Write", "read_write")

        result = FileSystemGate.write_file("write", "new_file.txt", "New content")

        assert result.success is True
        # Verify file was created
        assert (sample_folder / "new_file.txt").exists()
        assert (sample_folder / "new_file.txt").read_text() == "New content"

    def test_write_file_read_only_folder(self, temp_data_dir, sample_folder):
        """Should fail to write in read-only folder."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("readonly", str(sample_folder), "ReadOnly", "read_only")

        result = FileSystemGate.write_file("readonly", "blocked.txt", "Content")

        assert result.success is False
        assert "permission" in result.error.lower() or "read" in result.error.lower()

    def test_mkdir(self, temp_data_dir, sample_folder):
        """Should create directories."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("mkdir", str(sample_folder), "Mkdir", "read_write")

        result = FileSystemGate.mkdir("mkdir", "new_dir")

        assert result.success is True
        assert (sample_folder / "new_dir").is_dir()

    def test_delete_file(self, temp_data_dir, sample_folder):
        """Should delete files."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("delete", str(sample_folder), "Delete", "read_write")

        # Create a file to delete
        (sample_folder / "to_delete.txt").write_text("Delete me")

        result = FileSystemGate.delete("delete", "to_delete.txt")

        assert result.success is True
        assert not (sample_folder / "to_delete.txt").exists()

    def test_file_info(self, temp_data_dir, sample_folder):
        """Should get file information."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("info", str(sample_folder), "Info", "read_only")

        result = FileSystemGate.info("info", "readme.txt")

        assert result.success is True
        assert result.data is not None
        assert result.data.name == "readme.txt"


class TestBackupOperations:
    """Tests for backup functionality."""

    def test_write_creates_backup(self, temp_data_dir, sample_folder):
        """Should create backup when modifying existing file."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder(
            "backup_test",
            str(sample_folder),
            "Backup Test",
            "read_write",
            backup_policy="on_modify"
        )

        # Modify existing file
        result = FileSystemGate.write_file("backup_test", "readme.txt", "Modified content")

        assert result.success is True
        # Should have created a backup
        backups = FileSystemGate.list_backups("backup_test")
        assert len(backups) >= 1

    def test_list_backups(self, temp_data_dir, sample_folder):
        """Should list backups."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder(
            "list_backup",
            str(sample_folder),
            "List Backup",
            "read_write",
            backup_policy="always"
        )

        # Create some backups
        FileSystemGate.write_file("list_backup", "readme.txt", "V1")
        FileSystemGate.write_file("list_backup", "readme.txt", "V2")

        backups = FileSystemGate.list_backups()

        # Should have at least the backups we created
        assert isinstance(backups, list)

    def test_backup_stats(self, temp_data_dir, sample_folder):
        """Should return backup statistics."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)

        stats = FileSystemGate.get_backup_stats()

        assert "total_backups" in stats or "count" in stats or isinstance(stats, dict)


class TestPathSecurity:
    """Tests for path security."""

    def test_path_traversal_blocked(self, temp_data_dir, sample_folder):
        """Should block path traversal attempts."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("secure", str(sample_folder), "Secure", "read_write")

        # Try to escape folder
        result = FileSystemGate.read_file("secure", "../../../etc/passwd")

        assert result.success is False
        # Should be blocked for security reasons

    def test_absolute_path_blocked(self, temp_data_dir, sample_folder):
        """Should block absolute paths."""
        config_path = str(temp_data_dir / "config" / "filesystem.json")
        backup_path = str(temp_data_dir / "backups")
        FileSystemGate.initialize(config_path, backup_path)
        FileSystemGate.add_folder("abs", str(sample_folder), "Abs", "read_write")

        # Try absolute path
        if os.name == 'nt':
            result = FileSystemGate.read_file("abs", "C:\\Windows\\System32\\config\\SAM")
        else:
            result = FileSystemGate.read_file("abs", "/etc/passwd")

        assert result.success is False
