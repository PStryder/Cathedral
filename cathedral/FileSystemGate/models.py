"""
FileSystemGate Pydantic models.

Defines folder configurations, permissions, backup records, and file metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FolderPermission(str, Enum):
    """Permission level for folder access."""
    NO_ACCESS = "no_access"
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class BackupPolicy(str, Enum):
    """When to create automatic backups."""
    NEVER = "never"
    ALWAYS = "always"
    ON_MODIFY = "on_modify"
    ON_DELETE = "on_delete"


class FolderConfig(BaseModel):
    """Configuration for a managed folder."""
    id: str = Field(description="Shortcut identifier (e.g., 'projects')")
    path: str = Field(description="Absolute filesystem path")
    name: str = Field(description="Display name for UI")
    permission: FolderPermission = Field(default=FolderPermission.READ_ONLY)
    backup_policy: BackupPolicy = Field(default=BackupPolicy.ON_MODIFY)
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    allowed_extensions: Optional[List[str]] = Field(
        default=None,
        description="Whitelist of extensions (None = all allowed)"
    )
    blocked_extensions: List[str] = Field(
        default_factory=lambda: [".exe", ".dll", ".bat", ".cmd", ".ps1", ".scr", ".com"]
    )
    recursive: bool = Field(default=True, description="Allow access to subdirectories")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderConfig":
        """Create from dict."""
        return cls.model_validate(data)

    def can_read(self) -> bool:
        """Check if reading is allowed."""
        return self.permission in (FolderPermission.READ_ONLY, FolderPermission.READ_WRITE)

    def can_write(self) -> bool:
        """Check if writing is allowed."""
        return self.permission == FolderPermission.READ_WRITE


class BackupRecord(BaseModel):
    """Record of a backup operation."""
    id: str = Field(description="Unique backup identifier")
    folder_id: str = Field(description="Source folder ID")
    original_path: str = Field(description="Original file/folder path")
    backup_path: str = Field(description="Path to backup in backup storage")
    operation: str = Field(description="Operation that triggered backup (modify/delete)")
    file_size_bytes: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")


class FileInfo(BaseModel):
    """Information about a file or directory."""
    name: str
    path: str = Field(description="Path relative to folder root")
    absolute_path: str
    is_directory: bool
    size_bytes: int = 0
    extension: Optional[str] = None
    modified_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    is_readable: bool = True
    is_writable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")


class OperationResult(BaseModel):
    """Result of a file system operation."""
    success: bool
    operation: str = Field(description="Operation type: read/write/list/mkdir/delete")
    path: str
    message: str = ""
    data: Optional[Any] = None
    backup_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")


class FileSystemConfig(BaseModel):
    """Global FileSystemGate configuration."""
    folders: List[FolderConfig] = Field(default_factory=list)
    require_unlock_for_write: bool = Field(default=True)
    require_unlock_for_read: bool = Field(default=False)
    max_backup_age_days: int = Field(default=30)
    max_backup_count: int = Field(default=100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileSystemConfig":
        """Create from dict."""
        return cls.model_validate(data)

    def get_folder(self, folder_id: str) -> Optional[FolderConfig]:
        """Get folder config by ID."""
        for folder in self.folders:
            if folder.id == folder_id:
                return folder
        return None

    def add_folder(self, folder: FolderConfig) -> bool:
        """Add a folder config. Returns False if ID already exists."""
        if self.get_folder(folder.id) is not None:
            return False
        self.folders.append(folder)
        return True

    def remove_folder(self, folder_id: str) -> bool:
        """Remove a folder config by ID."""
        for i, folder in enumerate(self.folders):
            if folder.id == folder_id:
                self.folders.pop(i)
                return True
        return False

    def update_folder(self, folder_id: str, updates: Dict[str, Any]) -> bool:
        """Update a folder config."""
        folder = self.get_folder(folder_id)
        if folder is None:
            return False

        for key, value in updates.items():
            if hasattr(folder, key):
                setattr(folder, key, value)
        folder.updated_at = datetime.utcnow()
        return True
