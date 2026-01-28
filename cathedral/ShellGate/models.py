"""
ShellGate Pydantic models.

Defines command execution state, configuration, and results.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CommandStatus(str, Enum):
    """Status of a command execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class CommandExecution(BaseModel):
    """Record of a command execution."""
    id: str = Field(description="Unique execution identifier")
    command: str = Field(description="The command that was executed")
    working_dir: str = Field(description="Working directory for execution")
    status: CommandStatus = Field(default=CommandStatus.PENDING)
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = Field(default=60)
    is_background: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommandExecution":
        """Create from dict."""
        return cls.model_validate(data)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def is_done(self) -> bool:
        """Check if execution is complete."""
        return self.status in (
            CommandStatus.COMPLETED,
            CommandStatus.FAILED,
            CommandStatus.TIMEOUT,
            CommandStatus.CANCELLED
        )


class CommandResult(BaseModel):
    """Result of a command execution."""
    success: bool
    command: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    status: CommandStatus = CommandStatus.COMPLETED
    error: Optional[str] = None
    execution_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")


class CommandConfig(BaseModel):
    """Configuration for shell command execution."""
    default_timeout_seconds: int = Field(default=60, ge=1, le=600)
    max_timeout_seconds: int = Field(default=300, ge=1, le=600)
    default_working_dir: Optional[str] = None
    max_output_bytes: int = Field(default=1024 * 1024, description="Max output size (1MB)")

    # Security settings
    allowed_commands: List[str] = Field(
        default_factory=list,
        description="Allowlist of commands (empty = all except blocked)"
    )
    blocked_commands: List[str] = Field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf /*",
            "format",
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "init 0",
            "init 6",
            ":(){:|:&};:",  # Fork bomb
        ]
    )
    blocked_patterns: List[str] = Field(
        default_factory=lambda: [
            r"rm\s+-[rf]+\s+/\s*$",
            r"rm\s+-[rf]+\s+/\*",
            r">\s*/dev/sd[a-z]",
            r"dd\s+.*of=/dev/sd[a-z]",
            r"mkfs\.",
            r":\(\)\{.*\}",  # Fork bomb pattern
        ]
    )
    env_blocklist: List[str] = Field(
        default_factory=lambda: [
            "AWS_SECRET_ACCESS_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "AZURE_API_KEY",
            "DATABASE_PASSWORD",
            "DB_PASSWORD",
            "SECRET_KEY",
            "PRIVATE_KEY",
        ]
    )
    require_unlock: bool = Field(default=True, description="Require security unlock for execution")
    log_commands: bool = Field(default=True, description="Log all commands to history")
    max_concurrent_background: int = Field(default=5, description="Max concurrent background commands")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommandConfig":
        """Create from dict."""
        return cls.model_validate(data)


class ShellConfig(BaseModel):
    """Global ShellGate configuration."""
    command_config: CommandConfig = Field(default_factory=CommandConfig)
    history_max_entries: int = Field(default=1000)
    history_retention_days: int = Field(default=30)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShellConfig":
        """Create from dict."""
        return cls.model_validate(data)


class HistoryEntry(BaseModel):
    """Entry in command history."""
    id: str
    command: str
    working_dir: str
    exit_code: int
    success: bool
    started_at: datetime
    duration_seconds: float
    stdout_preview: str = Field(default="", description="First 500 chars of stdout")
    stderr_preview: str = Field(default="", description="First 500 chars of stderr")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_execution(cls, execution: CommandExecution) -> "HistoryEntry":
        """Create history entry from execution."""
        return cls(
            id=execution.id,
            command=execution.command,
            working_dir=execution.working_dir,
            exit_code=execution.exit_code or 0,
            success=execution.status == CommandStatus.COMPLETED and execution.exit_code == 0,
            started_at=execution.started_at or datetime.utcnow(),
            duration_seconds=execution.duration_seconds or 0,
            stdout_preview=execution.stdout[:500] if execution.stdout else "",
            stderr_preview=execution.stderr[:500] if execution.stderr else ""
        )
