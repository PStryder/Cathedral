"""
ToolGate Protocol Models.

Defines the Cathedral Tool Call Protocol types for provider-agnostic tool calling.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class PolicyClass(str, Enum):
    """Security policy classes for tools."""

    READ_ONLY = "read_only"  # No side effects, always safe
    WRITE = "write"  # Creates/modifies data
    DESTRUCTIVE = "destructive"  # Can delete data
    PRIVILEGED = "privileged"  # System-level access (shell, etc.)
    NETWORK = "network"  # External network access


class ToolCall(BaseModel):
    """Single tool call from model."""

    type: Literal["tool_call"] = "tool_call"
    id: str = Field(description="Unique call identifier, e.g., tc_abc123")
    tool: str = Field(description="Tool name, e.g., MemoryGate.search")
    args: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"ToolCall({self.tool}, id={self.id})"


class ToolCallBatch(BaseModel):
    """Batch of tool calls."""

    type: Literal["tool_calls"] = "tool_calls"
    calls: List[ToolCall]

    def __len__(self) -> int:
        return len(self.calls)


class ToolResult(BaseModel):
    """Result from tool execution."""

    type: Literal["tool_result"] = "tool_result"
    id: str
    ok: bool
    result: Optional[Any] = None
    error: Optional[str] = None

    @classmethod
    def success(cls, call_id: str, result: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(id=call_id, ok=True, result=result)

    @classmethod
    def failure(cls, call_id: str, error: str) -> "ToolResult":
        """Create a failed result."""
        return cls(id=call_id, ok=False, error=error)

    def to_compact(self) -> Dict[str, Any]:
        """Convert to compact dict for conversation injection."""
        if self.ok:
            return {"id": self.id, "ok": True, "result": self.result}
        return {"id": self.id, "ok": False, "error": self.error}


class ArgSchema(BaseModel):
    """JSON Schema for a tool argument."""

    type: str = "string"
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, Any]] = None  # For array types


class ToolDefinition(BaseModel):
    """Complete tool definition for registry."""

    name: str = Field(description="Full tool name, e.g., MemoryGate.search")
    description: str = Field(description="Human-readable description")
    gate: str = Field(description="Owning gate name, e.g., MemoryGate")
    method: str = Field(description="Method name on gate, e.g., search")
    policy_class: PolicyClass = Field(default=PolicyClass.READ_ONLY)
    is_async: bool = Field(default=False, description="Whether method is async")
    args_schema: Dict[str, ArgSchema] = Field(
        default_factory=dict, description="Argument schemas"
    )

    def get_json_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema for this tool's arguments."""
        properties = {}
        required = []

        for arg_name, arg_schema in self.args_schema.items():
            prop = {"type": arg_schema.type, "description": arg_schema.description}

            if arg_schema.enum:
                prop["enum"] = arg_schema.enum
            if arg_schema.items:
                prop["items"] = arg_schema.items
            if arg_schema.default is not None:
                prop["default"] = arg_schema.default

            properties[arg_name] = prop

            if arg_schema.required:
                required.append(arg_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_prompt_schema(self) -> str:
        """Generate schema documentation for system prompt."""
        lines = [f"### {self.name}", f"{self.description}", ""]

        if self.args_schema:
            lines.append("**Arguments:**")
            for arg_name, arg_schema in self.args_schema.items():
                req = "(required)" if arg_schema.required else "(optional)"
                default = f", default={arg_schema.default}" if arg_schema.default is not None else ""
                lines.append(f"- `{arg_name}`: {arg_schema.type} {req}{default}")
                if arg_schema.description:
                    lines.append(f"  {arg_schema.description}")
        else:
            lines.append("**Arguments:** None")

        return "\n".join(lines)


class ToolBudget(BaseModel):
    """Budget tracking for tool execution."""

    max_iterations: int = Field(default=6, description="Max agentic loop iterations")
    max_calls_per_step: int = Field(default=5, description="Max tool calls per step")
    max_total_calls: int = Field(default=20, description="Max total calls per session")

    # Tracking
    iterations_used: int = 0
    calls_used: int = 0

    def can_iterate(self) -> bool:
        """Check if another iteration is allowed."""
        return self.iterations_used < self.max_iterations

    def can_call(self, count: int = 1) -> bool:
        """Check if more calls are allowed."""
        return self.calls_used + count <= self.max_total_calls

    def use_iteration(self) -> None:
        """Record an iteration."""
        self.iterations_used += 1

    def use_calls(self, count: int) -> None:
        """Record tool calls."""
        self.calls_used += count

    def exceeded(self) -> bool:
        """Check if any budget is exceeded."""
        return (
            self.iterations_used >= self.max_iterations
            or self.calls_used >= self.max_total_calls
        )


# Type alias for tool call union
ToolCallUnion = Union[ToolCall, ToolCallBatch]


__all__ = [
    "PolicyClass",
    "ToolCall",
    "ToolCallBatch",
    "ToolResult",
    "ArgSchema",
    "ToolDefinition",
    "ToolBudget",
    "ToolCallUnion",
]
