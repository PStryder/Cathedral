"""
ToolGate - Cathedral Tool Call Protocol.

Provides provider-agnostic tool calling for AI agents interacting with Cathedral Gates.

## Usage

```python
from cathedral.ToolGate import (
    build_tool_prompt,
    create_orchestrator,
    PolicyClass,
)

# Build system prompt section
prompt = build_tool_prompt(enabled_policies={PolicyClass.READ_ONLY})

# Create orchestrator for tool execution
orchestrator = create_orchestrator(
    enabled_policies=[PolicyClass.READ_ONLY, PolicyClass.WRITE],
    max_iterations=6,
)

# Execute tool loop
async for token in orchestrator.execute_loop(
    initial_response=model_output,
    messages=conversation_history,
    model="gpt-4",
    temperature=0.7,
):
    print(token, end="")
```

## Protocol Format

Tool calls are JSON objects in the model response:

```json
{"type": "tool_call", "id": "tc_123", "tool": "MemoryGate.search", "args": {"query": "foo"}}
```

Results are injected back into the conversation:

```json
{"type": "tool_result", "id": "tc_123", "ok": true, "result": {...}}
```
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set

from cathedral.shared.gate import GateLogger, build_health_status

# Models
from cathedral.ToolGate.models import (
    ArgSchema,
    PolicyClass,
    ToolBudget,
    ToolCall,
    ToolCallBatch,
    ToolCallUnion,
    ToolDefinition,
    ToolResult,
)

# Registry
from cathedral.ToolGate.registry import ToolRegistry

# Protocol
from cathedral.ToolGate.protocol import (
    format_tool_error,
    format_tool_results,
    generate_call_id,
    is_tool_response,
    parse_tool_calls,
    validate_args,
)

# Policy
from cathedral.ToolGate.policy import (
    PolicyManager,
    get_policy_manager,
    reset_policy_manager,
)

# Orchestrator
from cathedral.ToolGate.orchestrator import (
    ToolOrchestrator,
    create_orchestrator,
)

# Prompt
from cathedral.ToolGate.prompt import (
    build_minimal_tool_prompt,
    build_tool_prompt,
    get_tool_count,
)

_log = GateLogger.get("ToolGate")
_initialized = False


# =============================================================================
# Public API
# =============================================================================


def initialize() -> bool:
    """
    Initialize ToolGate.

    Registers all tools from Gates and sets up the policy manager.

    Returns:
        True if initialization successful
    """
    global _initialized

    if _initialized:
        return True

    try:
        ToolRegistry.initialize()
        _initialized = True
        _log.info("ToolGate initialized")
        return True
    except Exception as e:
        _log.error(f"ToolGate initialization failed: {e}")
        return False


def is_initialized() -> bool:
    """Check if ToolGate is initialized."""
    return _initialized


def is_healthy() -> bool:
    """Check if ToolGate is operational."""
    return _initialized and len(ToolRegistry.list_tools()) > 0


def get_health_status() -> dict:
    """Get detailed health status."""
    tool_count = len(ToolRegistry.list_tools()) if _initialized else 0

    return build_health_status(
        gate_name="ToolGate",
        initialized=_initialized,
        dependencies=[],
        checks={
            "registry_loaded": _initialized,
            "tools_available": tool_count > 0,
        },
        details={
            "tool_count": tool_count,
            "enabled_policies": [p.value for p in get_policy_manager().get_enabled_policies()],
        },
    )


def get_dependencies() -> List[str]:
    """List ToolGate dependencies."""
    return []


def list_tools(
    policy_filter: Optional[Set[PolicyClass]] = None,
    gate_filter: Optional[str] = None,
) -> List[ToolDefinition]:
    """
    List available tools.

    Args:
        policy_filter: Filter by policy classes
        gate_filter: Filter by gate name

    Returns:
        List of tool definitions
    """
    initialize()
    return ToolRegistry.list_tools(policy_filter, gate_filter)


def list_tool_names(policy_filter: Optional[Set[PolicyClass]] = None) -> List[str]:
    """List available tool names."""
    initialize()
    return ToolRegistry.list_tool_names(policy_filter)


def get_tool(name: str) -> Optional[ToolDefinition]:
    """Get a tool by name."""
    initialize()
    return ToolRegistry.get_tool(name)


def get_orchestrator(
    enabled_policies: Optional[List[PolicyClass]] = None,
    max_iterations: int = 6,
    max_calls_per_step: int = 5,
    emit_event: Optional[Callable] = None,
) -> ToolOrchestrator:
    """
    Get a configured tool orchestrator.

    Args:
        enabled_policies: Policies to enable
        max_iterations: Max loop iterations
        max_calls_per_step: Max calls per step
        emit_event: Event callback

    Returns:
        Configured ToolOrchestrator
    """
    initialize()
    return create_orchestrator(
        enabled_policies=enabled_policies,
        max_iterations=max_iterations,
        max_calls_per_step=max_calls_per_step,
        emit_event=emit_event,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Initialization
    "initialize",
    "is_initialized",
    "is_healthy",
    "get_health_status",
    "get_dependencies",
    # Models
    "PolicyClass",
    "ToolCall",
    "ToolCallBatch",
    "ToolResult",
    "ToolDefinition",
    "ToolBudget",
    "ArgSchema",
    "ToolCallUnion",
    # Registry
    "ToolRegistry",
    "list_tools",
    "list_tool_names",
    "get_tool",
    # Protocol
    "parse_tool_calls",
    "validate_args",
    "format_tool_results",
    "format_tool_error",
    "generate_call_id",
    "is_tool_response",
    # Policy
    "PolicyManager",
    "get_policy_manager",
    "reset_policy_manager",
    # Orchestrator
    "ToolOrchestrator",
    "create_orchestrator",
    "get_orchestrator",
    # Prompt
    "build_tool_prompt",
    "build_minimal_tool_prompt",
    "get_tool_count",
]
