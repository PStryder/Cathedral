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
    get_prompt_config,
    set_custom_prompt,
    restore_default_prompt,
    get_prompt_version,
    get_edit_warning,
    get_json_correction_message,
    ToolPromptManager,
    DEFAULT_TOOL_PROTOCOL_PROMPT,
    PROMPT_VERSION,
    TOOL_CALL_EXAMPLE,
    JSON_CORRECTION_MESSAGE,
    validate_prompt,
    is_prompt_functional,
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

    # Get prompt status
    ToolPromptManager.initialize()
    prompt_config = ToolPromptManager.get_config()

    return build_health_status(
        gate_name="ToolGate",
        initialized=_initialized,
        dependencies=[],
        checks={
            "registry_loaded": _initialized,
            "tools_available": tool_count > 0,
            "prompt_functional": not prompt_config.get("is_using_fallback", False),
        },
        details={
            "tool_count": tool_count,
            "enabled_policies": [p.value for p in get_policy_manager().get_enabled_policies()],
            "prompt_version": PROMPT_VERSION,
            "prompt_is_custom": prompt_config.get("is_custom", False),
            "prompt_using_fallback": prompt_config.get("is_using_fallback", False),
        },
    )


def get_dependencies() -> List[str]:
    """List ToolGate dependencies."""
    return []


def get_info() -> dict:
    """
    Get comprehensive documentation for ToolGate and the entire tool system.

    Returns complete documentation including protocol format, policy system,
    available gates, and how to discover tool documentation.
    """
    initialize()

    # Get tool counts per gate
    all_tools = ToolRegistry.list_tools()
    gate_tools = {}
    for tool in all_tools:
        gate = tool.gate
        if gate not in gate_tools:
            gate_tools[gate] = []
        gate_tools[gate].append(tool.name)

    return {
        "gate": "ToolGate",
        "version": "1.0",
        "purpose": "Provider-agnostic tool calling protocol for AI agents interacting with Cathedral Gates. Handles tool discovery, validation, policy enforcement, and execution orchestration.",

        "protocol": {
            "description": "Tools are called via JSON objects in model responses",
            "call_format": {
                "type": "tool_call",
                "id": "string - unique call ID (auto-generated if omitted)",
                "tool": "string - Gate.method format (e.g., 'MemoryGate.search')",
                "args": "object - method arguments",
            },
            "result_format": {
                "type": "tool_result",
                "id": "string - matches call ID",
                "ok": "boolean - true if successful",
                "result": "any - method return value (if ok=true)",
                "error": "string - error message (if ok=false)",
            },
            "example_call": '{"type": "tool_call", "id": "tc_001", "tool": "MemoryGate.search", "args": {"query": "python tutorials"}}',
            "example_result": '{"type": "tool_result", "id": "tc_001", "ok": true, "result": {"items": [...], "total": 5}}',
        },

        "policy_classes": {
            "description": "Tools are categorized by security policy. Only enabled policies can be called.",
            "classes": {
                "read_only": "Safe read operations - no side effects (e.g., search, list, get)",
                "write": "Modifies data within Cathedral (e.g., store, update, delete)",
                "network": "External network access (e.g., web search, API calls)",
                "privileged": "System operations requiring elevated trust (e.g., shell commands)",
            },
            "note": "The orchestrator enforces policies at runtime. Calls to disabled policy tools return errors.",
        },

        "available_gates": {
            gate: {
                "tool_count": len(tools),
                "tools": tools,
                "info_endpoint": f"{gate}.get_info",
            }
            for gate, tools in sorted(gate_tools.items())
        },

        "discovering_tools": {
            "description": "Each gate provides a get_info() method with comprehensive documentation",
            "steps": [
                "1. Call ToolGate.get_info() (this) to see available gates",
                "2. Call {Gate}.get_info() for detailed tool documentation",
                "3. Each get_info() includes: purpose, call_format, response schema, examples",
            ],
            "example": "To learn about memory tools, call MemoryGate.get_info()",
        },

        "orchestrator": {
            "description": "The ToolOrchestrator executes tool calls in a loop until completion",
            "features": [
                "Parses tool calls from model output",
                "Validates arguments against schemas",
                "Enforces policy restrictions",
                "Handles parallel and sequential calls",
                "Injects results back into conversation",
            ],
            "limits": {
                "max_iterations": "Default 6 - prevents infinite loops",
                "max_calls_per_step": "Default 5 - batch size limit",
            },
        },

        "best_practices": [
            "Always call get_info() on a gate before using its tools",
            "Use specific tool names (Gate.method) not just method names",
            "Check 'ok' field in results before using 'result' data",
            "Handle errors gracefully - tools can fail",
            "Prefer read_only tools when write isn't needed",
            "Use parallel calls for independent operations",
        ],
    }


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
    gate_filter: Optional[List[str]] = None,
) -> ToolOrchestrator:
    """
    Get a configured tool orchestrator.

    Args:
        enabled_policies: Policies to enable
        max_iterations: Max loop iterations
        max_calls_per_step: Max calls per step
        emit_event: Event callback
        gate_filter: Optional list of gate names to enable (e.g., ["MemoryGate", "ShellGate"])

    Returns:
        Configured ToolOrchestrator
    """
    initialize()
    return create_orchestrator(
        enabled_policies=enabled_policies,
        max_iterations=max_iterations,
        max_calls_per_step=max_calls_per_step,
        emit_event=emit_event,
        gate_filter=gate_filter,
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
    "get_info",
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
    # Prompt Configuration
    "get_prompt_config",
    "set_custom_prompt",
    "restore_default_prompt",
    "get_prompt_version",
    "get_edit_warning",
    "get_json_correction_message",
    "ToolPromptManager",
    "DEFAULT_TOOL_PROTOCOL_PROMPT",
    "PROMPT_VERSION",
    "TOOL_CALL_EXAMPLE",
    "JSON_CORRECTION_MESSAGE",
    "validate_prompt",
    "is_prompt_functional",
]
