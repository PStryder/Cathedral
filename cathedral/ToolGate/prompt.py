"""
ToolGate Prompt Builder.

Generates system prompt sections for tool instructions.
"""

from __future__ import annotations

from typing import Optional, Set

from cathedral.ToolGate.models import PolicyClass
from cathedral.ToolGate.registry import ToolRegistry


# =============================================================================
# Prompt Templates
# =============================================================================

TOOL_PROMPT_HEADER = """## Available Tools

You have access to tools that extend your capabilities. When you need to use a tool, output a JSON object in your response.

### Tool Call Format

For a single tool call:
```json
{{"type": "tool_call", "id": "tc_<unique_id>", "tool": "<GateName>.<method>", "args": {{...}}}}
```

For multiple tool calls:
```json
{{"type": "tool_calls", "calls": [
  {{"type": "tool_call", "id": "tc_1", "tool": "...", "args": {{...}}}},
  {{"type": "tool_call", "id": "tc_2", "tool": "...", "args": {{...}}}}
]}}
```

### Important Guidelines

1. **Use tools only when needed** - Don't call tools for information you already have
2. **Provide complete arguments** - Include all required arguments
3. **Unique IDs** - Each call needs a unique id (e.g., tc_abc, tc_123)
4. **Handle errors gracefully** - If a tool fails, acknowledge and adapt
5. **Maximum {max_calls} calls per response** - Batch related calls together
6. **Wait for results** - After emitting tool calls, results will be provided before you continue

### Tool Results

Results are provided in this format:
```
[TOOL RESULTS]
Tool tc_1: SUCCESS
Result: {{...}}

Tool tc_2: FAILED
Error: <reason>
[/TOOL RESULTS]
```

Continue your response after receiving results.
"""

TOOL_PROMPT_FOOTER = """
---

When not using tools, respond normally with text.
"""


# =============================================================================
# Prompt Building
# =============================================================================


def build_tool_prompt(
    enabled_policies: Optional[Set[PolicyClass]] = None,
    max_calls: int = 5,
    include_schemas: bool = True,
) -> str:
    """
    Build the system prompt section for tool usage.

    Args:
        enabled_policies: Set of enabled policy classes
        max_calls: Maximum tool calls per response
        include_schemas: Whether to include full tool schemas

    Returns:
        System prompt section for tool instructions
    """
    if enabled_policies is None:
        enabled_policies = {PolicyClass.READ_ONLY}

    # Initialize registry
    ToolRegistry.initialize()

    # Build header with max_calls
    header = TOOL_PROMPT_HEADER.format(max_calls=max_calls)

    # Build tool reference
    sections = [header, "### Available Tools\n"]

    if include_schemas:
        # Full schemas
        tool_docs = ToolRegistry.get_tools_for_prompt(enabled_policies)
        if tool_docs:
            sections.append(tool_docs)
        else:
            sections.append("*No tools available with current permissions.*")
    else:
        # Just tool names
        tool_names = ToolRegistry.list_tool_names(enabled_policies)
        if tool_names:
            for name in sorted(tool_names):
                sections.append(f"- `{name}`")
        else:
            sections.append("*No tools available with current permissions.*")

    sections.append(TOOL_PROMPT_FOOTER)

    return "\n".join(sections)


def build_minimal_tool_prompt(enabled_policies: Optional[Set[PolicyClass]] = None) -> str:
    """
    Build a minimal tool prompt (just format, no schemas).

    Useful when token budget is constrained.

    Args:
        enabled_policies: Set of enabled policy classes

    Returns:
        Minimal tool prompt
    """
    if enabled_policies is None:
        enabled_policies = {PolicyClass.READ_ONLY}

    ToolRegistry.initialize()
    tool_names = ToolRegistry.list_tool_names(enabled_policies)

    if not tool_names:
        return ""

    tools_list = ", ".join(f"`{name}`" for name in sorted(tool_names)[:10])
    if len(tool_names) > 10:
        tools_list += f" (+{len(tool_names) - 10} more)"

    return f"""## Tools

You can call tools using JSON: {{"type": "tool_call", "id": "tc_X", "tool": "Name", "args": {{...}}}}

Available: {tools_list}

Results will be provided after each call. Continue your response after receiving results.
"""


def get_tool_count(enabled_policies: Optional[Set[PolicyClass]] = None) -> int:
    """Get count of available tools for given policies."""
    if enabled_policies is None:
        enabled_policies = {PolicyClass.READ_ONLY}

    ToolRegistry.initialize()
    return len(ToolRegistry.list_tools(enabled_policies))


__all__ = [
    "build_tool_prompt",
    "build_minimal_tool_prompt",
    "get_tool_count",
]
