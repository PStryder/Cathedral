"""
ToolGate Prompt Builder.

Generates system prompt sections for tool instructions.

The tool protocol prompt is configurable but protected:
- Default prompt is versioned and validated
- Custom prompts require explicit risk acknowledgment
- Broken prompts fall back to default safely
"""

from __future__ import annotations

from typing import Optional, Set

from cathedral.ToolGate.models import PolicyClass
from cathedral.ToolGate.registry import ToolRegistry
from cathedral.ToolGate.prompt_config import (
    ToolPromptManager,
    DEFAULT_TOOL_PROTOCOL_PROMPT,
    PROMPT_VERSION,
    validate_prompt,
    is_prompt_functional,
    EDIT_WARNING,
)


# =============================================================================
# Example and Correction Messages
# =============================================================================

TOOL_CALL_EXAMPLE = """
## Example

If asked to search memory, you would respond with ONLY:
{"type": "tool_call", "id": "tc_1", "tool": "MemoryGate.search", "args": {"query": "example", "limit": 5}}

Then wait for the result before continuing.
"""

JSON_CORRECTION_MESSAGE = """Your previous response contained invalid JSON for a tool call.

Please re-emit ONLY the valid JSON tool call, with no other text.

Remember the format:
{"type": "tool_call", "id": "<unique_id>", "tool": "<ToolName>", "args": {...}}

Emit only the JSON, nothing else."""


# =============================================================================
# Prompt Building
# =============================================================================


def build_tool_prompt(
    enabled_policies: Optional[Set[PolicyClass]] = None,
    max_calls: int = 5,
    include_schemas: bool = True,
    include_example: bool = True,
    use_custom_prompt: bool = True,
) -> str:
    """
    Build the system prompt section for tool usage.

    Combines the configurable tool protocol prompt with the tool schema reference.

    Args:
        enabled_policies: Set of enabled policy classes
        max_calls: Maximum tool calls per response
        include_schemas: Whether to include full tool schemas
        include_example: Whether to include example tool call
        use_custom_prompt: Whether to use custom prompt (if configured)

    Returns:
        System prompt section for tool instructions
    """
    if enabled_policies is None:
        enabled_policies = {PolicyClass.READ_ONLY}

    # Initialize registry and prompt manager
    ToolRegistry.initialize()
    ToolPromptManager.initialize()

    # Get the protocol prompt (configurable)
    if use_custom_prompt:
        protocol_prompt = ToolPromptManager.get_prompt()
    else:
        protocol_prompt = DEFAULT_TOOL_PROTOCOL_PROMPT

    # Build tool reference section
    sections = [protocol_prompt]

    # Add example if requested
    if include_example:
        sections.append(TOOL_CALL_EXAMPLE)

    sections.append("\n### Available Tools\n")

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

    # Add max calls note
    sections.append(f"\n*Maximum {max_calls} tool calls per response.*")

    return "\n".join(sections)


def get_json_correction_message() -> str:
    """Get the message to inject when model emits invalid JSON."""
    return JSON_CORRECTION_MESSAGE


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


# =============================================================================
# Prompt Configuration API
# =============================================================================


def get_prompt_config() -> dict:
    """Get the current prompt configuration."""
    ToolPromptManager.initialize()
    return ToolPromptManager.get_config()


def set_custom_prompt(prompt: str, acknowledge_risk: bool = False) -> tuple[bool, str]:
    """
    Set a custom tool protocol prompt.

    WARNING: Breaking this prompt will disable tool calling.

    Args:
        prompt: The new prompt content
        acknowledge_risk: Must be True to proceed

    Returns:
        Tuple of (success, message)
    """
    ToolPromptManager.initialize()
    return ToolPromptManager.set_custom_prompt(prompt, acknowledge_risk)


def restore_default_prompt() -> tuple[bool, str]:
    """Restore the default tool protocol prompt."""
    ToolPromptManager.initialize()
    return ToolPromptManager.restore_default()


def get_prompt_version() -> str:
    """Get the current prompt version."""
    return PROMPT_VERSION


def get_edit_warning() -> str:
    """Get the warning message for editing the prompt."""
    return EDIT_WARNING


__all__ = [
    # Prompt building
    "build_tool_prompt",
    "build_minimal_tool_prompt",
    "get_tool_count",
    "get_json_correction_message",
    # Example and correction messages
    "TOOL_CALL_EXAMPLE",
    "JSON_CORRECTION_MESSAGE",
    # Prompt configuration
    "get_prompt_config",
    "set_custom_prompt",
    "restore_default_prompt",
    "get_prompt_version",
    "get_edit_warning",
    # Re-exports from prompt_config
    "ToolPromptManager",
    "DEFAULT_TOOL_PROTOCOL_PROMPT",
    "PROMPT_VERSION",
    "validate_prompt",
    "is_prompt_functional",
]
