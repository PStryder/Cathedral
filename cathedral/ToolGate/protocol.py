"""
ToolGate Protocol Parser.

Parses tool calls from model output and formats results for injection.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from cathedral.shared.gate import GateLogger
from cathedral.ToolGate.models import (
    ArgSchema,
    ToolCall,
    ToolCallBatch,
    ToolDefinition,
    ToolResult,
)

_log = GateLogger.get("ToolGate")


# =============================================================================
# JSON Extraction Patterns
# =============================================================================

# Pattern to find JSON blocks (with or without markdown fences)
JSON_BLOCK_PATTERN = re.compile(
    r'```(?:json)?\s*\n?({[\s\S]*?})\n?```|({[\s\S]*?"type"\s*:\s*"tool_call[s]?"[\s\S]*?}(?:\s*})?)',
    re.MULTILINE,
)

# Simpler pattern for tool_call objects
TOOL_CALL_PATTERN = re.compile(
    r'\{[^{}]*"type"\s*:\s*"tool_call(?:s)?"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
    re.DOTALL,
)


def _extract_json_objects(text: str) -> List[str]:
    """
    Extract potential JSON objects from text.

    Returns list of JSON string candidates.
    """
    candidates = []

    # Try markdown code blocks first
    for match in re.finditer(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text):
        content = match.group(1).strip()
        if content.startswith('{'):
            candidates.append(content)

    # Try to find raw JSON objects with tool_call type
    for match in re.finditer(r'\{[^{}]*"type"\s*:\s*"tool_call', text):
        start = match.start()
        # Find matching brace
        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end > start:
            candidate = text[start:end]
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


def _try_parse_tool_call(json_str: str) -> Optional[ToolCall]:
    """Try to parse a single tool call from JSON string."""
    try:
        data = json.loads(json_str)
        if data.get("type") == "tool_call":
            return ToolCall.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        _log.debug(f"Failed to parse tool call: {e}")
    return None


def _try_parse_tool_batch(json_str: str) -> Optional[ToolCallBatch]:
    """Try to parse a batch of tool calls from JSON string."""
    try:
        data = json.loads(json_str)
        if data.get("type") == "tool_calls":
            return ToolCallBatch.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        _log.debug(f"Failed to parse tool batch: {e}")
    return None


# =============================================================================
# Main Parsing Functions
# =============================================================================


def parse_tool_calls(text: str) -> Tuple[str, List[ToolCall]]:
    """
    Parse tool calls from model output text.

    Args:
        text: Model output text that may contain tool calls

    Returns:
        Tuple of (remaining_text, list_of_tool_calls)
        - remaining_text: Text with tool call JSON removed
        - list_of_tool_calls: Extracted ToolCall objects
    """
    if not text:
        return text, []

    # Quick check for tool call markers
    if '"type"' not in text or 'tool_call' not in text:
        return text, []

    tool_calls = []
    seen_ids = set()  # Track seen call IDs to deduplicate
    positions_to_remove = []

    # Extract JSON candidates
    candidates = _extract_json_objects(text)

    for json_str in candidates:
        # Try batch first (to avoid duplicating individual calls from within batch)
        batch = _try_parse_tool_batch(json_str)
        if batch:
            for call in batch.calls:
                if call.id not in seen_ids:
                    tool_calls.append(call)
                    seen_ids.add(call.id)
            pos = text.find(json_str)
            if pos >= 0:
                positions_to_remove.append((pos, pos + len(json_str)))
            continue

        # Try single call
        call = _try_parse_tool_call(json_str)
        if call and call.id not in seen_ids:
            tool_calls.append(call)
            seen_ids.add(call.id)
            # Find position in original text
            pos = text.find(json_str)
            if pos >= 0:
                positions_to_remove.append((pos, pos + len(json_str)))
            continue

        # Try batch
        batch = _try_parse_tool_batch(json_str)
        if batch:
            tool_calls.extend(batch.calls)
            pos = text.find(json_str)
            if pos >= 0:
                positions_to_remove.append((pos, pos + len(json_str)))

    # Also check for markdown-fenced tool calls
    for match in re.finditer(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text):
        full_match = match.group(0)
        if full_match not in [text[s:e] for s, e in positions_to_remove]:
            content = match.group(1).strip()
            if '"type"' in content and 'tool_call' in content:
                positions_to_remove.append((match.start(), match.end()))

    # Build remaining text by removing tool call regions
    if positions_to_remove:
        # Sort by position, remove in reverse to preserve indices
        positions_to_remove.sort(key=lambda x: x[0], reverse=True)
        remaining = text
        for start, end in positions_to_remove:
            remaining = remaining[:start] + remaining[end:]
        remaining = remaining.strip()
    else:
        remaining = text

    return remaining, tool_calls


def validate_args(tool: ToolDefinition, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate arguments against tool's schema.

    Args:
        tool: Tool definition with schema
        args: Arguments to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    schema = tool.args_schema

    # Check required arguments
    for arg_name, arg_schema in schema.items():
        if arg_schema.required and arg_name not in args:
            return False, f"Missing required argument: {arg_name}"

    # Type checking (basic)
    for arg_name, value in args.items():
        if arg_name not in schema:
            # Unknown argument - could be strict or lenient
            continue

        arg_schema = schema[arg_name]
        expected_type = arg_schema.type

        if value is None and not arg_schema.required:
            continue

        # Basic type validation
        if expected_type == "string" and not isinstance(value, str):
            return False, f"Argument {arg_name} must be string, got {type(value).__name__}"
        elif expected_type == "integer" and not isinstance(value, int):
            return False, f"Argument {arg_name} must be integer, got {type(value).__name__}"
        elif expected_type == "number" and not isinstance(value, (int, float)):
            return False, f"Argument {arg_name} must be number, got {type(value).__name__}"
        elif expected_type == "boolean" and not isinstance(value, bool):
            return False, f"Argument {arg_name} must be boolean, got {type(value).__name__}"
        elif expected_type == "array" and not isinstance(value, list):
            return False, f"Argument {arg_name} must be array, got {type(value).__name__}"
        elif expected_type == "object" and not isinstance(value, dict):
            return False, f"Argument {arg_name} must be object, got {type(value).__name__}"

        # Enum validation
        if arg_schema.enum and value not in arg_schema.enum:
            return False, f"Argument {arg_name} must be one of {arg_schema.enum}"

    return True, None


def format_tool_results(results: List[ToolResult]) -> str:
    """
    Format tool results for injection into conversation.

    Args:
        results: List of tool execution results

    Returns:
        Formatted string for conversation injection
    """
    if not results:
        return ""

    formatted = []

    for result in results:
        if result.ok:
            # Truncate large results
            result_str = json.dumps(result.result, default=str, ensure_ascii=False)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "...[truncated]"

            formatted.append(
                f"Tool {result.id}: SUCCESS\n"
                f"Result: {result_str}"
            )
        else:
            formatted.append(
                f"Tool {result.id}: FAILED\n"
                f"Error: {result.error}"
            )

    header = "[TOOL RESULTS]"
    body = "\n\n".join(formatted)
    footer = "[/TOOL RESULTS]"

    return f"{header}\n{body}\n{footer}"


def format_tool_error(call_id: str, error: str) -> str:
    """Format a single tool error for the model."""
    return json.dumps({
        "type": "tool_result",
        "id": call_id,
        "ok": False,
        "error": error
    })


# =============================================================================
# Utility Functions
# =============================================================================


def generate_call_id() -> str:
    """Generate a unique tool call ID."""
    import uuid
    return f"tc_{uuid.uuid4().hex[:8]}"


def is_tool_response(text: str) -> bool:
    """Quick check if text contains tool calls."""
    return '"type"' in text and 'tool_call' in text


__all__ = [
    "parse_tool_calls",
    "validate_args",
    "format_tool_results",
    "format_tool_error",
    "generate_call_id",
    "is_tool_response",
]
