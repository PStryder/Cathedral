#!/usr/bin/env python
"""
SubAgent Worker - Runs as a subprocess to handle delegated tasks.

Usage: python worker.py <agent_id>

Reads task from data/agents/<agent_id>.task.json
Writes result to data/agents/<agent_id>.json
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("SubAgent")


AGENT_DATA_DIR = PROJECT_ROOT / "data" / "agents"

# Limits for context injection to prevent prompt explosion
MAX_CONTEXT_VALUE_LEN = 500
MAX_CONTEXT_TOTAL_LEN = 2000


def _format_context_value(value, max_len: int = MAX_CONTEXT_VALUE_LEN) -> str:
    """
    Safely format a context value for prompt injection.

    - Strings are truncated with ellipsis
    - Dicts/lists are serialized as compact JSON and truncated
    - Other types are converted to string and truncated
    """
    if value is None:
        return "null"

    if isinstance(value, str):
        if len(value) > max_len:
            return value[:max_len - 3] + "..."
        return value

    if isinstance(value, (dict, list)):
        try:
            # Compact JSON, no extra whitespace
            serialized = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            if len(serialized) > max_len:
                return serialized[:max_len - 3] + "..."
            return serialized
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            fallback = str(value)
            if len(fallback) > max_len:
                return fallback[:max_len - 3] + "..."
            return fallback

    # Numbers, booleans, etc.
    s = str(value)
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


def _format_context(context: dict, max_total: int = MAX_CONTEXT_TOTAL_LEN) -> str:
    """
    Format context dict for safe prompt injection.

    Returns formatted string with truncated values and total length cap.
    """
    if not context:
        return ""

    lines = []
    current_len = 0

    for k, v in context.items():
        formatted = _format_context_value(v)
        line = f"- {k}: {formatted}"

        # Check if adding this line would exceed total limit
        if current_len + len(line) + 1 > max_total:
            # Add truncation notice and stop
            lines.append("- [additional context truncated]")
            break

        lines.append(line)
        current_len += len(line) + 1  # +1 for newline

    return "\n".join(lines)


def load_task(agent_id: str) -> dict:
    """Load task definition from file."""
    task_file = AGENT_DATA_DIR / f"{agent_id}.task.json"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    with open(task_file) as f:
        return json.load(f)


def save_result(agent_id: str, status: str, result: str = None, error: str = None):
    """Save result to file."""
    result_file = AGENT_DATA_DIR / f"{agent_id}.json"
    data = {
        "id": agent_id,
        "status": status,
        "result": result,
        "error": error,
        "completed_at": datetime.utcnow().isoformat()
    }
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)


async def run_task(task_data: dict) -> str:
    """Execute the task using StarMirror."""
    # Import here to avoid loading everything at module level
    from cathedral.StarMirror.StarMirrorGate import transmit_async

    task = task_data["task"]
    context = task_data.get("context", {})
    system_prompt = task_data.get("system_prompt")
    max_tokens = task_data.get("max_tokens", 2000)
    temperature = task_data.get("temperature", 0.7)
    personality_id = task_data.get("personality")
    model = task_data.get("model")

    # Extract conversation history if this is a continued conversation
    conversation_history = context.pop("conversation_history", None)
    parent_agent_id = context.pop("parent_agent_id", None)

    # Load personality if specified
    personality = None
    if personality_id:
        try:
            from cathedral import PersonalityGate
            PersonalityGate.initialize()
            personality = PersonalityGate.load(personality_id)
            if personality:
                _log.info(f"Using personality: {personality.name}")
        except Exception as e:
            _log.warning(f"Could not load personality: {e}")

    # Use personality settings if available
    if personality:
        system_prompt = personality.get_system_prompt()
        temperature = personality.llm_config.temperature
        max_tokens = personality.llm_config.max_tokens
        model = personality.llm_config.model
    else:
        # Build default system prompt for bounded agent
        if conversation_history:
            # Continued conversation - adjust prompt
            default_system = """You are a focused sub-agent continuing a conversation.
You have context from a previous exchange. Continue assisting based on the
new follow-up message while maintaining continuity with the prior discussion.
Be concise but comprehensive."""
        else:
            default_system = """You are a focused sub-agent with a specific task to complete.
Complete the task thoroughly and return your findings/results.
Be concise but comprehensive. Focus only on the assigned task."""

        if system_prompt:
            system_prompt = f"{default_system}\n\n{system_prompt}"
        else:
            system_prompt = default_system

    # Add context if provided (safely truncated to prevent prompt explosion)
    # Note: conversation_history was already popped out
    if context:
        context_str = _format_context(context)
        if context_str:
            system_prompt += f"\n\nContext:\n{context_str}"

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if this is a continued conversation
    if conversation_history:
        _log.info(f"Continuing from parent agent {parent_agent_id}, {len(conversation_history)} prior turns")
        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Add current task/message
    messages.append({"role": "user", "content": task})

    # Build kwargs
    kwargs = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if model:
        kwargs["model"] = model

    # Execute
    result = await transmit_async(**kwargs)

    return result


async def main(agent_id: str):
    """Main entry point."""
    _log.info(f"[{agent_id}] Starting...")

    try:
        # Load task
        task_data = load_task(agent_id)
        _log.info(f"[{agent_id}] Task: {task_data['task'][:100]}...")

        # Execute
        result = await run_task(task_data)
        _log.info(f"[{agent_id}] Completed, result length: {len(result)}")

        # Save result
        save_result(agent_id, "completed", result=result)

    except Exception as e:
        _log.error(f"[{agent_id}] Failed: {e}")
        save_result(agent_id, "failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python worker.py <agent_id>")
        sys.exit(1)

    agent_id = sys.argv[1]
    asyncio.run(main(agent_id))
