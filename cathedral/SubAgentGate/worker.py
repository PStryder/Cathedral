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


AGENT_DATA_DIR = PROJECT_ROOT / "data" / "agents"


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

    # Load personality if specified
    personality = None
    if personality_id:
        try:
            from cathedral import PersonalityGate
            PersonalityGate.initialize()
            personality = PersonalityGate.load(personality_id)
            if personality:
                print(f"[SubAgent] Using personality: {personality.name}")
        except Exception as e:
            print(f"[SubAgent] Could not load personality: {e}")

    # Use personality settings if available
    if personality:
        system_prompt = personality.get_system_prompt()
        temperature = personality.llm_config.temperature
        max_tokens = personality.llm_config.max_tokens
        model = personality.llm_config.model
    else:
        # Build default system prompt for bounded agent
        default_system = """You are a focused sub-agent with a specific task to complete.
Complete the task thoroughly and return your findings/results.
Be concise but comprehensive. Focus only on the assigned task."""

        if system_prompt:
            system_prompt = f"{default_system}\n\n{system_prompt}"
        else:
            system_prompt = default_system

    # Add context if provided
    if context:
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        system_prompt += f"\n\nContext:\n{context_str}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

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
    print(f"[SubAgent {agent_id}] Starting...")

    try:
        # Load task
        task_data = load_task(agent_id)
        print(f"[SubAgent {agent_id}] Task: {task_data['task'][:100]}...")

        # Execute
        result = await run_task(task_data)
        print(f"[SubAgent {agent_id}] Completed, result length: {len(result)}")

        # Save result
        save_result(agent_id, "completed", result=result)

    except Exception as e:
        print(f"[SubAgent {agent_id}] Failed: {e}")
        save_result(agent_id, "failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python worker.py <agent_id>")
        sys.exit(1)

    agent_id = sys.argv[1]
    asyncio.run(main(agent_id))
