#!/usr/bin/env python
"""
CLI Agent Worker - Runs Claude Code or Codex CLI as autonomous agents.

Usage: python cli_worker.py <agent_id>

Reads task from data/agents/<agent_id>.task.json
Writes result to data/agents/<agent_id>.json

This worker spawns the actual CLI tool (claude or codex) and captures
its output. The CLI runs in agentic mode with full tool access.
"""

import os
import sys
import json
import shutil
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("SubAgent.CLI")


AGENT_DATA_DIR = PROJECT_ROOT / "data" / "agents"

# CLI configuration
CLAUDE_CMD_ENV = "CLAUDE_CLI_CMD"
CODEX_CMD_ENV = "CODEX_CLI_CMD"
DEFAULT_CLAUDE_CMD = "claude"
DEFAULT_CODEX_CMD = "codex"

# Timeout for CLI execution (30 minutes default)
CLI_TIMEOUT = int(os.getenv("SUBAGENT_CLI_TIMEOUT", "1800"))


def _find_executable(name: str, env_var: str, default: str) -> Optional[str]:
    """Find CLI executable, checking env var, PATH, and common locations."""
    # Check environment variable first
    from_env = os.getenv(env_var)
    if from_env:
        if Path(from_env).exists() or shutil.which(from_env):
            return from_env

    # Check if default is in PATH
    if shutil.which(default):
        return default

    # Check common installation locations
    common_paths = [
        # npm global
        Path.home() / ".npm-global" / "bin" / name,
        Path.home() / "AppData" / "Roaming" / "npm" / f"{name}.cmd",  # Windows npm
        Path.home() / "AppData" / "Roaming" / "npm" / name,
        # Local node_modules
        PROJECT_ROOT / "node_modules" / ".bin" / name,
        PROJECT_ROOT / "node_modules" / ".bin" / f"{name}.cmd",
        # Homebrew
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
    ]

    for path in common_paths:
        if path.exists():
            return str(path)

    return None


def _get_claude_cmd() -> List[str]:
    """Get Claude CLI command."""
    executable = _find_executable("claude", CLAUDE_CMD_ENV, DEFAULT_CLAUDE_CMD)
    if not executable:
        raise RuntimeError(
            f"Claude CLI not found. Install it with 'npm install -g @anthropic-ai/claude-code' "
            f"or set {CLAUDE_CMD_ENV} environment variable."
        )
    return [executable]


def _get_codex_cmd() -> List[str]:
    """Get Codex CLI command."""
    executable = _find_executable("codex", CODEX_CMD_ENV, DEFAULT_CODEX_CMD)
    if not executable:
        raise RuntimeError(
            f"Codex CLI not found. Install it or set {CODEX_CMD_ENV} environment variable."
        )
    return [executable]


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


def _build_prompt(task: str, context: dict) -> str:
    """Build the prompt to send to the CLI."""
    parts = [task]

    if context:
        parts.append("\n\nContext:")
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
            parts.append(f"- {key}: {value}")

    return "\n".join(parts)


async def run_claude_code(task_data: dict) -> str:
    """
    Run Claude Code CLI in agentic mode.

    Claude Code will:
    - Have full file system access in the working directory
    - Be able to read/write files
    - Execute shell commands
    - Use its full toolset autonomously
    """
    task = task_data["task"]
    context = task_data.get("context", {})
    working_dir = task_data.get("working_dir", str(PROJECT_ROOT))

    prompt = _build_prompt(task, context)

    # Get Claude CLI command
    cmd = _get_claude_cmd()

    # Add flags for non-interactive agentic mode
    # --print: Output to stdout instead of interactive UI
    # -p: Pass prompt directly
    # --allowedTools: Allow all tools for autonomous operation
    cmd.extend([
        "--print",
        "-p", prompt,
        "--allowedTools", "all",  # Allow full autonomy
    ])

    _log.info(f"Running Claude Code in {working_dir}")
    _log.info(f"Prompt: {prompt[:200]}...")

    # Run the CLI
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=working_dir,
    )

    # Collect output with timeout
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=CLI_TIMEOUT
        )
        output = stdout.decode("utf-8", errors="replace")

    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"Claude Code timed out after {CLI_TIMEOUT} seconds")

    if process.returncode != 0:
        raise RuntimeError(f"Claude Code exited with code {process.returncode}: {output[:500]}")

    return output


async def run_codex(task_data: dict) -> str:
    """
    Run Codex CLI in agentic mode.

    Similar to Claude Code but for the Codex CLI.
    """
    task = task_data["task"]
    context = task_data.get("context", {})
    working_dir = task_data.get("working_dir", str(PROJECT_ROOT))

    prompt = _build_prompt(task, context)

    # Get Codex CLI command
    cmd = _get_codex_cmd()

    # Add the prompt - Codex typically takes prompt as argument
    cmd.append(prompt)

    _log.info(f"Running Codex in {working_dir}")
    _log.info(f"Prompt: {prompt[:200]}...")

    # Run the CLI
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=working_dir,
    )

    # Collect output with timeout
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=CLI_TIMEOUT
        )
        output = stdout.decode("utf-8", errors="replace")

    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"Codex timed out after {CLI_TIMEOUT} seconds")

    if process.returncode != 0:
        raise RuntimeError(f"Codex exited with code {process.returncode}: {output[:500]}")

    return output


async def main(agent_id: str):
    """Main entry point."""
    _log.info(f"[{agent_id}] CLI Worker starting...")

    try:
        # Load task
        task_data = load_task(agent_id)
        agent_type = task_data.get("agent_type", "claude_code")
        _log.info(f"[{agent_id}] Agent type: {agent_type}")
        _log.info(f"[{agent_id}] Task: {task_data['task'][:100]}...")

        # Execute based on agent type
        if agent_type == "claude_code":
            result = await run_claude_code(task_data)
        elif agent_type == "codex":
            result = await run_codex(task_data)
        else:
            raise ValueError(f"Unknown agent type for CLI worker: {agent_type}")

        _log.info(f"[{agent_id}] Completed, result length: {len(result)}")

        # Save result
        save_result(agent_id, "completed", result=result)

    except Exception as e:
        _log.error(f"[{agent_id}] Failed: {e}")
        save_result(agent_id, "failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli_worker.py <agent_id>")
        sys.exit(1)

    agent_id = sys.argv[1]
    asyncio.run(main(agent_id))
