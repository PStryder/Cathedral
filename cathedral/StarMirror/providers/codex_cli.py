"""
Codex CLI provider for StarMirror.

Uses a locally installed Codex CLI with streaming via stdout.
"""

import asyncio
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from cathedral import Config
from cathedral.StarMirror.providers import serialize_messages


DEFAULT_CMD = "codex"
CMD_ENV = "CODEX_CLI_CMD"
REQUIRED_FLAGS = ["--stream"]


def _command_exists(executable: str) -> bool:
    if not executable:
        return False
    if os.path.isabs(executable) or "/" in executable or "\\" in executable:
        return Path(executable).exists()
    return shutil.which(executable) is not None


def _resolve_cmd() -> List[str]:
    cmd_value = os.getenv(CMD_ENV) or Config.get(CMD_ENV, DEFAULT_CMD) or DEFAULT_CMD
    parts = shlex.split(cmd_value)
    if not parts:
        raise RuntimeError(f"{CMD_ENV} is empty; set it to your Codex CLI command.")

    if not _command_exists(parts[0]):
        raise RuntimeError(
            "Codex CLI not found. Install it or set CODEX_CLI_CMD to the correct command."
        )

    for flag in REQUIRED_FLAGS:
        if flag not in parts:
            parts.append(flag)
    return parts


def ensure_available() -> None:
    """Raise if the Codex CLI is not installed or configured."""
    _resolve_cmd()


async def stream(
    messages: List[Dict],
    *,
    temperature: float = 0.7,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from Codex CLI by reading stdout line-by-line.
    """
    _ = temperature, model, max_tokens  # CLI settings are controlled via CODEX_CLI_CMD
    cmd = _resolve_cmd()
    prompt = serialize_messages(messages)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert process.stdin is not None
    process.stdin.write(prompt.encode("utf-8"))
    await process.stdin.drain()
    process.stdin.close()
    if hasattr(process.stdin, "wait_closed"):
        await process.stdin.wait_closed()

    assert process.stdout is not None
    async for line in process.stdout:
        chunk = line.decode("utf-8", errors="replace")
        yield chunk

    return_code = await process.wait()
    if return_code != 0:
        raise RuntimeError(f"Codex CLI exited with {return_code}.")


def transmit(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    verbose: bool = True,
) -> str:
    """
    Sync completion using Codex CLI (non-streaming).
    """
    _ = temperature, model, max_tokens, verbose
    cmd = _resolve_cmd()
    prompt = serialize_messages(messages)

    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Codex CLI exited with {result.returncode}: {result.stderr.strip()}"
        )

    return result.stdout


async def transmit_async(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Async completion that collects the streamed output.
    """
    chunks: List[str] = []
    async for chunk in stream(
        messages,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
    ):
        chunks.append(chunk)
    return "".join(chunks)


__all__ = ["stream", "transmit", "transmit_async", "ensure_available"]
