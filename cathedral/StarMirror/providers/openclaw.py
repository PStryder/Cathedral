# cathedral/StarMirror/providers/openclaw.py
"""
OpenClaw provider for StarMirror.

Connects to OpenClaw Gateway's OpenAI-compatible endpoint for LLM inference.
Supports SSE streaming like OpenRouter but targets local/remote OpenClaw instances.
"""

import os
import json
import requests
import httpx
from dotenv import load_dotenv
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any


env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

# OpenClaw Gateway defaults - local by default, can point to remote
API_URL = os.getenv("OPENCLAW_API_URL", "http://127.0.0.1:18789/v1/chat/completions")
DEFAULT_MODEL = os.getenv("OPENCLAW_MODEL", "default")  # OpenClaw routes to configured model


def _get_api_key() -> Optional[str]:
    """Get optional API token for remote OpenClaw instances."""
    return os.getenv("OPENCLAW_TOKEN")


def _get_headers() -> Dict[str, str]:
    """Build request headers, including auth if token is set."""
    headers = {"Content-Type": "application/json"}
    token = _get_api_key()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _format_content_for_log(content: Any, max_len: int = 100) -> str:
    """Format message content for logging (handles multi-modal)."""
    if isinstance(content, str):
        return content[:max_len] + "..." if len(content) > max_len else content
    if isinstance(content, list):
        parts = []
        for p in content:
            if p.get("type") == "text":
                text = p.get("text", "")[:50]
                parts.append(f"text:{text}...")
            elif p.get("type") == "image_url":
                url = p.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    parts.append("[image:base64]")
                else:
                    parts.append(f"[image:{url[:30]}...]")
            elif p.get("type") == "input_audio":
                parts.append("[audio]")
        return " + ".join(parts)
    return str(content)[:max_len]


async def stream(
    messages: List[Dict],
    *,
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL,
    max_tokens: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from OpenClaw Gateway using SSE.

    OpenClaw Gateway exposes an OpenAI-compatible /v1/chat/completions endpoint
    with SSE streaming support.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    headers = _get_headers()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if choices := data.get("choices"):
                            if delta := choices[0].get("delta"):
                                if content := delta.get("content"):
                                    yield content
                    except json.JSONDecodeError:
                        continue


def transmit(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    verbose: bool = True
) -> str:
    """
    Send messages to OpenClaw Gateway and return response (sync, non-streaming).
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    headers = _get_headers()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    if verbose:
        print("[StarMirror:OpenClaw] Transmit payload:")
        for msg in messages:
            content_str = _format_content_for_log(msg.get("content", ""))
            print(f"  - {msg.get('role', 'unknown')}: {content_str}")

    response = None
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        print("[StarMirror:OpenClaw] API call failed.")
        if response is not None:
            print("Status code:", getattr(response, "status_code", "no response"))
            print("Response text:", getattr(response, "text", "no text")[:500])
        raise RuntimeError(f"Failed to transmit to OpenClaw: {e}")

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print("[StarMirror:OpenClaw] Malformed response:")
        print(response.json())
        raise RuntimeError(f"Malformed response structure: {e}")


async def transmit_async(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Async non-streaming transmit. Returns complete response.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    headers = _get_headers()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# Backwards-compatible alias
transmit_stream = stream


__all__ = [
    "stream",
    "transmit",
    "transmit_stream",
    "transmit_async",
]
