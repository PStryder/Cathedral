# cathedral/StarMirror/providers/openclaw_agent.py
"""
OpenClaw Agent provider for StarMirror.

This provider calls OpenClaw Gateway's agent hook endpoint (/hooks/agent) instead of
the raw chat completions endpoint. OpenClaw handles tool execution internally.

Hybrid flow:
1. Cathedral performs context injection (MemoryGate + ScriptureGate)
2. This provider sends the enriched context to OpenClaw agent
3. OpenClaw executes its tool universe and returns the final response
4. Cathedral stores the response (but does NOT run ToolGate orchestration)

Configuration:
    OPENCLAW_AGENT_URL: Agent hook endpoint (default: http://127.0.0.1:18789/hooks/agent)
    OPENCLAW_TOKEN: Optional auth token for remote gateways
    OPENCLAW_SESSION_KEY: Session identifier for OpenClaw (optional)
    OPENCLAW_AGENT_LABEL: Agent label/personality in OpenClaw (optional)
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

# OpenClaw Agent endpoint - hooks/agent instead of /v1/chat/completions
API_URL = os.getenv("OPENCLAW_AGENT_URL", "http://127.0.0.1:18789/hooks/agent")

# Session/agent identification
DEFAULT_SESSION_KEY = os.getenv("OPENCLAW_SESSION_KEY", "cathedral")
DEFAULT_AGENT_LABEL = os.getenv("OPENCLAW_AGENT_LABEL", None)

# Timeout for agent operations (longer than chat - agents may execute tools)
AGENT_TIMEOUT = float(os.getenv("OPENCLAW_AGENT_TIMEOUT", "300"))


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
    """Format message content for logging."""
    if isinstance(content, str):
        return content[:max_len] + "..." if len(content) > max_len else content
    if isinstance(content, list):
        parts = []
        for p in content:
            if p.get("type") == "text":
                text = p.get("text", "")[:50]
                parts.append(f"text:{text}...")
            elif p.get("type") == "image_url":
                parts.append("[image]")
            elif p.get("type") == "input_audio":
                parts.append("[audio]")
        return " + ".join(parts)
    return str(content)[:max_len]


def _extract_user_message_and_context(messages: List[Dict]) -> tuple[str, str]:
    """
    Extract the current user message and assembled context from Cathedral's message array.

    Cathedral's context assembly order:
    - Position 0: Tool Protocol (skipped - we're using OpenClaw's tools)
    - Position 1: Personality/Guidance (system prompt)
    - Position 2: Scripture/RAG Context (system - labeled)
    - Position 3: Memory Context (system - labeled)
    - Position 4+: Prior history
    - Position LAST: Current user message

    Returns:
        (user_message, context_block) - The current query and assembled context
    """
    if not messages:
        return "", ""

    # Current user message is always last
    user_message = ""
    if messages[-1].get("role") == "user":
        user_message = messages[-1].get("content", "")
        if isinstance(user_message, list):
            # Handle multimodal - extract text parts
            text_parts = [p.get("text", "") for p in user_message if p.get("type") == "text"]
            user_message = " ".join(text_parts)

    # Build context block from system messages and history
    context_parts = []

    for msg in messages[:-1]:  # Exclude last (current user message)
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            # Handle multimodal
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = " ".join(text_parts)

        if role == "system":
            # Include system prompts and context blocks
            context_parts.append(content)
        elif role in ("user", "assistant"):
            # Include conversation history
            context_parts.append(f"[{role.upper()}]: {content}")

    context_block = "\n\n".join(context_parts)
    return user_message, context_block


def _build_agent_payload(
    messages: List[Dict],
    session_key: Optional[str] = None,
    agent_label: Optional[str] = None,
    stream: bool = True,
) -> Dict[str, Any]:
    """
    Build the payload for OpenClaw's /hooks/agent endpoint.

    This extracts Cathedral's injected context and formats it for OpenClaw.

    Expected OpenClaw /hooks/agent schema (VERIFY WITH ACTUAL API):
    {
        "sessionKey": "cathedral",          # Session identifier
        "label": "assistant",               # Agent label/personality (optional)
        "message": "user's query",          # Current user message
        "context": "injected context...",   # Cathedral's assembled context
        "stream": true                      # Enable SSE streaming
    }

    Adjust this function once the actual API schema is confirmed.
    """
    user_message, context_block = _extract_user_message_and_context(messages)

    payload: Dict[str, Any] = {
        "message": user_message,
        "stream": stream,
    }

    # Add session key if provided
    if session_key or DEFAULT_SESSION_KEY:
        payload["sessionKey"] = session_key or DEFAULT_SESSION_KEY

    # Add agent label if provided
    if agent_label or DEFAULT_AGENT_LABEL:
        payload["label"] = agent_label or DEFAULT_AGENT_LABEL

    # Include Cathedral's assembled context if present
    if context_block:
        payload["context"] = context_block

    return payload


async def stream(
    messages: List[Dict],
    *,
    temperature: float = 0.7,  # May be ignored by OpenClaw agent
    model: str = None,  # May be ignored - OpenClaw uses its configured model
    max_tokens: Optional[int] = None,
    session_key: Optional[str] = None,
    agent_label: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream response from OpenClaw Agent via /hooks/agent endpoint.

    Cathedral injects context, OpenClaw executes tools and returns response.
    This provider does NOT trigger Cathedral's ToolGate - OpenClaw handles tools.

    Args:
        messages: Cathedral's assembled message array (with injected context)
        temperature: Hint for temperature (may be ignored by agent)
        model: Hint for model (may be ignored - agent uses its config)
        max_tokens: Hint for max tokens (may be ignored)
        session_key: OpenClaw session identifier
        agent_label: OpenClaw agent label/personality

    Yields:
        Response text tokens from OpenClaw agent
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    headers = _get_headers()
    payload = _build_agent_payload(
        messages,
        session_key=session_key,
        agent_label=agent_label,
        stream=True,
    )

    # Add optional hints (OpenClaw may or may not use these)
    if temperature != 0.7:
        payload["temperature"] = temperature
    if model:
        payload["model"] = model
    if max_tokens:
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        async with client.stream("POST", API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()

            # Parse SSE stream
            # OpenClaw may use different SSE format - adjust as needed
            async for line in response.aiter_lines():
                if not line:
                    continue

                # Standard SSE format: data: {...}
                if line.startswith("data: "):
                    data_str = line[6:]

                    # Check for stream end
                    if data_str.strip() in ("[DONE]", ""):
                        break

                    try:
                        data = json.loads(data_str)

                        # Try multiple response formats (adjust based on actual API)
                        # Format 1: OpenAI-compatible
                        if choices := data.get("choices"):
                            if delta := choices[0].get("delta"):
                                if content := delta.get("content"):
                                    yield content
                        # Format 2: Direct content field
                        elif content := data.get("content"):
                            yield content
                        # Format 3: Direct text field
                        elif text := data.get("text"):
                            yield text
                        # Format 4: Message field
                        elif message := data.get("message"):
                            yield message
                        # Format 5: Output field
                        elif output := data.get("output"):
                            yield output

                    except json.JSONDecodeError:
                        # Non-JSON line - might be raw text
                        if data_str.strip():
                            yield data_str

                # Alternative: raw text streaming (no data: prefix)
                elif not line.startswith(":"):  # Skip SSE comments
                    # Could be raw text output
                    pass


def transmit(
    messages: List[Dict],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    verbose: bool = True,
    session_key: Optional[str] = None,
    agent_label: Optional[str] = None,
) -> str:
    """
    Send messages to OpenClaw Agent and return complete response (sync, non-streaming).
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    headers = _get_headers()
    payload = _build_agent_payload(
        messages,
        session_key=session_key,
        agent_label=agent_label,
        stream=False,
    )

    if temperature != 0.7:
        payload["temperature"] = temperature
    if model:
        payload["model"] = model
    if max_tokens:
        payload["max_tokens"] = max_tokens

    if verbose:
        print("[StarMirror:OpenClawAgent] Transmit to agent:")
        user_msg, ctx = _extract_user_message_and_context(messages)
        print(f"  - User: {_format_content_for_log(user_msg)}")
        print(f"  - Context: {len(ctx)} chars")

    response = None
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=AGENT_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        print("[StarMirror:OpenClawAgent] API call failed.")
        if response is not None:
            print("Status code:", getattr(response, "status_code", "no response"))
            print("Response text:", getattr(response, "text", "no text")[:500])
        raise RuntimeError(f"Failed to transmit to OpenClaw Agent: {e}")

    # Parse response (adjust based on actual API format)
    try:
        data = response.json()
        # Try multiple response formats
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "content" in data:
            return data["content"]
        elif "text" in data:
            return data["text"]
        elif "message" in data:
            return data["message"]
        elif "output" in data:
            return data["output"]
        else:
            # Return raw response if structure unknown
            return str(data)
    except (KeyError, IndexError, TypeError) as e:
        print("[StarMirror:OpenClawAgent] Malformed response:")
        print(response.json() if response else "no response")
        raise RuntimeError(f"Malformed response structure: {e}")


async def transmit_async(
    messages: List[Dict],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    session_key: Optional[str] = None,
    agent_label: Optional[str] = None,
) -> str:
    """
    Async non-streaming transmit. Returns complete response.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    headers = _get_headers()
    payload = _build_agent_payload(
        messages,
        session_key=session_key,
        agent_label=agent_label,
        stream=False,
    )

    if temperature != 0.7:
        payload["temperature"] = temperature
    if model:
        payload["model"] = model
    if max_tokens:
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        response = await client.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Parse response (multiple formats)
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "content" in data:
            return data["content"]
        elif "text" in data:
            return data["text"]
        elif "message" in data:
            return data["message"]
        elif "output" in data:
            return data["output"]
        else:
            return str(data)


# Flag to indicate this backend handles its own tools
# Cathedral's pipeline checks this to skip ToolGate orchestration
HANDLES_TOOLS_INTERNALLY = True


# Backwards-compatible alias
transmit_stream = stream


__all__ = [
    "stream",
    "transmit",
    "transmit_stream",
    "transmit_async",
    "HANDLES_TOOLS_INTERNALLY",
]
