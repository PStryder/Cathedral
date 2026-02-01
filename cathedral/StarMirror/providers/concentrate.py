# cathedral/StarMirror/providers/concentrate.py
"""
Concentrate.ai provider for StarMirror.

Uses the Concentrate.ai /v1/responses/ API format.
"""

import os
import json
import httpx
from dotenv import load_dotenv
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any

env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

API_URL = "https://api.concentrate.ai/v1/responses/"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-5.1")


def _get_api_key() -> str:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("CONCENTRATE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY or CONCENTRATE_API_KEY is not set."
        )
    return api_key


def _extract_text_from_response(data: Dict) -> str:
    """
    Extract text from Concentrate.ai response format.

    Response format:
    {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "actual text"}
                ]
            }
        ]
    }
    """
    # Try the nested output format first
    if "output" in data and isinstance(data["output"], list):
        texts = []
        for item in data["output"]:
            if isinstance(item, dict):
                # Check for content array
                if "content" in item and isinstance(item["content"], list):
                    for content_item in item["content"]:
                        if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                            if text := content_item.get("text"):
                                texts.append(text)
                # Check for direct text
                elif "text" in item:
                    texts.append(item["text"])
        if texts:
            return "\n".join(texts)

    # Fallback to simpler formats
    if "text" in data and isinstance(data["text"], str):
        return data["text"]
    if "content" in data and isinstance(data["content"], str):
        return data["content"]
    if "response" in data and isinstance(data["response"], str):
        return data["response"]

    return ""


def _messages_to_input(messages: List[Dict]) -> str:
    """
    Convert OpenAI-style messages array to a single input string.

    Concentrate.ai expects a single 'input' field, so we format
    the conversation history into a readable prompt.
    """
    parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle multi-modal content (extract text only)
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = " ".join(text_parts)

        if role == "system":
            parts.append(f"[System]: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    return "\n\n".join(parts)


async def stream(
    messages: List[Dict],
    *,
    temperature: float = 0.7,
    model: str = None,
    max_tokens: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from Concentrate.ai API.

    Note: If Concentrate doesn't support streaming, this will
    fetch the full response and yield it as one chunk.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    model = model or DEFAULT_MODEL
    api_key = _get_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    # Convert messages to single input
    input_text = _messages_to_input(messages)

    payload = {
        "model": model,
        "input": input_text,
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    # Add temperature if supported
    if temperature != 0.7:
        payload["temperature"] = temperature

    # Debug: log request
    print(f"[Concentrate] Sending to {API_URL}")
    print(f"[Concentrate] Model: {model}")
    print(f"[Concentrate] Input length: {len(input_text)} chars")

    # Try streaming first
    stream_url = API_URL.rstrip('/') + '?stream=true'

    async with httpx.AsyncClient(timeout=120.0) as client:
        # First try streaming endpoint
        try:
            async with client.stream("POST", stream_url, headers=headers, json=payload) as response:
                if response.status_code == 200:
                    raw_content = ""
                    async for line in response.aiter_lines():
                        print(f"[Concentrate] Stream line: {line[:200] if line else '(empty)'}")
                        raw_content += line + "\n"

                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                # Try various response formats
                                if "choices" in data:
                                    if delta := data["choices"][0].get("delta", {}).get("content"):
                                        yield delta
                                elif "output" in data:
                                    yield data["output"]
                                elif "text" in data:
                                    yield data["text"]
                                elif "content" in data:
                                    yield data["content"]
                            except json.JSONDecodeError:
                                # Might be raw text
                                if line.strip():
                                    yield line

                    # If we got content but yielded nothing, try parsing as complete response
                    if raw_content.strip():
                        print(f"[Concentrate] Raw response (first 500): {raw_content[:500]}")
                        try:
                            data = json.loads(raw_content)
                            text = _extract_text_from_response(data)
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            pass
                    return
        except Exception as e:
            print(f"[Concentrate] Streaming error: {e}")
            pass  # Fall back to non-streaming

        # Fall back to non-streaming request
        response = await client.post(API_URL, headers=headers, json=payload)

        # Log error details for debugging
        if response.status_code >= 400:
            print(f"[Concentrate] Error {response.status_code}: {response.text}")

        response.raise_for_status()
        data = response.json()

        # Extract response using the helper function
        content = _extract_text_from_response(data)

        # Fallback to OpenAI format
        if not content and "choices" in data:
            content = data["choices"][0].get("message", {}).get("content", "")

        # Last resort - return raw for debugging
        if not content:
            content = f"[Debug] Raw response: {str(data)[:500]}"

        yield content


async def transmit_async(
    messages: List[Dict],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Async non-streaming transmit. Returns complete response.
    """
    result = ""
    async for chunk in stream(messages, temperature=temperature, model=model, max_tokens=max_tokens):
        result += chunk
    return result


def transmit(
    messages: List[Dict],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    verbose: bool = True
) -> str:
    """
    Sync transmit. Returns complete response.
    """
    import asyncio
    return asyncio.run(transmit_async(messages, model, temperature, max_tokens))


# Alias for compatibility
transmit_stream = stream


__all__ = [
    "stream",
    "transmit",
    "transmit_stream",
    "transmit_async",
]
