# cathedral/StarMirror/providers/openrouter.py
"""
OpenRouter provider for StarMirror.

Preserves existing OpenRouter behavior, including multimodal support and
Whisper transcription.
"""

import os
import json
import requests
import httpx
from dotenv import load_dotenv
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any

from cathedral.StarMirror.MediaGate import (
    ContentBuilder,
    MediaItem,
    ImageDetail,
    build_multimodal_message,
    supports_vision,
    load_image,
    load_audio,
)

env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

# Configurable API endpoint - defaults to OpenRouter
API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-2024-11-20")
DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4o-2024-11-20")

def _get_api_key() -> str:
    # Try LLM_API_KEY first, fall back to OPENROUTER_API_KEY
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY or OPENROUTER_API_KEY is not set. Ensure your .env file exists and contains the key."
        )
    return api_key


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
    Stream tokens from OpenRouter API using SSE.
    Supports both text-only and multi-modal messages.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

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
    Send messages to OpenRouter API and return response.
    Supports both text-only and multi-modal messages.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")

    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    if verbose:
        print("[StarMirror] Transmit payload:")
        for msg in messages:
            content_str = _format_content_for_log(msg.get("content", ""))
            print(f"  - {msg.get('role', 'unknown')}: {content_str}")

    response = None
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        print("[StarMirror] API call failed.")
        if response is not None:
            print("Status code:", getattr(response, "status_code", "no response"))
            print("Response text:", getattr(response, "text", "no text")[:500])
        raise RuntimeError(f"Failed to transmit to LLM: {e}")

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print("[StarMirror] Malformed response:")
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

    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

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


# ==================== Multi-Modal Convenience Functions ====================


async def transmit_vision_stream(
    prompt: str,
    images: List[str],
    model: str = DEFAULT_VISION_MODEL,
    temperature: float = 0.7,
    image_detail: ImageDetail = ImageDetail.AUTO,
    system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Stream a vision query with images.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build multi-modal user message
    user_msg = build_multimodal_message(
        text=prompt,
        images=images,
        image_detail=image_detail
    )
    messages.append(user_msg)

    async for token in stream(messages, model=model, temperature=temperature):
        yield token


def transmit_vision(
    prompt: str,
    images: List[str],
    model: str = DEFAULT_VISION_MODEL,
    temperature: float = 0.7,
    image_detail: ImageDetail = ImageDetail.AUTO,
    system_prompt: Optional[str] = None
) -> str:
    """
    Send a vision query with images (sync).
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_msg = build_multimodal_message(
        text=prompt,
        images=images,
        image_detail=image_detail
    )
    messages.append(user_msg)

    return transmit(messages, model=model, temperature=temperature)


async def describe_image(
    image: str,
    prompt: str = "Describe this image in detail.",
    model: str = DEFAULT_VISION_MODEL
) -> str:
    """
    Get a description of an image.
    """
    return await transmit_async(
        messages=[build_multimodal_message(text=prompt, images=[image])],
        model=model
    )


async def compare_images(
    images: List[str],
    prompt: str = "Compare these images and describe the differences.",
    model: str = DEFAULT_VISION_MODEL
) -> str:
    """
    Compare multiple images.
    """
    return await transmit_async(
        messages=[build_multimodal_message(text=prompt, images=images)],
        model=model
    )


# ==================== Audio Support ====================


WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"


async def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None
) -> str:
    """
    Transcribe audio using OpenAI Whisper API.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY required for audio transcription")

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    headers = {
        "Authorization": f"Bearer {openai_key}",
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "audio/mpeg")}
            data = {"model": "whisper-1"}

            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt

            response = await client.post(
                WHISPER_API_URL,
                headers=headers,
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json().get("text", "")


async def transmit_with_audio(
    prompt: str,
    audio_path: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    transcribe: bool = True
) -> str:
    """
    Process audio input (transcribe and include in prompt).
    """
    if transcribe:
        transcription = await transcribe_audio(audio_path)
        combined_prompt = f"{prompt}\n\n[Audio transcription]:\n{transcription}"
        messages = [{"role": "user", "content": combined_prompt}]
    else:
        # Some models support direct audio input
        audio_item = load_audio(audio_path)
        builder = ContentBuilder().text(prompt).audio_item(audio_item)
        messages = [builder.build_message()]

    return await transmit_async(messages, model=model, temperature=temperature)


# Backwards-compatible alias
transmit_stream = stream


__all__ = [
    "stream",
    "transmit",
    "transmit_stream",
    "transmit_async",
    "transmit_vision",
    "transmit_vision_stream",
    "describe_image",
    "compare_images",
    "transcribe_audio",
    "transmit_with_audio",
    "ContentBuilder",
    "MediaItem",
    "ImageDetail",
    "build_multimodal_message",
    "load_image",
    "load_audio",
    "supports_vision",
]
