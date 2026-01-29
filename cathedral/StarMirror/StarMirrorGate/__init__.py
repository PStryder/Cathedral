# cathedral/StarMirror/StarMirrorGate/__init__.py
"""
StarMirrorGate - LLM API communication layer with multi-modal support.

Delegates to the StarMirror router for provider selection.
"""

from typing import AsyncGenerator, Dict, List, Optional

from .. import router as _router
from cathedral.StarMirror.MediaGate import (
    ContentBuilder,
    MediaItem,
    ImageDetail,
    build_multimodal_message,
    load_image,
    load_audio,
    supports_vision,
)

DEFAULT_MODEL = "openai/gpt-4o-2024-11-20"
DEFAULT_VISION_MODEL = "openai/gpt-4o-2024-11-20"


def _validate_messages(messages: List[Dict]) -> None:
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Cannot transmit: message list is empty or invalid.")


async def transmit_stream(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from the configured LLM backend.
    Supports both text-only and multi-modal messages.
    """
    _validate_messages(messages)

    async for token in _router.stream(
        messages,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
    ):
        yield token


def transmit(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    verbose: bool = True
) -> str:
    """
    Send messages to the configured LLM backend and return response.
    Supports both text-only and multi-modal messages.
    """
    _validate_messages(messages)
    return _router.transmit(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
    )


async def transmit_async(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Async non-streaming transmit. Returns complete response.
    """
    _validate_messages(messages)
    return await _router.transmit_async(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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
    messages: List[Dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build multi-modal user message
    user_msg = build_multimodal_message(
        text=prompt,
        images=images,
        image_detail=image_detail
    )
    messages.append(user_msg)

    async for token in transmit_stream(messages, model=model, temperature=temperature):
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
    messages: List[Dict] = []

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


async def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None
) -> str:
    """
    Transcribe audio using OpenAI Whisper API (via provider support).
    """
    return await _router.transcribe_audio(
        audio_path=audio_path,
        language=language,
        prompt=prompt,
    )


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
    return await _router.transmit_with_audio(
        prompt=prompt,
        audio_path=audio_path,
        model=model,
        temperature=temperature,
        transcribe=transcribe,
    )


# ==================== Exports ====================

__all__ = [
    # Core transmit functions
    "transmit",
    "transmit_stream",
    "transmit_async",
    # Vision functions
    "transmit_vision",
    "transmit_vision_stream",
    "describe_image",
    "compare_images",
    # Audio functions
    "transcribe_audio",
    "transmit_with_audio",
    # Re-exports from MediaGate
    "ContentBuilder",
    "MediaItem",
    "ImageDetail",
    "build_multimodal_message",
    "load_image",
    "load_audio",
    "supports_vision",
]
