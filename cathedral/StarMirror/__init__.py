# cathedral/StarMirror/__init__.py
"""
StarMirror - LLM communication interface with multi-modal support.

Text, images, and audio supported through OpenRouter API.
"""

from typing import AsyncGenerator, List, Optional
from cathedral.StarMirror.StarMirrorGate import (
    # Core functions
    transmit,
    transmit_stream,
    transmit_async,
    # Vision functions
    transmit_vision,
    transmit_vision_stream,
    describe_image,
    compare_images,
    # Audio functions
    transcribe_audio,
    transmit_with_audio,
    # Multi-modal building
    ContentBuilder,
    MediaItem,
    ImageDetail,
    build_multimodal_message,
    load_image,
    load_audio,
    supports_vision,
)


def reflect(prompt_history, system_prompt=None, temperature=0.7):
    """
    Send prompt to LLM and get response (sync, text-only).

    Args:
        prompt_history: List of {role, content} message dicts
        system_prompt: Optional system message
        temperature: Sampling temperature

    Returns:
        Model response text
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(prompt_history)
    return transmit(messages, temperature=temperature)


async def reflect_stream(
    prompt_history,
    system_prompt=None,
    temperature=0.7,
    model=None,
    max_tokens=None
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from LLM (async, supports multi-modal).

    Args:
        prompt_history: List of {role, content} message dicts
                       (content can be string or multi-modal array)
        system_prompt: Optional system message
        temperature: Sampling temperature
        model: Model to use (None uses default)
        max_tokens: Max response tokens

    Yields:
        Response tokens
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(prompt_history)

    kwargs = {"temperature": temperature}
    if model:
        kwargs["model"] = model
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    async for token in transmit_stream(messages, **kwargs):
        yield token


async def reflect_async(prompt_history, system_prompt=None, temperature=0.7) -> str:
    """
    Send prompt to LLM and get response (async, supports multi-modal).

    Args:
        prompt_history: List of {role, content} message dicts
        system_prompt: Optional system message
        temperature: Sampling temperature

    Returns:
        Model response text
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(prompt_history)
    return await transmit_async(messages, temperature=temperature)


async def reflect_vision(
    prompt: str,
    images: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    detail: ImageDetail = ImageDetail.AUTO
) -> str:
    """
    Query LLM about images (async).

    Args:
        prompt: Text question about the images
        images: List of image paths, URLs, or base64 strings
        system_prompt: Optional system message
        temperature: Sampling temperature
        detail: Image detail level (low/high/auto)

    Returns:
        Model response
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_msg = build_multimodal_message(
        text=prompt,
        images=images,
        image_detail=detail
    )
    messages.append(user_msg)

    return await transmit_async(messages, temperature=temperature)


async def reflect_vision_stream(
    prompt: str,
    images: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    detail: ImageDetail = ImageDetail.AUTO
) -> AsyncGenerator[str, None]:
    """
    Stream vision query response (async).

    Args:
        prompt: Text question about the images
        images: List of image paths, URLs, or base64 strings
        system_prompt: Optional system message
        temperature: Sampling temperature
        detail: Image detail level

    Yields:
        Response tokens
    """
    async for token in transmit_vision_stream(
        prompt=prompt,
        images=images,
        system_prompt=system_prompt,
        temperature=temperature,
        image_detail=detail
    ):
        yield token


async def reflect_audio(
    prompt: str,
    audio_path: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7
) -> str:
    """
    Query LLM with audio input (transcribes then queries).

    Args:
        prompt: Text prompt about the audio
        audio_path: Path to audio file
        system_prompt: Optional system message
        temperature: Sampling temperature

    Returns:
        Model response
    """
    # Transcribe audio first
    transcription = await transcribe_audio(audio_path)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    combined = f"{prompt}\n\n[Audio transcription]:\n{transcription}"
    messages.append({"role": "user", "content": combined})

    return await transmit_async(messages, temperature=temperature)


# Export everything
__all__ = [
    # Text reflection
    "reflect",
    "reflect_stream",
    "reflect_async",
    # Vision reflection
    "reflect_vision",
    "reflect_vision_stream",
    "describe_image",
    "compare_images",
    # Audio reflection
    "reflect_audio",
    "transcribe_audio",
    # Multi-modal building
    "ContentBuilder",
    "MediaItem",
    "ImageDetail",
    "build_multimodal_message",
    "load_image",
    "load_audio",
    "supports_vision",
    # Low-level (re-exports)
    "transmit",
    "transmit_stream",
    "transmit_async",
    "transmit_vision",
    "transmit_vision_stream",
    "transmit_with_audio",
]
