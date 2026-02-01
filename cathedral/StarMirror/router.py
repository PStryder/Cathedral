"""
StarMirror provider router.

Selects the active LLM backend and delegates calls to that provider.
"""

import importlib
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from cathedral import Config


SUPPORTED_BACKENDS = ("openrouter", "concentrate", "claude_cli", "codex_cli")
_provider_cache: Dict[str, Any] = {}
_last_backend: Optional[str] = None


def get_backend() -> str:
    """Return the configured LLM backend name."""
    backend = Config.get("LLM_BACKEND", os.getenv("LLM_BACKEND", "openrouter"))
    if backend is None:
        backend = "openrouter"
    backend = str(backend).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise RuntimeError(
            f"Unsupported LLM_BACKEND '{backend}'. "
            f"Choose one of: {', '.join(SUPPORTED_BACKENDS)}."
        )
    return backend


def _load_provider(backend: str):
    module = importlib.import_module(f"cathedral.StarMirror.providers.{backend}")
    return module


def _get_provider():
    global _last_backend
    backend = get_backend()

    if backend != _last_backend or backend not in _provider_cache:
        _provider_cache[backend] = _load_provider(backend)
        _last_backend = backend

    provider = _provider_cache[backend]
    if hasattr(provider, "ensure_available"):
        provider.ensure_available()
    return provider


async def stream(
    messages: List[Dict],
    *,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    provider = _get_provider()
    kwargs: Dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if model is not None:
        kwargs["model"] = model
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    async for chunk in provider.stream(messages, **kwargs):
        yield chunk


def transmit(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    verbose: bool = True,
) -> str:
    provider = _get_provider()
    if hasattr(provider, "transmit"):
        kwargs: Dict[str, Any] = {"messages": messages, "verbose": verbose}
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return provider.transmit(**kwargs)

    import asyncio

    async def _collect() -> str:
        chunks: List[str] = []
        async for chunk in stream(
            messages,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    try:
        return asyncio.run(_collect())
    except RuntimeError:
        raise RuntimeError("transmit() cannot be called from an async context; use transmit_async instead.")


async def transmit_async(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    provider = _get_provider()
    if hasattr(provider, "transmit_async"):
        kwargs: Dict[str, Any] = {"messages": messages}
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return await provider.transmit_async(**kwargs)

    chunks: List[str] = []
    async for chunk in stream(
        messages,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
    ):
        chunks.append(chunk)
    return "".join(chunks)


async def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    provider = _get_provider()
    if hasattr(provider, "transcribe_audio"):
        return await provider.transcribe_audio(
            audio_path=audio_path,
            language=language,
            prompt=prompt,
        )
    raise RuntimeError("Audio transcription is not supported by the selected LLM backend.")


async def transmit_with_audio(
    prompt: str,
    audio_path: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    transcribe: bool = True,
) -> str:
    provider = _get_provider()
    if hasattr(provider, "transmit_with_audio"):
        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "audio_path": audio_path,
            "transcribe": transcribe,
        }
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        return await provider.transmit_with_audio(**kwargs)
    raise RuntimeError("Audio input is not supported by the selected LLM backend.")


__all__ = [
    "SUPPORTED_BACKENDS",
    "get_backend",
    "stream",
    "transmit",
    "transmit_async",
    "transcribe_audio",
    "transmit_with_audio",
]
