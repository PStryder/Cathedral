"""
VoiceGate - Local TTS for Cathedral voice output.

Supports multiple TTS providers:
- Qwen3-TTS: Fast local TTS with sub-100ms latency
- PersonaPlex: NVIDIA's full-duplex S2S with persona control

Uses sentence-buffered streaming for smooth playback.

Architecture:
    Chat Response Stream
            │
            v
    Token Accumulator (buffer until sentence boundary)
            │
            v
    VoiceGate.synthesize_sentence()  ─── GPU check ─── skip if no GPU
            │
            v
    Audio Queue (gapless playback)
            │
            v
    WebSocket → Frontend Audio Player
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional, Dict, Any, List

from cathedral.shared.gate import GateLogger, build_health_status
from cathedral import Config

from .models import (
    VoiceConfig,
    TTSProvider,
    AudioChunk,
    VoiceStatus,
    PersonaPlexConfig,
    PersonaPlexVoice,
    VoicePreset,
    DEFAULT_VOICE_PRESETS,
)
from .sentence_buffer import SentenceBuffer
from .audio_queue import AudioQueue, AudioQueueManager, get_queue_manager

_log = GateLogger.get("VoiceGate")

# Module state
_initialized = False
_synthesizer = None
_personaplex_synthesizer = None  # Separate instance for PersonaPlex
_config: Optional[VoiceConfig] = None


def _get_config() -> VoiceConfig:
    """Get or create the voice configuration."""
    global _config
    if _config is None:
        # Parse provider, defaulting to qwen3
        provider_str = os.getenv("TTS_PROVIDER", "qwen3").lower()
        try:
            provider = TTSProvider(provider_str)
        except ValueError:
            provider = TTSProvider.QWEN3

        # Build PersonaPlex config from env
        personaplex_config = PersonaPlexConfig(
            server_url=os.getenv("PERSONAPLEX_URL", "wss://localhost:8998"),
            voice_prompt=PersonaPlexVoice(os.getenv("PERSONAPLEX_VOICE", "NATF2")),
            text_prompt=os.getenv("PERSONAPLEX_PERSONA", None),
            hf_token=os.getenv("HF_TOKEN", None),
            cpu_offload=os.getenv("PERSONAPLEX_CPU_OFFLOAD", "false").lower() == "true",
        )

        _config = VoiceConfig(
            enabled=os.getenv("TTS_ENABLED", "false").lower() == "true",
            provider=provider,
            model_variant=os.getenv("TTS_MODEL", "Qwen3-TTS-0.6B"),
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
            personaplex=personaplex_config,
        )
    return _config


def initialize() -> bool:
    """
    Initialize the VoiceGate.

    Checks for GPU availability and prepares the TTS backend.
    Does not load the model until first use (lazy loading).

    Returns:
        True if initialization successful
    """
    global _initialized, _synthesizer, _personaplex_synthesizer

    if _initialized:
        return True

    config = _get_config()

    # Check if TTS is enabled
    if not config.enabled:
        _log.info("VoiceGate disabled by configuration")
        _initialized = True
        return True

    # Check for GPU
    try:
        import torch
        if not torch.cuda.is_available():
            _log.warning("No GPU available, VoiceGate will be disabled")
            _initialized = True
            return True
    except ImportError:
        _log.warning("PyTorch not available, VoiceGate will be disabled")
        _initialized = True
        return True

    # Initialize synthesizer based on provider
    if config.provider == TTSProvider.QWEN3:
        from .qwen_backend import Qwen3Synthesizer
        _synthesizer = Qwen3Synthesizer(
            model_variant=config.model_variant,
            sample_rate=config.sample_rate,
        )
        _log.info(f"VoiceGate initialized with Qwen3 ({config.model_variant})")

    elif config.provider == TTSProvider.PERSONAPLEX:
        try:
            from .personaplex_backend import PersonaPlexSynthesizer
            _synthesizer = PersonaPlexSynthesizer(config.personaplex)
            _log.info(f"VoiceGate initialized with PersonaPlex ({config.personaplex.voice_prompt.value})")
        except Exception as e:
            _log.error(f"Failed to initialize PersonaPlex: {e}")
            # Fall back to Qwen3
            from .qwen_backend import Qwen3Synthesizer
            _synthesizer = Qwen3Synthesizer(
                model_variant=config.model_variant,
                sample_rate=config.sample_rate,
            )
            _log.info("Fell back to Qwen3-TTS")

    _initialized = True
    return True


def is_available() -> bool:
    """
    Check if TTS synthesis is available.

    Returns:
        True if GPU is present and synthesizer is ready
    """
    if not _initialized:
        initialize()

    if _synthesizer is None:
        return False

    return _synthesizer.is_available()


def is_enabled() -> bool:
    """
    Check if TTS is enabled by configuration.

    Returns:
        True if TTS is enabled
    """
    return _get_config().enabled


async def synthesize_sentence(text: str) -> AsyncGenerator[bytes, None]:
    """
    Synthesize a sentence to audio, yielding chunks.

    Args:
        text: The sentence text to synthesize

    Yields:
        Audio chunks as bytes (16-bit PCM at configured sample rate)
    """
    if not is_available():
        return

    if not text.strip():
        return

    _log.debug(f"Synthesizing: {text[:50]}...")

    async for chunk in _synthesizer.synthesize(text):
        yield chunk


async def synthesize_to_queue(
    text: str,
    queue: AudioQueue,
    is_final: bool = False,
) -> bool:
    """
    Synthesize a sentence and enqueue the audio.

    Args:
        text: Text to synthesize
        queue: Audio queue to add chunks to
        is_final: Whether this is the final sentence

    Returns:
        True if synthesis completed successfully
    """
    if not is_available():
        return False

    if not text.strip():
        return False

    try:
        async for chunk in _synthesizer.synthesize(text):
            await queue.enqueue(chunk, is_final=False)

        # Mark the last chunk as final if requested
        if is_final:
            # Send an empty final chunk as end marker
            await queue.enqueue(b"", is_final=True)

        return True

    except Exception as e:
        _log.error(f"Synthesis to queue failed: {e}")
        return False


def get_sentence_buffer() -> SentenceBuffer:
    """
    Get a new sentence buffer for token accumulation.

    Returns:
        Fresh SentenceBuffer instance
    """
    return SentenceBuffer()


async def get_audio_queue(thread_uid: str) -> AudioQueue:
    """
    Get an audio queue for a thread.

    Args:
        thread_uid: Thread identifier

    Returns:
        AudioQueue for streaming
    """
    config = _get_config()
    manager = get_queue_manager()
    return await manager.get_queue(thread_uid, sample_rate=config.sample_rate)


def get_status() -> VoiceStatus:
    """
    Get the current voice synthesis status.

    Returns:
        VoiceStatus with availability and configuration info
    """
    config = _get_config()

    status = VoiceStatus(
        available=is_available(),
        provider=config.provider if config.enabled else TTSProvider.DISABLED,
        gpu_available=False,
        model_loaded=False,
        model_name=None,
        error=None,
        personaplex_connected=False,
        current_voice=None,
    )

    # Check GPU
    try:
        import torch
        status.gpu_available = torch.cuda.is_available()
    except ImportError:
        status.error = "PyTorch not installed"

    # Check synthesizer
    if _synthesizer is not None:
        info = _synthesizer.get_info()
        status.model_loaded = info.get("loaded", False) or info.get("available", False)
        status.model_name = info.get("model_variant") or info.get("backend")

        if info.get("error"):
            status.error = info["error"]

        # PersonaPlex specific
        if config.provider == TTSProvider.PERSONAPLEX:
            status.personaplex_connected = info.get("connected", False)
            status.current_voice = info.get("voice")

    return status


def get_info() -> Dict[str, Any]:
    """
    Get detailed information about VoiceGate.

    Returns:
        Dict with gate information
    """
    config = _get_config()
    status = get_status()

    # config.provider may be enum or string (due to use_enum_values)
    provider_val = config.provider.value if hasattr(config.provider, 'value') else config.provider

    info = {
        "gate": "VoiceGate",
        "version": "1.1",
        "purpose": "Local TTS for voice output",
        "enabled": config.enabled,
        "provider": provider_val,
        "sample_rate": config.sample_rate,
        "gpu_available": status.gpu_available,
        "model_loaded": status.model_loaded,
        "available": status.available,
        "error": status.error,
        "providers": [p.value for p in TTSProvider if p != TTSProvider.DISABLED],
    }

    # Provider-specific info
    if provider_val == "qwen3":
        info["model"] = config.model_variant
    elif provider_val == "personaplex":
        voice_val = config.personaplex.voice_prompt.value if hasattr(config.personaplex.voice_prompt, 'value') else config.personaplex.voice_prompt
        info["personaplex"] = {
            "server_url": config.personaplex.server_url,
            "voice": voice_val,
            "connected": status.personaplex_connected,
        }

    return info


def list_voices() -> List[Dict[str, Any]]:
    """
    List available voices for the current provider.

    Returns:
        List of voice info dicts
    """
    config = _get_config()

    if config.provider == TTSProvider.PERSONAPLEX and _synthesizer is not None:
        return _synthesizer.list_voices()

    # Default voices for Qwen3
    return [
        {"id": "default", "name": "Default", "available": True},
    ]


def get_voice_presets() -> List[Dict[str, Any]]:
    """
    Get available voice presets.

    Returns:
        List of VoicePreset dicts
    """
    return DEFAULT_VOICE_PRESETS


async def set_voice(voice_id: str) -> bool:
    """
    Set the active voice.

    Args:
        voice_id: Voice identifier

    Returns:
        True if voice was set successfully
    """
    global _config

    config = _get_config()

    if config.provider == TTSProvider.PERSONAPLEX:
        try:
            voice = PersonaPlexVoice(voice_id)
            config.personaplex.voice_prompt = voice
            _log.info(f"Voice set to {voice_id}")
            return True
        except ValueError:
            _log.warning(f"Unknown PersonaPlex voice: {voice_id}")
            return False

    return False


# ==================== Health Checks ====================


def is_healthy() -> bool:
    """Check if the gate is operational."""
    return _initialized


def get_health_status() -> Dict[str, Any]:
    """Get detailed health information."""
    config = _get_config()
    status = get_status()

    checks = {
        "initialized": _initialized,
        "enabled": config.enabled,
        "gpu_available": status.gpu_available,
    }

    if config.enabled and status.gpu_available:
        checks["synthesizer_ready"] = _synthesizer is not None

    return build_health_status(
        gate_name="VoiceGate",
        initialized=_initialized,
        dependencies=["torch", "transformers"] if config.enabled else [],
        checks=checks,
        details=get_info(),
    )


def get_dependencies() -> list:
    """List external dependencies."""
    config = _get_config()
    if not config.enabled:
        return []

    deps = ["torch", "numpy", "soundfile"]

    if config.provider == TTSProvider.QWEN3:
        deps.append("transformers")
    elif config.provider == TTSProvider.PERSONAPLEX:
        deps.extend(["moshi", "websockets"])

    return deps


# ==================== Cleanup ====================


async def shutdown():
    """Shutdown the VoiceGate, freeing resources."""
    global _initialized, _synthesizer, _personaplex_synthesizer

    if _synthesizer is not None:
        if hasattr(_synthesizer, "shutdown"):
            await _synthesizer.shutdown()
        elif hasattr(_synthesizer, "unload"):
            _synthesizer.unload()
        _synthesizer = None

    if _personaplex_synthesizer is not None:
        await _personaplex_synthesizer.shutdown()
        _personaplex_synthesizer = None

    _initialized = False
    _log.info("VoiceGate shutdown complete")


def shutdown_sync():
    """Synchronous shutdown (for non-async contexts)."""
    global _initialized, _synthesizer

    if _synthesizer is not None:
        if hasattr(_synthesizer, "unload"):
            _synthesizer.unload()
        _synthesizer = None

    _initialized = False
    _log.info("VoiceGate shutdown complete")


__all__ = [
    # Lifecycle
    "initialize",
    "shutdown",
    "shutdown_sync",
    # Status
    "is_available",
    "is_enabled",
    "get_status",
    "get_info",
    # Synthesis
    "synthesize_sentence",
    "synthesize_to_queue",
    # Voice management
    "list_voices",
    "get_voice_presets",
    "set_voice",
    # Utilities
    "get_sentence_buffer",
    "get_audio_queue",
    # Health
    "is_healthy",
    "get_health_status",
    "get_dependencies",
    # Models
    "VoiceConfig",
    "TTSProvider",
    "PersonaPlexVoice",
    "PersonaPlexConfig",
    "AudioChunk",
    "VoiceStatus",
    "VoicePreset",
    "SentenceBuffer",
    "AudioQueue",
]
