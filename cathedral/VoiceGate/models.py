"""
VoiceGate data models.

Provides Pydantic models for TTS configuration and audio data.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class TTSProvider(str, Enum):
    """Supported TTS providers."""
    QWEN3 = "qwen3"
    PERSONAPLEX = "personaplex"
    DISABLED = "disabled"


class PersonaPlexVoice(str, Enum):
    """Pre-configured PersonaPlex voices."""
    # Natural voices - Female
    NATF1 = "NATF1"
    NATF2 = "NATF2"
    NATF3 = "NATF3"
    NATF4 = "NATF4"
    # Natural voices - Male
    NATM1 = "NATM1"
    NATM2 = "NATM2"
    NATM3 = "NATM3"
    NATM4 = "NATM4"
    # Variety voices - Female
    VARF1 = "VARF1"
    VARF2 = "VARF2"
    VARF3 = "VARF3"
    VARF4 = "VARF4"
    VARF5 = "VARF5"
    # Variety voices - Male
    VARM1 = "VARM1"
    VARM2 = "VARM2"
    VARM3 = "VARM3"
    VARM4 = "VARM4"
    VARM5 = "VARM5"


class PersonaPlexConfig(BaseModel):
    """PersonaPlex-specific configuration."""
    server_url: str = "wss://localhost:8998"  # PersonaPlex server WebSocket URL (wss for SSL)
    voice_prompt: PersonaPlexVoice = PersonaPlexVoice.NATF2  # Default voice
    text_prompt: Optional[str] = None  # Persona/role definition
    ssl_enabled: bool = True  # PersonaPlex Docker uses SSL by default
    hf_token: Optional[str] = None  # HuggingFace token for model access

    # Advanced settings
    use_cuda_graphs: bool = True
    cpu_offload: bool = False


class VoiceConfig(BaseModel):
    """Voice synthesis configuration."""
    enabled: bool = False
    provider: TTSProvider = TTSProvider.QWEN3
    model_variant: str = "Qwen3-TTS-0.6B"  # or 1.7B for higher quality
    sample_rate: int = 24000
    voice_id: Optional[str] = None  # For voice cloning (future)

    # Streaming settings
    chunk_size: int = 4096  # Audio chunk size in bytes
    buffer_sentences: bool = True  # Buffer until sentence boundary

    # PersonaPlex settings
    personaplex: PersonaPlexConfig = Field(default_factory=PersonaPlexConfig)

    class Config:
        use_enum_values = True


class SynthesisRequest(BaseModel):
    """Request to synthesize speech from text."""
    text: str
    voice_id: Optional[str] = None
    sample_rate: int = 24000
    stream: bool = True  # Stream audio chunks


class AudioChunk(BaseModel):
    """A chunk of audio data for streaming."""
    sequence: int
    data: bytes  # PCM audio bytes
    is_final: bool = False
    duration_ms: int = Field(default=0, description="Estimated duration in milliseconds")
    sample_rate: int = 24000

    class Config:
        # Allow bytes in JSON serialization
        arbitrary_types_allowed = True

    def to_wire(self) -> dict:
        """Convert to wire format (base64 encode data)."""
        import base64
        return {
            "sequence": self.sequence,
            "data": base64.b64encode(self.data).decode("ascii"),
            "is_final": self.is_final,
            "duration_ms": self.duration_ms,
            "sample_rate": self.sample_rate,
        }


class VoiceStatus(BaseModel):
    """Status of the voice synthesis system."""
    available: bool = False
    provider: TTSProvider = TTSProvider.DISABLED
    gpu_available: bool = False
    model_loaded: bool = False
    model_name: Optional[str] = None
    error: Optional[str] = None
    # PersonaPlex-specific
    personaplex_connected: bool = False
    current_voice: Optional[str] = None


class VoicePreset(BaseModel):
    """A voice preset combining provider settings."""
    id: str
    name: str
    description: str
    provider: TTSProvider
    voice_id: Optional[str] = None  # Provider-specific voice ID
    personaplex_voice: Optional[PersonaPlexVoice] = None
    text_prompt: Optional[str] = None  # Persona prompt for PersonaPlex


# Default voice presets
DEFAULT_VOICE_PRESETS: List[dict] = [
    {
        "id": "default",
        "name": "Default (Qwen3)",
        "description": "Default Qwen3-TTS voice",
        "provider": "qwen3",
    },
    {
        "id": "natf2",
        "name": "Natural Female 2",
        "description": "PersonaPlex natural female voice",
        "provider": "personaplex",
        "personaplex_voice": "NATF2",
    },
    {
        "id": "natm1",
        "name": "Natural Male 1",
        "description": "PersonaPlex natural male voice",
        "provider": "personaplex",
        "personaplex_voice": "NATM1",
    },
    {
        "id": "assistant",
        "name": "Helpful Assistant",
        "description": "PersonaPlex voice with assistant persona",
        "provider": "personaplex",
        "personaplex_voice": "NATF2",
        "text_prompt": "You are a helpful, knowledgeable assistant who speaks clearly and concisely.",
    },
]


__all__ = [
    "TTSProvider",
    "PersonaPlexVoice",
    "PersonaPlexConfig",
    "VoiceConfig",
    "SynthesisRequest",
    "AudioChunk",
    "VoiceStatus",
    "VoicePreset",
    "DEFAULT_VOICE_PRESETS",
]
