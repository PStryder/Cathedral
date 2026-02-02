"""
Opus Codec Wrapper for VoiceGate.

Provides encoding/decoding of Opus audio for PersonaPlex communication.
PersonaPlex uses Opus at 24kHz mono for bidirectional audio streaming.

This is a thin wrapper around opuslib that handles:
- Frame size management
- Error handling
- Fallback when opuslib is not available
"""

from __future__ import annotations

from typing import Optional, List
import struct

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("VoiceGate.Opus")

# Opus constants
SAMPLE_RATE = 24000  # 24kHz - PersonaPlex default
CHANNELS = 1         # Mono
FRAME_DURATION_MS = 20  # 20ms frames (Opus default)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples per frame

# Try to import opuslib
_opuslib_available = False
_opus_encoder = None
_opus_decoder = None

try:
    import opuslib
    import opuslib.api.encoder
    import opuslib.api.decoder
    _opuslib_available = True
    _log.info("opuslib available - Opus codec enabled")
except ImportError:
    _log.warning("opuslib not installed - Opus codec disabled (pip install opuslib)")


class OpusEncoder:
    """
    Opus audio encoder.

    Converts PCM audio to Opus-encoded bytes for transmission.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        application: str = "voip",
    ):
        """
        Initialize the Opus encoder.

        Args:
            sample_rate: Audio sample rate (default 24000)
            channels: Number of channels (default 1 - mono)
            application: Opus application type (voip, audio, lowdelay)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = int(sample_rate * FRAME_DURATION_MS / 1000)
        self._encoder = None

        if _opuslib_available:
            import opuslib
            app_map = {
                "voip": opuslib.APPLICATION_VOIP,
                "audio": opuslib.APPLICATION_AUDIO,
                "lowdelay": opuslib.APPLICATION_RESTRICTED_LOWDELAY,
            }
            app = app_map.get(application, opuslib.APPLICATION_VOIP)

            try:
                self._encoder = opuslib.Encoder(sample_rate, channels, app)
                _log.debug(f"Opus encoder initialized: {sample_rate}Hz, {channels}ch")
            except Exception as e:
                _log.error(f"Failed to create Opus encoder: {e}")

    def encode(self, pcm_data: bytes) -> Optional[bytes]:
        """
        Encode PCM audio to Opus.

        Args:
            pcm_data: Raw PCM audio (16-bit signed, little-endian)

        Returns:
            Opus-encoded bytes, or None on error
        """
        if self._encoder is None:
            return None

        try:
            # PCM data should be frame_size * channels * 2 bytes (16-bit samples)
            expected_size = self.frame_size * self.channels * 2

            if len(pcm_data) != expected_size:
                _log.warning(f"PCM size mismatch: got {len(pcm_data)}, expected {expected_size}")
                # Pad or truncate
                if len(pcm_data) < expected_size:
                    pcm_data = pcm_data + b'\x00' * (expected_size - len(pcm_data))
                else:
                    pcm_data = pcm_data[:expected_size]

            return self._encoder.encode(pcm_data, self.frame_size)

        except Exception as e:
            _log.error(f"Opus encode error: {e}")
            return None

    def encode_frames(self, pcm_data: bytes) -> List[bytes]:
        """
        Encode PCM audio into multiple Opus frames.

        Splits the PCM data into frame-sized chunks and encodes each.

        Args:
            pcm_data: Raw PCM audio (any length)

        Returns:
            List of Opus-encoded frames
        """
        if self._encoder is None:
            return []

        frames = []
        bytes_per_frame = self.frame_size * self.channels * 2

        for i in range(0, len(pcm_data), bytes_per_frame):
            chunk = pcm_data[i:i + bytes_per_frame]

            # Pad last frame if needed
            if len(chunk) < bytes_per_frame:
                chunk = chunk + b'\x00' * (bytes_per_frame - len(chunk))

            encoded = self.encode(chunk)
            if encoded:
                frames.append(encoded)

        return frames

    @property
    def is_available(self) -> bool:
        """Check if encoder is available."""
        return self._encoder is not None


class OpusDecoder:
    """
    Opus audio decoder.

    Converts Opus-encoded bytes back to PCM audio.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
    ):
        """
        Initialize the Opus decoder.

        Args:
            sample_rate: Audio sample rate (default 24000)
            channels: Number of channels (default 1 - mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = int(sample_rate * FRAME_DURATION_MS / 1000)
        self._decoder = None

        if _opuslib_available:
            import opuslib
            try:
                self._decoder = opuslib.Decoder(sample_rate, channels)
                _log.debug(f"Opus decoder initialized: {sample_rate}Hz, {channels}ch")
            except Exception as e:
                _log.error(f"Failed to create Opus decoder: {e}")

    def decode(self, opus_data: bytes) -> Optional[bytes]:
        """
        Decode Opus audio to PCM.

        Args:
            opus_data: Opus-encoded bytes

        Returns:
            Raw PCM audio (16-bit signed, little-endian), or None on error
        """
        if self._decoder is None:
            return None

        try:
            return self._decoder.decode(opus_data, self.frame_size)
        except Exception as e:
            _log.error(f"Opus decode error: {e}")
            return None

    def decode_frames(self, opus_frames: List[bytes]) -> bytes:
        """
        Decode multiple Opus frames to PCM.

        Args:
            opus_frames: List of Opus-encoded frames

        Returns:
            Concatenated PCM audio
        """
        pcm_chunks = []
        for frame in opus_frames:
            pcm = self.decode(frame)
            if pcm:
                pcm_chunks.append(pcm)
        return b''.join(pcm_chunks)

    @property
    def is_available(self) -> bool:
        """Check if decoder is available."""
        return self._decoder is not None


class OpusCodec:
    """
    Combined Opus encoder/decoder for bidirectional audio.

    Convenience class that manages both encoding and decoding
    for a voice session.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
    ):
        """
        Initialize the codec.

        Args:
            sample_rate: Audio sample rate
            channels: Number of channels
        """
        self.encoder = OpusEncoder(sample_rate, channels)
        self.decoder = OpusDecoder(sample_rate, channels)
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = int(sample_rate * FRAME_DURATION_MS / 1000)

    def encode(self, pcm_data: bytes) -> Optional[bytes]:
        """Encode PCM to Opus."""
        return self.encoder.encode(pcm_data)

    def decode(self, opus_data: bytes) -> Optional[bytes]:
        """Decode Opus to PCM."""
        return self.decoder.decode(opus_data)

    @property
    def is_available(self) -> bool:
        """Check if codec is available."""
        return self.encoder.is_available and self.decoder.is_available

    @staticmethod
    def check_availability() -> bool:
        """Check if Opus codec is available on this system."""
        return _opuslib_available


def is_opus_available() -> bool:
    """Check if Opus codec is available."""
    return _opuslib_available


def get_codec_info() -> dict:
    """Get Opus codec information."""
    return {
        "available": _opuslib_available,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "frame_duration_ms": FRAME_DURATION_MS,
        "frame_size": FRAME_SIZE,
    }


__all__ = [
    "OpusEncoder",
    "OpusDecoder",
    "OpusCodec",
    "is_opus_available",
    "get_codec_info",
    "SAMPLE_RATE",
    "CHANNELS",
    "FRAME_DURATION_MS",
    "FRAME_SIZE",
]
