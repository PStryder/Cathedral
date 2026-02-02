"""
Opus Codec Wrapper for VoiceGate.

Provides encoding/decoding of Opus audio for PersonaPlex communication.
PersonaPlex uses the sphn library's OpusStreamWriter/Reader for streaming Opus.

This module uses sphn for compatibility with PersonaPlex, falling back to
opuslib for basic frame-by-frame encoding if sphn is not available.
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

# Try to import sphn (preferred for PersonaPlex compatibility)
_sphn_available = False
try:
    import sphn
    _sphn_available = True
    _log.info("sphn available - using PersonaPlex-compatible Opus codec")
except ImportError:
    _log.warning("sphn not installed - PersonaPlex audio may not work (pip install sphn)")

# Fallback to opuslib for basic encoding
_opuslib_available = False
try:
    import opuslib
    import opuslib.api.encoder
    import opuslib.api.decoder
    _opuslib_available = True
    if not _sphn_available:
        _log.info("opuslib available as fallback")
except ImportError:
    if not _sphn_available:
        _log.warning("Neither sphn nor opuslib installed - Opus codec disabled")


class SphnOpusEncoder:
    """
    Opus encoder using sphn library for PersonaPlex compatibility.

    Uses OpusStreamWriter for streaming Opus format that matches PersonaPlex.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """Initialize the sphn Opus encoder."""
        self.sample_rate = sample_rate
        self._writer = None

        if _sphn_available:
            import sphn
            self._writer = sphn.OpusStreamWriter(sample_rate)
            _log.debug(f"sphn Opus encoder initialized: {sample_rate}Hz")

    def encode(self, pcm_data: bytes) -> Optional[bytes]:
        """
        Encode PCM audio to Opus using sphn streaming format.

        Args:
            pcm_data: Raw PCM audio (16-bit signed, little-endian)

        Returns:
            Opus-encoded bytes in sphn streaming format
        """
        if self._writer is None:
            return None

        try:
            import numpy as np

            # Convert bytes to float32 array (sphn expects float32 in [-1, 1])
            pcm_int16 = np.frombuffer(pcm_data, dtype=np.int16)
            pcm_float = pcm_int16.astype(np.float32) / 32768.0

            # sphn 0.2+ API: append_pcm returns bytes directly
            opus_bytes = self._writer.append_pcm(pcm_float)
            return opus_bytes if opus_bytes else None

        except Exception as e:
            _log.error(f"sphn Opus encode error: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if encoder is available."""
        return self._writer is not None


class SphnOpusDecoder:
    """
    Opus decoder using sphn library for PersonaPlex compatibility.

    Uses OpusStreamReader for streaming Opus format that matches PersonaPlex.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """Initialize the sphn Opus decoder."""
        self.sample_rate = sample_rate
        self._reader = None

        if _sphn_available:
            import sphn
            self._reader = sphn.OpusStreamReader(sample_rate)
            _log.debug(f"sphn Opus decoder initialized: {sample_rate}Hz")

    def decode(self, opus_data: bytes) -> Optional[bytes]:
        """
        Decode Opus audio to PCM using sphn streaming format.

        Args:
            opus_data: Opus-encoded bytes from PersonaPlex

        Returns:
            Raw PCM audio (16-bit signed, little-endian), or None on error
        """
        if self._reader is None:
            return None

        try:
            import numpy as np

            # sphn 0.2+ API: append_bytes returns numpy array directly
            pcm_float = self._reader.append_bytes(opus_data)

            if pcm_float is None or len(pcm_float) == 0:
                return None

            # Check audio level
            max_level = float(np.max(np.abs(pcm_float)))
            _log.info(f"sphn decode: {len(pcm_float)} samples, max_level={max_level:.6f}")

            # Skip near-silent chunks (decoder warmup produces ~0.00006 max)
            if max_level < 0.01:
                _log.debug(f"Skipping silent chunk (max={max_level:.6f})")
                return None

            # Convert float32 to int16 bytes
            pcm_int16 = (pcm_float * 32767).astype(np.int16)
            return pcm_int16.tobytes()

        except Exception as e:
            _log.error(f"sphn Opus decode error: {e}")
            return None

    def close(self):
        """Close the decoder stream."""
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Check if decoder is available."""
        return self._reader is not None


class OpusEncoder:
    """
    Opus audio encoder.

    Uses sphn if available (for PersonaPlex compatibility), otherwise opuslib.
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
        self._sphn_encoder = None

        # Prefer sphn for PersonaPlex compatibility
        if _sphn_available:
            self._sphn_encoder = SphnOpusEncoder(sample_rate)
        elif _opuslib_available:
            import opuslib
            app_map = {
                "voip": opuslib.APPLICATION_VOIP,
                "audio": opuslib.APPLICATION_AUDIO,
                "lowdelay": opuslib.APPLICATION_RESTRICTED_LOWDELAY,
            }
            app = app_map.get(application, opuslib.APPLICATION_VOIP)

            try:
                self._encoder = opuslib.Encoder(sample_rate, channels, app)
                _log.debug(f"opuslib encoder initialized: {sample_rate}Hz, {channels}ch")
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
        # Use sphn if available
        if self._sphn_encoder is not None:
            return self._sphn_encoder.encode(pcm_data)

        # Fall back to opuslib
        if self._encoder is None:
            return None

        try:
            expected_size = self.frame_size * self.channels * 2

            if len(pcm_data) != expected_size:
                _log.warning(f"PCM size mismatch: got {len(pcm_data)}, expected {expected_size}")
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
        """
        if self._sphn_encoder is not None:
            # sphn handles framing internally
            result = self._sphn_encoder.encode(pcm_data)
            return [result] if result else []

        if self._encoder is None:
            return []

        frames = []
        bytes_per_frame = self.frame_size * self.channels * 2

        for i in range(0, len(pcm_data), bytes_per_frame):
            chunk = pcm_data[i:i + bytes_per_frame]
            if len(chunk) < bytes_per_frame:
                chunk = chunk + b'\x00' * (bytes_per_frame - len(chunk))
            encoded = self.encode(chunk)
            if encoded:
                frames.append(encoded)

        return frames

    @property
    def is_available(self) -> bool:
        """Check if encoder is available."""
        if self._sphn_encoder is not None:
            return self._sphn_encoder.is_available
        return self._encoder is not None


class OpusDecoder:
    """
    Opus audio decoder.

    Uses sphn if available (for PersonaPlex compatibility), otherwise opuslib.
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
        self._sphn_decoder = None

        # Prefer sphn for PersonaPlex compatibility
        if _sphn_available:
            self._sphn_decoder = SphnOpusDecoder(sample_rate)
        elif _opuslib_available:
            import opuslib
            try:
                self._decoder = opuslib.Decoder(sample_rate, channels)
                _log.debug(f"opuslib decoder initialized: {sample_rate}Hz, {channels}ch")
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
        # Use sphn if available
        if self._sphn_decoder is not None:
            return self._sphn_decoder.decode(opus_data)

        # Fall back to opuslib
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
        """
        pcm_chunks = []
        for frame in opus_frames:
            pcm = self.decode(frame)
            if pcm:
                pcm_chunks.append(pcm)
        return b''.join(pcm_chunks)

    def close(self):
        """Close the decoder."""
        if self._sphn_decoder is not None:
            self._sphn_decoder.close()

    @property
    def is_available(self) -> bool:
        """Check if decoder is available."""
        if self._sphn_decoder is not None:
            return self._sphn_decoder.is_available
        return self._decoder is not None


class OpusCodec:
    """
    Combined Opus encoder/decoder for bidirectional audio.

    Uses sphn for PersonaPlex-compatible streaming Opus format.
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

    def close(self):
        """Close the codec (release resources)."""
        self.decoder.close()

    @property
    def is_available(self) -> bool:
        """Check if codec is available."""
        return self.encoder.is_available and self.decoder.is_available

    @staticmethod
    def check_availability() -> bool:
        """Check if Opus codec is available on this system."""
        return _sphn_available or _opuslib_available


def is_opus_available() -> bool:
    """Check if Opus codec is available."""
    return _sphn_available or _opuslib_available


def get_codec_info() -> dict:
    """Get Opus codec information."""
    return {
        "available": _sphn_available or _opuslib_available,
        "backend": "sphn" if _sphn_available else ("opuslib" if _opuslib_available else "none"),
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
