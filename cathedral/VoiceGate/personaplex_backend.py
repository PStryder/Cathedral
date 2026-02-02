"""
PersonaPlex synthesis backend.

Provides integration with NVIDIA PersonaPlex for voice synthesis.
PersonaPlex is a full-duplex speech-to-speech system that supports
persona control through text prompts and voice conditioning.

See: https://github.com/NVIDIA/personaplex
"""

from __future__ import annotations

import asyncio
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional
from concurrent.futures import ThreadPoolExecutor

from cathedral.shared.gate import GateLogger
from .models import PersonaPlexConfig, PersonaPlexVoice

_log = GateLogger.get("VoiceGate.PersonaPlex")

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="personaplex_")


class PersonaPlexSynthesizer:
    """
    PersonaPlex synthesis backend.

    Can operate in two modes:
    1. Server mode: Connects to a running PersonaPlex server via WebSocket
    2. Offline mode: Uses the offline inference directly

    Note: PersonaPlex is primarily a speech-to-speech system. For TTS,
    we use the offline mode with a structured prompt to generate speech.
    """

    # Voice prompt file names
    VOICE_PROMPTS = {v.value: f"{v.value}.pt" for v in PersonaPlexVoice}

    def __init__(self, config: PersonaPlexConfig):
        """
        Initialize the PersonaPlex synthesizer.

        Args:
            config: PersonaPlex configuration
        """
        self.config = config
        self.sample_rate = 24000  # PersonaPlex uses 24kHz

        # State
        self._available = False
        self._connected = False
        self._websocket = None
        self._server_process = None
        self._voice_prompt_dir: Optional[Path] = None
        self._error: Optional[str] = None
        self._use_websocket = False

        # Check availability
        self._check_availability()

    def _check_availability(self):
        """Check if PersonaPlex is available."""
        try:
            # For WebSocket mode, we only need websockets library
            # (moshi is only needed for offline mode)
            import importlib.util

            # Check for websockets (required for server mode)
            ws_spec = importlib.util.find_spec("websockets")
            if ws_spec is not None:
                # WebSocket mode available - will connect to PersonaPlex server
                self._available = True
                self._use_websocket = True
                _log.info("PersonaPlex available (WebSocket mode)")
                return

            # Check if moshi package is installed (for offline mode)
            spec = importlib.util.find_spec("moshi")
            if spec is None:
                self._error = "Neither websockets nor moshi package installed"
                self._available = False
                return

            # Check for GPU (only needed for offline mode)
            import torch
            if not torch.cuda.is_available():
                self._error = "No GPU available for PersonaPlex offline mode"
                self._available = False
                return

            self._available = True
            self._use_websocket = False
            _log.info("PersonaPlex available (offline mode)")

        except ImportError as e:
            self._error = f"Missing dependency: {e}"
            self._available = False

    def is_available(self) -> bool:
        """Check if PersonaPlex is available."""
        return self._available

    def is_connected(self) -> bool:
        """Check if connected to PersonaPlex server."""
        return self._connected

    async def connect(self) -> bool:
        """
        Connect to PersonaPlex server.

        If server is not running, attempts to start it.

        Returns:
            True if connected successfully
        """
        if not self._available:
            return False

        try:
            import websockets
            import ssl

            # Try to connect to existing server
            ws_url = self.config.server_url
            if self.config.ssl_enabled:
                ws_url = ws_url.replace("ws://", "wss://")

            # Create SSL context that accepts self-signed certificates
            ssl_context = None
            if ws_url.startswith("wss://"):
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            try:
                self._websocket = await asyncio.wait_for(
                    websockets.connect(ws_url, ssl=ssl_context),
                    timeout=5.0
                )
                self._connected = True
                _log.info(f"Connected to PersonaPlex server at {ws_url}")
                return True

            except (ConnectionRefusedError, asyncio.TimeoutError, OSError) as e:
                _log.info(f"PersonaPlex server not available: {e}")
                return False

        except ImportError:
            self._error = "websockets package not installed"
            return False
        except Exception as e:
            self._error = str(e)
            _log.error(f"Failed to connect to PersonaPlex: {e}")
            return False

    async def disconnect(self):
        """Disconnect from PersonaPlex server."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        self._connected = False

    def _get_voice_prompt_path(self, voice: PersonaPlexVoice) -> Optional[Path]:
        """Get the path to a voice prompt file."""
        if self._voice_prompt_dir is None:
            # Check common locations
            candidates = [
                Path.home() / ".cache" / "personaplex" / "voice_prompts",
                Path("/opt/personaplex/voice_prompts"),
                Path(os.getenv("PERSONAPLEX_VOICE_DIR", "")),
            ]
            for path in candidates:
                if path.exists() and (path / f"{voice.value}.pt").exists():
                    self._voice_prompt_dir = path
                    break

        if self._voice_prompt_dir:
            prompt_path = self._voice_prompt_dir / f"{voice.value}.pt"
            if prompt_path.exists():
                return prompt_path

        return None

    def _synthesize_offline_sync(
        self,
        text: str,
        voice: PersonaPlexVoice,
        persona_prompt: Optional[str] = None,
    ) -> bytes:
        """
        Synthesize text using offline mode (synchronous).

        This uses PersonaPlex's offline inference to generate speech.
        Since PersonaPlex is S2S, we craft a prompt that instructs
        the model to speak the given text.

        Args:
            text: Text to synthesize
            voice: Voice to use
            persona_prompt: Optional persona description

        Returns:
            PCM audio bytes
        """
        try:
            import tempfile
            import numpy as np

            # Create temporary files
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Create a short silence as "user input"
                # PersonaPlex expects user audio input
                silence_samples = int(0.5 * self.sample_rate)  # 0.5 seconds
                silence = np.zeros(silence_samples, dtype=np.float32)

                input_wav = tmpdir / "input.wav"
                output_wav = tmpdir / "output.wav"

                # Write silence as input
                import soundfile as sf
                sf.write(str(input_wav), silence, self.sample_rate)

                # Build the text prompt
                # We instruct the model to speak the text
                full_prompt = ""
                if persona_prompt:
                    full_prompt = persona_prompt + "\n\n"
                full_prompt += f"Please say the following exactly: \"{text}\""

                # Get voice prompt path
                voice_prompt_path = self._get_voice_prompt_path(voice)
                voice_arg = str(voice_prompt_path) if voice_prompt_path else voice.value

                # Run offline inference
                cmd = [
                    sys.executable, "-m", "moshi.offline",
                    "--input-wav", str(input_wav),
                    "--output-wav", str(output_wav),
                    "--text-prompt", full_prompt,
                    "--voice-prompt", voice_arg,
                ]

                if self.config.cpu_offload:
                    cmd.append("--cpu-offload")

                env = os.environ.copy()
                if self.config.hf_token:
                    env["HF_TOKEN"] = self.config.hf_token

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=60,
                )

                if result.returncode != 0:
                    _log.error(f"PersonaPlex offline failed: {result.stderr}")
                    return b""

                # Read output audio
                if output_wav.exists():
                    audio, sr = sf.read(str(output_wav), dtype=np.float32)
                    # Convert to 16-bit PCM
                    audio_int16 = (audio * 32767).astype(np.int16)
                    return audio_int16.tobytes()

                return b""

        except Exception as e:
            _log.error(f"Offline synthesis failed: {e}")
            return b""

    async def synthesize(
        self,
        text: str,
        voice: Optional[PersonaPlexVoice] = None,
        persona_prompt: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to audio, yielding chunks.

        Uses PersonaPlex's REST TTS endpoint (preferred) or falls back to
        WebSocket streaming or offline mode.

        Args:
            text: Text to synthesize
            voice: Voice to use (default from config)
            persona_prompt: Optional persona description

        Yields:
            Audio chunks as bytes
        """
        if not self._available:
            _log.warning("PersonaPlex not available")
            return

        voice = voice or self.config.voice_prompt
        persona = persona_prompt or self.config.text_prompt

        # Try HTTP TTS endpoint first (simplest and most reliable)
        audio_bytes = await self._synthesize_http(text, voice, persona)
        if audio_bytes:
            # Yield in chunks for streaming
            chunk_size = 4096  # ~85ms at 24kHz 16-bit mono
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]
            return

        # Fall back to offline mode if HTTP failed and moshi is available
        if not self._use_websocket:
            loop = asyncio.get_running_loop()
            audio_bytes = await loop.run_in_executor(
                _executor,
                self._synthesize_offline_sync,
                text,
                voice,
                persona,
            )

            if not audio_bytes:
                return

            # Yield in chunks for streaming
            chunk_size = 4096
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]
        else:
            _log.warning("PersonaPlex synthesis failed - no audio generated")

    async def _synthesize_http(
        self,
        text: str,
        voice: PersonaPlexVoice,
        persona: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Synthesize using PersonaPlex's HTTP TTS endpoint.

        Args:
            text: Text to synthesize
            voice: Voice to use
            persona: Optional persona prompt

        Returns:
            PCM audio bytes or None if failed
        """
        try:
            import httpx
            import ssl

            # Build TTS URL from WebSocket URL
            base_url = self.config.server_url.replace("wss://", "https://").replace("ws://", "http://")
            # Remove /api/chat path if present
            if base_url.endswith("/api/chat"):
                base_url = base_url[:-9]
            tts_url = f"{base_url}/api/tts"

            # Prepare request
            payload = {
                "text": text,
                "voice": voice.value if hasattr(voice, 'value') else str(voice),
            }
            if persona:
                payload["persona"] = persona

            _log.debug(f"Calling PersonaPlex TTS: {tts_url}")

            # Make request with SSL verification disabled (self-signed cert)
            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                response = await client.post(tts_url, json=payload)

                if response.status_code == 200:
                    # Response is WAV audio - extract PCM data
                    wav_data = response.content
                    pcm_data = self._wav_to_pcm(wav_data)
                    _log.info(f"PersonaPlex TTS generated {len(pcm_data)} bytes")
                    return pcm_data
                else:
                    error = response.text
                    _log.warning(f"PersonaPlex TTS failed: {response.status_code} - {error}")
                    return None

        except ImportError:
            _log.warning("httpx not installed - cannot use HTTP TTS endpoint")
            return None
        except Exception as e:
            _log.warning(f"PersonaPlex HTTP TTS failed: {e}")
            return None

    def _wav_to_pcm(self, wav_data: bytes) -> bytes:
        """Extract PCM data from WAV file."""
        import io
        import wave

        try:
            buffer = io.BytesIO(wav_data)
            with wave.open(buffer, 'rb') as wav:
                return wav.readframes(wav.getnframes())
        except Exception as e:
            _log.error(f"Failed to extract PCM from WAV: {e}")
            return b""

    async def synthesize_stream_ws(
        self,
        text: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize using WebSocket connection to PersonaPlex server.

        This is for real-time streaming when server is available.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as bytes
        """
        if not self._connected or not self._websocket:
            _log.warning("Not connected to PersonaPlex server")
            return

        try:
            # PersonaPlex server expects Opus-encoded audio
            # For TTS, we send a text message and receive audio
            # This is a simplified protocol - actual implementation
            # would need to match PersonaPlex's WebSocket protocol

            # Send text as a command
            await self._websocket.send(f"SAY:{text}")

            # Receive audio chunks
            while True:
                try:
                    data = await asyncio.wait_for(
                        self._websocket.recv(),
                        timeout=10.0
                    )

                    if isinstance(data, bytes):
                        # Audio chunk
                        yield data
                    elif isinstance(data, str):
                        if data == "END":
                            break

                except asyncio.TimeoutError:
                    break

        except Exception as e:
            _log.error(f"WebSocket synthesis failed: {e}")

    def get_info(self) -> dict:
        """Get synthesizer information."""
        return {
            "backend": "personaplex",
            "available": self._available,
            "connected": self._connected,
            "server_url": self.config.server_url,
            "voice": self.config.voice_prompt.value if self.config.voice_prompt else None,
            "sample_rate": self.sample_rate,
            "error": self._error,
            "voices": [v.value for v in PersonaPlexVoice],
        }

    def list_voices(self) -> list[dict]:
        """List available voices."""
        voices = []
        for voice in PersonaPlexVoice:
            category = "natural" if voice.value.startswith("NAT") else "variety"
            gender = "female" if voice.value.endswith(tuple("12345")) and "F" in voice.value else "male"

            voices.append({
                "id": voice.value,
                "name": voice.value,
                "category": category,
                "gender": gender,
                "available": self._get_voice_prompt_path(voice) is not None,
            })

        return voices

    async def shutdown(self):
        """Shutdown the synthesizer."""
        await self.disconnect()

        # Stop server if we started it
        if self._server_process:
            self._server_process.terminate()
            self._server_process = None


__all__ = ["PersonaPlexSynthesizer"]
