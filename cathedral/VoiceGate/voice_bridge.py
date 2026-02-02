"""
Voice Bridge - WebSocket connection to PersonaPlex.

Handles bidirectional audio streaming using PersonaPlex's native
WebSocket protocol. PersonaPlex sends/receives Opus-encoded audio
frames with text tokens for transcription.

Architecture:
    Browser <---> Cathedral VoiceBridge <---> PersonaPlex
                      |
                      v
               Event Envelope System
                      |
            +---------+---------+
            |         |         |
         Audio   Transcript   Tool
        Channel   Channel    Channel
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Optional, Dict, Any
from enum import IntEnum

from cathedral.shared.gate import GateLogger
from .events import (
    VoiceEvent, Channel, CommitLevel, EventType,
    user_utterance, agent_speech_chunk, agent_speech_complete,
    interrupt_event, cancel_event,
)
from .opus_codec import OpusCodec, OpusDecoder, OpusEncoder, SAMPLE_RATE, FRAME_SIZE
from .speech_queue import SpeechQueue, get_speech_queue_manager

_log = GateLogger.get("VoiceGate.Bridge")


class PersonaPlexMessageType(IntEnum):
    """PersonaPlex WebSocket message types."""
    # Binary messages
    AUDIO_USER = 1      # User audio (Opus) -> PersonaPlex
    AUDIO_AGENT = 2     # Agent audio (Opus) <- PersonaPlex

    # Text messages (JSON)
    CONFIG = 10         # Configuration
    TEXT_TOKEN = 11     # Text token from agent
    TEXT_FINAL = 12     # Final text (committed)
    CONTROL = 20        # Control messages (interrupt, etc.)
    ERROR = 99          # Error message


@dataclass
class BridgeConfig:
    """Configuration for the voice bridge."""
    personaplex_url: str = "wss://localhost:8998/api/chat"
    voice_prompt: str = "NATF2"
    text_prompt: str = ""
    sample_rate: int = SAMPLE_RATE
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0
    ssl_verify: bool = False  # PersonaPlex uses self-signed certs


class VoiceBridge:
    """
    Bidirectional voice bridge to PersonaPlex.

    Handles:
    - WebSocket connection lifecycle
    - Opus encoding/decoding
    - Event envelope generation
    - Turn management
    - Interrupt handling
    """

    def __init__(
        self,
        config: BridgeConfig,
        on_event: Optional[Callable[[VoiceEvent], None]] = None,
    ):
        """
        Initialize the voice bridge.

        Args:
            config: Bridge configuration
            on_event: Callback for voice events
        """
        self.config = config
        self.on_event = on_event

        # Connection state
        self._websocket = None
        self._connected = False
        self._connecting = False

        # Codec
        self._codec = OpusCodec(sample_rate=config.sample_rate)

        # Turn state
        self._current_turn_id: Optional[str] = None
        self._current_generation_id: Optional[str] = None
        self._agent_text_buffer: str = ""

        # Speech queue for output
        self._speech_queue = SpeechQueue()

        # Tasks
        self._recv_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None

        # Queues for bidirectional communication
        self._outbound_audio: asyncio.Queue[bytes] = asyncio.Queue()
        self._inbound_events: asyncio.Queue[VoiceEvent] = asyncio.Queue()

    async def connect(self) -> bool:
        """
        Connect to PersonaPlex server.

        Returns:
            True if connected successfully
        """
        if self._connected:
            return True

        if self._connecting:
            # Wait for existing connection attempt
            for _ in range(50):  # 5 second timeout
                await asyncio.sleep(0.1)
                if self._connected:
                    return True
            return False

        self._connecting = True

        try:
            import websockets
            import ssl

            # Build WebSocket URL with query params
            url = self.config.personaplex_url
            params = f"?voice_prompt={self.config.voice_prompt}"
            if self.config.text_prompt:
                import urllib.parse
                params += f"&text_prompt={urllib.parse.quote(self.config.text_prompt)}"
            full_url = url + params

            # SSL context for self-signed certs
            ssl_context = None
            if url.startswith("wss://"):
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            _log.info(f"Connecting to PersonaPlex: {url}")

            self._websocket = await asyncio.wait_for(
                websockets.connect(
                    full_url,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                ),
                timeout=10.0
            )

            self._connected = True
            _log.info("Connected to PersonaPlex")

            # Start receive and send loops
            self._recv_task = asyncio.create_task(self._receive_loop())
            self._send_task = asyncio.create_task(self._send_loop())

            return True

        except asyncio.TimeoutError:
            _log.error("Connection to PersonaPlex timed out")
            return False
        except Exception as e:
            _log.error(f"Failed to connect to PersonaPlex: {e}")
            return False
        finally:
            self._connecting = False

    async def disconnect(self):
        """Disconnect from PersonaPlex."""
        self._connected = False

        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None

        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
            self._send_task = None

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        _log.info("Disconnected from PersonaPlex")

    async def _receive_loop(self):
        """Receive messages from PersonaPlex."""
        try:
            async for message in self._websocket:
                if isinstance(message, bytes):
                    # Binary message - audio from agent
                    await self._handle_audio_message(message)
                else:
                    # Text message - JSON
                    await self._handle_text_message(message)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.error(f"Receive loop error: {e}")
            self._connected = False

    async def _send_loop(self):
        """Send audio to PersonaPlex."""
        try:
            while self._connected:
                try:
                    # Get audio from outbound queue with timeout
                    audio_data = await asyncio.wait_for(
                        self._outbound_audio.get(),
                        timeout=0.1
                    )

                    if self._websocket and self._connected:
                        await self._websocket.send(audio_data)

                except asyncio.TimeoutError:
                    continue

        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.error(f"Send loop error: {e}")
            self._connected = False

    async def _handle_audio_message(self, data: bytes):
        """Handle incoming audio from PersonaPlex."""
        # Decode Opus to PCM
        pcm_data = self._codec.decode(data)
        if not pcm_data:
            return

        # Generate event
        if self._current_generation_id is None:
            self._current_generation_id = str(uuid.uuid4())
            self._current_turn_id = self._current_turn_id or str(uuid.uuid4())

        event = agent_speech_chunk(
            turn_id=self._current_turn_id,
            generation_id=self._current_generation_id,
            audio_data=pcm_data,
        )

        # Emit event
        if self.on_event:
            self.on_event(event)

        # Queue for output
        await self._speech_queue.enqueue(
            pcm_data,
            self._current_generation_id,
            is_final=False,
        )

        await self._inbound_events.put(event)

    async def _handle_text_message(self, message: str):
        """Handle incoming text message from PersonaPlex."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "text_token":
                # Partial text token
                token = data.get("token", "")
                self._agent_text_buffer += token

            elif msg_type == "text_final" or msg_type == "turn_end":
                # Final text - create committed event
                text = self._agent_text_buffer or data.get("text", "")

                if text and self._current_generation_id:
                    event = agent_speech_complete(
                        turn_id=self._current_turn_id or str(uuid.uuid4()),
                        generation_id=self._current_generation_id,
                        text=text,
                    )

                    if self.on_event:
                        self.on_event(event)

                    await self._inbound_events.put(event)

                    # Mark speech as final
                    await self._speech_queue.enqueue(
                        b"",  # Empty final chunk
                        self._current_generation_id,
                        is_final=True,
                        text_fragment=text,
                    )

                # Reset state
                self._agent_text_buffer = ""
                self._current_generation_id = None

            elif msg_type == "error":
                _log.error(f"PersonaPlex error: {data.get('message', 'unknown')}")

        except json.JSONDecodeError:
            _log.warning(f"Invalid JSON from PersonaPlex: {message[:100]}")
        except Exception as e:
            _log.error(f"Error handling text message: {e}")

    async def send_audio(self, pcm_data: bytes) -> bool:
        """
        Send user audio to PersonaPlex.

        Args:
            pcm_data: Raw PCM audio (16-bit, mono, 24kHz)

        Returns:
            True if sent successfully
        """
        if not self._connected:
            return False

        # Encode to Opus
        opus_frames = self._codec.encoder.encode_frames(pcm_data)

        for frame in opus_frames:
            await self._outbound_audio.put(frame)

        return True

    async def send_interrupt(self) -> Optional[str]:
        """
        Send interrupt signal - user is speaking.

        Returns:
            The cancelled generation ID, or None
        """
        if not self._current_generation_id:
            return None

        cancelled_id = self._current_generation_id

        # Cancel the speech queue
        self._speech_queue.cancel(cancelled_id)

        # Emit interrupt event
        event = interrupt_event(
            turn_id=self._current_turn_id or str(uuid.uuid4()),
            cancelled_generation_id=cancelled_id,
        )

        if self.on_event:
            self.on_event(event)

        # Reset agent state
        self._current_generation_id = None
        self._agent_text_buffer = ""

        _log.info(f"Interrupt sent, cancelled generation {cancelled_id[:8]}")

        return cancelled_id

    async def stream_events(self) -> AsyncGenerator[VoiceEvent, None]:
        """
        Stream voice events from the bridge.

        Yields events as they arrive from PersonaPlex.
        """
        while self._connected:
            try:
                event = await asyncio.wait_for(
                    self._inbound_events.get(),
                    timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue

    async def stream_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Stream decoded audio from PersonaPlex.

        Yields PCM audio chunks for playback.
        """
        async for chunk in self._speech_queue.stream():
            if chunk.data:
                yield chunk.data

    def start_turn(self, source: str = "user") -> str:
        """
        Start a new conversation turn.

        Args:
            source: Who is starting the turn ("user" or "agent")

        Returns:
            The new turn ID
        """
        self._current_turn_id = str(uuid.uuid4())
        if source == "agent":
            self._current_generation_id = str(uuid.uuid4())
        return self._current_turn_id

    @property
    def is_connected(self) -> bool:
        """Check if connected to PersonaPlex."""
        return self._connected

    @property
    def current_generation_id(self) -> Optional[str]:
        """Get the current speech generation ID."""
        return self._current_generation_id

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "connected": self._connected,
            "connecting": self._connecting,
            "codec_available": self._codec.is_available,
            "current_turn_id": self._current_turn_id,
            "current_generation_id": self._current_generation_id,
            "speech_queue_stats": self._speech_queue.get_stats(),
        }


# Factory function
async def create_voice_bridge(
    personaplex_url: str,
    voice: str = "NATF2",
    persona: str = "",
    on_event: Optional[Callable[[VoiceEvent], None]] = None,
) -> VoiceBridge:
    """
    Create and connect a voice bridge.

    Args:
        personaplex_url: PersonaPlex WebSocket URL
        voice: Voice prompt name
        persona: Optional persona/system prompt
        on_event: Event callback

    Returns:
        Connected VoiceBridge instance
    """
    config = BridgeConfig(
        personaplex_url=personaplex_url,
        voice_prompt=voice,
        text_prompt=persona,
    )

    bridge = VoiceBridge(config, on_event)
    await bridge.connect()

    return bridge


__all__ = [
    "BridgeConfig",
    "VoiceBridge",
    "PersonaPlexMessageType",
    "create_voice_bridge",
]
