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
from typing import AsyncGenerator, Callable, Optional, Dict, Any, Union, Awaitable
import asyncio
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
    # PersonaPlex requires a non-empty text_prompt (crashes if None/empty)
    text_prompt: str = "You are a helpful voice assistant."
    sample_rate: int = SAMPLE_RATE
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0
    ssl_verify: bool = False  # PersonaPlex uses self-signed certs
    # PersonaPlex model parameters
    text_temperature: float = 0.8
    text_topk: int = 250
    audio_temperature: float = 0.8
    audio_topk: int = 250
    pad_mult: float = 1.0
    repetition_penalty: float = 1.0
    repetition_penalty_context: float = 1.0


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
        on_event: Optional[Callable[[VoiceEvent], Union[None, Awaitable[None]]]] = None,
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

    async def _emit_event(self, event: VoiceEvent):
        """Emit an event, handling both sync and async callbacks."""
        if self.on_event:
            result = self.on_event(event)
            if asyncio.iscoroutine(result):
                await result

    async def connect(self) -> bool:
        """
        Connect to PersonaPlex server.

        PersonaPlex uses a binary protocol:
        - 0x00 = handshake (server sends first)
        - 0x01 = audio (Opus encoded, bidirectional)
        - 0x02 = text (server -> client)
        - 0x03 = control (client -> server)

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
            import urllib.parse
            import random

            # Build WebSocket URL with ALL required query params
            url = self.config.personaplex_url

            # Generate random seeds for reproducibility
            text_seed = random.randint(0, 2**31)
            audio_seed = random.randint(0, 2**31)

            # PersonaPlex requires these query parameters
            # voice_prompt must include .pt extension if it's a file
            voice_prompt = self.config.voice_prompt
            if not voice_prompt.endswith('.pt'):
                voice_prompt = f"{voice_prompt}.pt"

            # text_prompt must be non-empty or PersonaPlex crashes
            text_prompt = self.config.text_prompt or "You are a helpful voice assistant."

            params = {
                "voice_prompt": voice_prompt,
                "text_prompt": text_prompt,
                "text_temperature": str(self.config.text_temperature),
                "text_topk": str(self.config.text_topk),
                "audio_temperature": str(self.config.audio_temperature),
                "audio_topk": str(self.config.audio_topk),
                "pad_mult": str(self.config.pad_mult),
                "repetition_penalty": str(self.config.repetition_penalty),
                "repetition_penalty_context": str(self.config.repetition_penalty_context),
                "text_seed": str(text_seed),
                "audio_seed": str(audio_seed),
            }
            query_string = urllib.parse.urlencode(params)
            full_url = f"{url}?{query_string}"

            # SSL context for self-signed certs
            ssl_context = None
            if url.startswith("wss://"):
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            _log.info(f"Connecting to PersonaPlex: {url}")
            _log.debug(f"Full URL: {full_url[:200]}...")

            self._websocket = await asyncio.wait_for(
                websockets.connect(
                    full_url,
                    ssl=ssl_context,
                    ping_interval=None,  # Disable ping - PersonaPlex doesn't respond
                    ping_timeout=None,
                ),
                timeout=10.0
            )

            _log.info("WebSocket connected, waiting for handshake...")

            # Wait for handshake message (0x00) from server
            # PersonaPlex processes system prompts before sending handshake - can take 30+ seconds
            try:
                handshake = await asyncio.wait_for(self._websocket.recv(), timeout=60.0)
                if isinstance(handshake, bytes) and len(handshake) >= 1 and handshake[0] == 0x00:
                    _log.info(f"Received handshake: {handshake.hex()}")
                else:
                    _log.warning(f"Unexpected handshake: {handshake!r}")
            except asyncio.TimeoutError:
                _log.error("No handshake received from PersonaPlex")
                await self._websocket.close()
                return False

            self._connected = True
            _log.info("Connected to PersonaPlex (handshake complete)")

            # Start receive and send loops
            self._recv_task = asyncio.create_task(self._receive_loop())
            self._send_task = asyncio.create_task(self._send_loop())

            return True

        except asyncio.TimeoutError:
            _log.error("Connection to PersonaPlex timed out")
            return False
        except Exception as e:
            _log.error(f"Failed to connect to PersonaPlex: {e}")
            import traceback
            _log.error(traceback.format_exc())
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
        """
        Receive messages from PersonaPlex.

        Binary protocol:
        - 0x00 = handshake (already handled in connect)
        - 0x01 = audio (Opus encoded)
        - 0x02 = text output
        - 0x03 = control
        - 0x05 = error
        """
        try:
            _log.info("Receive loop started, waiting for messages...")
            msg_count = 0
            async for message in self._websocket:
                msg_count += 1

                if not isinstance(message, bytes) or len(message) < 1:
                    _log.warning(f"Unexpected message format: {type(message)}")
                    continue

                msg_type = message[0]
                payload = message[1:] if len(message) > 1 else b""

                # Log every 50th message or any audio message
                if msg_count % 50 == 1 or msg_type == 0x01:
                    _log.info(f"Msg #{msg_count}: type=0x{msg_type:02x}, payload={len(payload)} bytes")

                if msg_type == 0x01:
                    # Audio message (Opus encoded)
                    if payload:
                        _log.info(f"Received audio from PersonaPlex: {len(payload)} bytes")
                        await self._handle_audio_message(payload)

                elif msg_type == 0x02:
                    # Text message (UTF-8)
                    try:
                        text = payload.decode("utf-8")
                        _log.info(f"Received text: {text[:100]}...")
                        await self._handle_text_token(text)
                    except UnicodeDecodeError:
                        _log.warning(f"Invalid UTF-8 in text message")

                elif msg_type == 0x05:
                    # Error message
                    try:
                        error_text = payload.decode("utf-8")
                        _log.error(f"PersonaPlex error: {error_text}")
                    except UnicodeDecodeError:
                        _log.error(f"PersonaPlex error (binary): {payload.hex()}")

                elif msg_type == 0x00:
                    # Late handshake (ignore, already handled)
                    _log.debug("Received late handshake, ignoring")

                else:
                    _log.debug(f"Unknown message type: 0x{msg_type:02x}")

            _log.info(f"Receive loop ended normally after {msg_count} messages")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.error(f"Receive loop error: {e}")
            import traceback
            _log.error(f"Traceback: {traceback.format_exc()}")
            self._connected = False

    async def _send_loop(self):
        """Send audio to PersonaPlex with binary protocol."""
        _log.info("Send loop started")
        frames_sent = 0
        try:
            while self._connected:
                try:
                    # Get Opus audio from outbound queue with timeout
                    opus_data = await asyncio.wait_for(
                        self._outbound_audio.get(),
                        timeout=0.1
                    )

                    if self._websocket and self._connected and opus_data:
                        # Prefix with message type 0x01 (audio)
                        message = bytes([0x01]) + opus_data
                        await self._websocket.send(message)
                        frames_sent += 1
                        if frames_sent == 1:
                            _log.info(f"First Opus frame: {len(opus_data)} bytes, header: {opus_data[:20].hex() if len(opus_data) >= 20 else opus_data.hex()}")
                        if frames_sent % 50 == 1:
                            _log.info(f"Sent {frames_sent} audio frames to PersonaPlex ({len(opus_data)} bytes each)")

                except asyncio.TimeoutError:
                    continue

        except asyncio.CancelledError:
            _log.info(f"Send loop cancelled after {frames_sent} frames")
            raise
        except Exception as e:
            _log.error(f"Send loop error: {e}")
            self._connected = False

    async def _handle_audio_message(self, opus_data: bytes):
        """Handle incoming Opus audio from PersonaPlex."""
        # Decode Opus to PCM
        pcm_data = self._codec.decode(opus_data)
        if not pcm_data:
            _log.warning(f"Failed to decode {len(opus_data)} bytes of Opus")
            return

        _log.info(f"Decoded Opus to PCM: {len(pcm_data)} bytes")

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
        await self._emit_event(event)

        # Queue for output
        await self._speech_queue.enqueue(
            pcm_data,
            self._current_generation_id,
            is_final=False,
        )

        await self._inbound_events.put(event)

    async def _handle_text_token(self, text: str):
        """Handle incoming text token from PersonaPlex."""
        self._agent_text_buffer += text

        # Emit text token event for real-time display
        if self._current_generation_id is None:
            self._current_generation_id = str(uuid.uuid4())
            self._current_turn_id = self._current_turn_id or str(uuid.uuid4())

        # Create event for the text token
        event = VoiceEvent(
            event_id=str(uuid.uuid4()),
            turn_id=self._current_turn_id,
            generation_id=self._current_generation_id,
            source="agent",
            channel=Channel.TRANSCRIPT,
            type=EventType.TEXT_TOKEN,
            commit=CommitLevel.EPHEMERAL,
            payload={"token": text, "buffer": self._agent_text_buffer},
        )
        await self._emit_event(event)

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

                    await self._emit_event(event)

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
            _log.warning("send_audio called but not connected")
            return False

        _log.debug(f"send_audio: received {len(pcm_data)} bytes PCM")

        # Encode to Opus
        opus_frames = self._codec.encoder.encode_frames(pcm_data)
        _log.debug(f"send_audio: encoded to {len(opus_frames)} Opus frames")

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

        await self._emit_event(event)

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
        _log.info("stream_audio: Starting to yield chunks from speech queue")
        chunk_count = 0
        async for chunk in self._speech_queue.stream():
            if chunk.data:
                chunk_count += 1
                if chunk_count <= 3 or chunk_count % 10 == 0:
                    _log.info(f"stream_audio: Yielding chunk #{chunk_count}, {len(chunk.data)} bytes")
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
