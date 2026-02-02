"""
Voice API endpoints for Cathedral.

Provides:
- WebSocket endpoint for streaming TTS audio (one-way)
- WebSocket endpoint for full-duplex voice conversation (bidirectional)
- HTTP endpoints for voice status and configuration

Full-duplex mode uses the three-channel architecture:
- Audio channel: Real-time PCM/Opus bidirectional streaming
- Transcript channel: Durable text committed to MemoryGate
- Tool channel: Silent execution (not spoken)
"""

import asyncio
import base64
import json
import uuid
from typing import Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from cathedral import VoiceGate
from cathedral.VoiceGate import (
    VoiceBridge,
    BridgeConfig,
    TurnManager,
    TurnManagerConfig,
    TranscriptWriter,
    VoiceEvent,
    Channel,
    CommitLevel,
    EventType,
    is_opus_available,
)


def create_router(emit_event: Callable = None) -> APIRouter:
    """
    Create the voice API router.

    Args:
        emit_event: Event emitter for console output

    Returns:
        FastAPI router with voice endpoints
    """
    router = APIRouter(prefix="/api/voice", tags=["voice"])

    @router.get("/status")
    async def get_voice_status():
        """Get voice synthesis status."""
        VoiceGate.initialize()
        status = VoiceGate.get_status()
        return JSONResponse(content={
            "available": status.available,
            "enabled": VoiceGate.is_enabled(),
            "provider": status.provider.value if status.provider else "disabled",
            "gpu_available": status.gpu_available,
            "model_loaded": status.model_loaded,
            "model_name": status.model_name,
            "error": status.error,
            "personaplex_connected": status.personaplex_connected,
            "current_voice": status.current_voice,
        })

    @router.get("/info")
    async def get_voice_info():
        """Get detailed voice gate information."""
        VoiceGate.initialize()
        return JSONResponse(content=VoiceGate.get_info())

    @router.get("/voices")
    async def list_voices():
        """List available voices for the current provider."""
        VoiceGate.initialize()
        voices = VoiceGate.list_voices()
        return JSONResponse(content={"voices": voices})

    @router.get("/presets")
    async def get_voice_presets():
        """Get available voice presets."""
        VoiceGate.initialize()
        presets = VoiceGate.get_voice_presets()
        return JSONResponse(content={"presets": presets})

    @router.post("/voice")
    async def set_voice(request: dict):
        """
        Set the active voice.

        Request body:
        {
            "voice_id": "NATF2"  (PersonaPlex voice ID)
        }
        """
        VoiceGate.initialize()

        voice_id = request.get("voice_id")
        if not voice_id:
            raise HTTPException(status_code=400, detail="No voice_id provided")

        success = await VoiceGate.set_voice(voice_id)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to set voice: {voice_id}"
            )

        if emit_event:
            await emit_event("voice", f"Voice changed to {voice_id}")

        return JSONResponse(content={
            "status": "ok",
            "voice_id": voice_id,
        })

    @router.websocket("/{thread_uid}")
    async def voice_stream(websocket: WebSocket, thread_uid: str):
        """
        WebSocket endpoint for streaming TTS audio.

        Protocol:
        - Client connects and optionally sends configuration
        - Server sends audio chunks as binary messages
        - Final chunk has is_final=True in accompanying JSON

        Message format (server -> client):
        - Binary: Raw PCM audio bytes
        - Text: JSON with metadata {"sequence": N, "is_final": bool, "duration_ms": N}
        """
        await websocket.accept()

        if emit_event:
            await emit_event("voice", f"Voice stream connected for thread {thread_uid[:8]}")

        # Check if voice is available
        VoiceGate.initialize()
        if not VoiceGate.is_available():
            await websocket.send_json({
                "error": "Voice synthesis not available",
                "reason": VoiceGate.get_status().error or "No GPU or TTS disabled",
            })
            await websocket.close(code=4000, reason="Voice not available")
            return

        # Get audio queue for this thread
        queue = await VoiceGate.get_audio_queue(thread_uid)

        try:
            # Stream audio chunks to client
            async for chunk in queue.stream():
                # Send metadata first
                await websocket.send_json({
                    "sequence": chunk.sequence,
                    "is_final": chunk.is_final,
                    "duration_ms": chunk.duration_ms,
                    "sample_rate": chunk.sample_rate,
                })

                # Send audio data as binary
                if chunk.data:
                    await websocket.send_bytes(chunk.data)

                if chunk.is_final:
                    break

        except WebSocketDisconnect:
            if emit_event:
                await emit_event("voice", f"Voice stream disconnected for thread {thread_uid[:8]}")

        except Exception as e:
            if emit_event:
                await emit_event("voice", f"Voice stream error: {e}")
            await websocket.close(code=4001, reason=str(e))

    @router.websocket("/conversation/{thread_uid}")
    async def voice_conversation(websocket: WebSocket, thread_uid: str):
        """
        Full-duplex voice conversation WebSocket.

        This endpoint enables bidirectional audio streaming with PersonaPlex:
        - User audio (PCM) flows browser -> Cathedral -> PersonaPlex
        - Agent audio (PCM) flows PersonaPlex -> Cathedral -> browser
        - Interrupts are detected when user speaks during agent response

        Protocol (client -> server):
        - Binary: Raw PCM audio (16-bit, mono, 24kHz)
        - Text JSON: Control messages
            {"type": "interrupt"} - Signal user interrupt
            {"type": "config", "voice": "NATF2"} - Configuration

        Protocol (server -> client):
        - Binary: Raw PCM audio (16-bit, mono, 24kHz)
        - Text JSON: Events
            {"type": "event", "event": {...}} - Voice event envelope
            {"type": "audio_meta", "generation_id": "...", "is_final": bool}
            {"type": "turn", "source": "user"|"agent", "state": "start"|"end"}
            {"type": "transcript", "text": "...", "source": "user"|"agent"}
            {"type": "error", "message": "..."}
        """
        await websocket.accept()

        session_id = str(uuid.uuid4())[:8]
        if emit_event:
            await emit_event("voice", f"Voice conversation started: {session_id}")

        # Check availability
        VoiceGate.initialize()
        if not VoiceGate.is_available():
            await websocket.send_json({
                "type": "error",
                "message": "Voice synthesis not available",
                "reason": VoiceGate.get_status().error or "No GPU or TTS disabled",
            })
            await websocket.close(code=4000, reason="Voice not available")
            return

        # Check Opus codec
        if not is_opus_available():
            await websocket.send_json({
                "type": "error",
                "message": "Opus codec not available",
                "reason": "opuslib not installed",
            })
            await websocket.close(code=4001, reason="Opus not available")
            return

        # Get PersonaPlex URL from environment
        import os
        personaplex_url = os.getenv("PERSONAPLEX_URL", "wss://host.docker.internal:8998/api/chat")

        # Create components
        bridge_config = BridgeConfig(
            personaplex_url=personaplex_url,
            voice_prompt=os.getenv("PERSONAPLEX_VOICE", "NATF2"),
        )
        bridge = VoiceBridge(bridge_config)
        turn_manager = TurnManager(
            config=TurnManagerConfig(),
            on_interrupt=lambda gen_id: asyncio.create_task(
                _handle_interrupt(websocket, bridge, gen_id)
            ),
        )
        transcript_writer = TranscriptWriter(
            thread_uid=thread_uid,
            on_memory_event=lambda text: asyncio.create_task(
                _store_to_memory(thread_uid, text)
            ),
        )

        # Event handler - forwards events to browser
        async def on_voice_event(event: VoiceEvent):
            # Write to transcript (handles commit level filtering)
            await transcript_writer.write(event)

            # Forward to browser
            try:
                await websocket.send_json({
                    "type": "event",
                    "event": {
                        "event_id": event.event_id,
                        "turn_id": event.turn_id,
                        "generation_id": event.generation_id,
                        "source": event.source,
                        "channel": event.channel.value,
                        "event_type": event.type.value,
                        "commit": event.commit.value,
                        "ts": event.ts,
                        "payload": event.payload,
                    }
                })
            except Exception:
                pass  # WebSocket may be closed

        bridge.on_event = on_voice_event

        # Connect to PersonaPlex
        connected = await bridge.connect()
        if not connected:
            await websocket.send_json({
                "type": "error",
                "message": "Failed to connect to PersonaPlex",
            })
            await websocket.close(code=4002, reason="PersonaPlex connection failed")
            return

        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "opus_available": True,
        })

        # Tasks for bidirectional streaming
        receive_task = None
        send_task = None

        try:
            # Task: Receive from browser, send to PersonaPlex
            async def receive_from_browser():
                try:
                    while True:
                        message = await websocket.receive()

                        if message["type"] == "websocket.disconnect":
                            break

                        if "bytes" in message:
                            # User audio - send to PersonaPlex
                            pcm_data = message["bytes"]
                            await bridge.send_audio(pcm_data)

                            # Signal to turn manager
                            event = await turn_manager.user_audio_received()
                            if event:
                                await on_voice_event(event)

                        elif "text" in message:
                            # Control message
                            try:
                                data = json.loads(message["text"])
                                msg_type = data.get("type", "")

                                if msg_type == "interrupt":
                                    cancelled_id = await bridge.send_interrupt()
                                    if cancelled_id:
                                        await websocket.send_json({
                                            "type": "interrupted",
                                            "cancelled_generation_id": cancelled_id,
                                        })

                                elif msg_type == "silence":
                                    # User stopped speaking
                                    event = await turn_manager.user_silence_detected()
                                    if event:
                                        await on_voice_event(event)

                                elif msg_type == "config":
                                    # Handle configuration changes
                                    if "voice" in data:
                                        await VoiceGate.set_voice(data["voice"])

                            except json.JSONDecodeError:
                                pass

                except WebSocketDisconnect:
                    pass
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if emit_event:
                        await emit_event("voice", f"Receive error: {e}")

            # Task: Receive from PersonaPlex, send to browser
            async def send_to_browser():
                try:
                    async for chunk in bridge.stream_audio():
                        if chunk:
                            # Send audio metadata
                            await websocket.send_json({
                                "type": "audio_meta",
                                "generation_id": bridge.current_generation_id,
                                "is_final": False,
                            })
                            # Send audio data
                            await websocket.send_bytes(chunk)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if emit_event:
                        await emit_event("voice", f"Send error: {e}")

            # Run both tasks
            receive_task = asyncio.create_task(receive_from_browser())
            send_task = asyncio.create_task(send_to_browser())

            # Wait for either to complete
            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            if emit_event:
                await emit_event("voice", f"Conversation error: {e}")

        finally:
            # Cleanup
            if receive_task and not receive_task.done():
                receive_task.cancel()
            if send_task and not send_task.done():
                send_task.cancel()

            await bridge.disconnect()

            if emit_event:
                await emit_event("voice", f"Voice conversation ended: {session_id}")

    @router.get("/codec/info")
    async def get_codec_info():
        """Get Opus codec availability and configuration."""
        from cathedral.VoiceGate import get_codec_info as _get_codec_info
        return JSONResponse(content=_get_codec_info())

    @router.post("/synthesize")
    async def synthesize_text(request: dict):
        """
        Synthesize text to audio (non-streaming).

        Request body:
        {
            "text": "Text to synthesize",
            "sample_rate": 24000  (optional)
        }

        Returns base64-encoded PCM audio.
        """
        VoiceGate.initialize()

        if not VoiceGate.is_available():
            raise HTTPException(
                status_code=503,
                detail="Voice synthesis not available"
            )

        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Collect all audio chunks
        audio_chunks = []
        async for chunk in VoiceGate.synthesize_sentence(text):
            audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)

        return JSONResponse(content={
            "audio": base64.b64encode(audio_data).decode("ascii"),
            "sample_rate": VoiceGate._get_config().sample_rate,
            "format": "pcm_s16le",
            "duration_ms": len(audio_data) // 48,  # Rough estimate
        })

    return router


async def _handle_interrupt(websocket: WebSocket, bridge: VoiceBridge, generation_id: str):
    """Handle a voice interrupt by notifying the browser."""
    try:
        await websocket.send_json({
            "type": "interrupted",
            "cancelled_generation_id": generation_id,
        })
    except Exception:
        pass  # WebSocket may be closed


async def _store_to_memory(thread_uid: str, text: str):
    """Store committed voice event to MemoryGate."""
    try:
        from cathedral import MemoryGate
        if MemoryGate.is_initialized():
            # Store as observation with voice domain
            # Run in executor since store_observation is sync
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: MemoryGate.store_observation(
                    text=text,
                    domain="voice",
                    evidence=[f"thread:{thread_uid}"],
                ),
            )
    except Exception:
        pass  # Don't fail the voice stream on memory errors


__all__ = ["create_router"]
