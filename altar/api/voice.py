"""
Voice API endpoints for Cathedral.

Provides WebSocket endpoint for streaming TTS audio
and HTTP endpoints for voice status.
"""

import asyncio
import base64
import json
from typing import Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from cathedral import VoiceGate


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


__all__ = ["create_router"]
