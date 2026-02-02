"""
Qwen3-TTS synthesis backend.

Provides local TTS using the Qwen3-TTS model family.
Supports lazy model loading and async synthesis.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional
from concurrent.futures import ThreadPoolExecutor

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("VoiceGate.Qwen3")

# Thread pool for blocking TTS operations
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts_")


class Qwen3Synthesizer:
    """
    Qwen3-TTS synthesis backend.

    Lazily loads the model on first use to avoid startup overhead.
    All synthesis runs in a thread pool to avoid blocking async code.
    """

    # Model variants available
    MODELS = {
        "Qwen3-TTS-0.6B": "Qwen/Qwen3-TTS-0.6B",
        "Qwen3-TTS-1.7B": "Qwen/Qwen3-TTS-1.7B",
    }

    def __init__(self, model_variant: str = "Qwen3-TTS-0.6B", sample_rate: int = 24000):
        """
        Initialize the synthesizer.

        Args:
            model_variant: Model variant to use (0.6B or 1.7B)
            sample_rate: Output sample rate (default 24000)
        """
        self.model_variant = model_variant
        self.model_id = self.MODELS.get(model_variant, self.MODELS["Qwen3-TTS-0.6B"])
        self.sample_rate = sample_rate

        # Lazy loaded
        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False
        self._load_error: Optional[str] = None

    def is_available(self) -> bool:
        """Check if GPU is available for TTS."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def device(self) -> Optional[str]:
        """Get the device to use."""
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else None
            except ImportError:
                self._device = None
        return self._device

    def _load_model_sync(self) -> bool:
        """
        Load the model synchronously (runs in thread pool).

        Returns:
            True if model loaded successfully
        """
        if self._loaded:
            return True

        if not self.is_available():
            self._load_error = "No GPU available"
            _log.warning(self._load_error)
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _log.info(f"Loading {self.model_id}...")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            self._loaded = True
            _log.info(f"Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            self._load_error = str(e)
            _log.error(f"Failed to load model: {e}")
            return False

    async def load_model(self) -> bool:
        """
        Load the model asynchronously.

        Returns:
            True if model loaded successfully
        """
        if self._loaded:
            return True

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._load_model_sync)

    def _synthesize_sync(self, text: str) -> bytes:
        """
        Synthesize text to audio synchronously.

        Args:
            text: Text to synthesize

        Returns:
            PCM audio bytes
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        try:
            import torch
            import numpy as np

            # Prepare input
            # Note: Qwen3-TTS uses a specific prompt format
            prompt = f"<|text|>{text}<|audio|>"

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.device)

            # Generate audio tokens
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                )

            # Decode audio tokens to waveform
            # The actual decoding depends on Qwen3-TTS architecture
            # This is a placeholder - real implementation needs model-specific code
            audio_tokens = outputs[0, inputs.input_ids.shape[1]:]

            # Convert to audio (model-specific)
            # For now, return empty bytes as placeholder
            # Real implementation would use model's audio decoder
            if hasattr(self._model, "decode_audio"):
                audio = self._model.decode_audio(audio_tokens)
                audio_np = audio.cpu().numpy()
            else:
                # Fallback: synthesize silence (for testing)
                duration_samples = int(len(text) * 0.1 * self.sample_rate)
                audio_np = np.zeros(duration_samples, dtype=np.float32)

            # Convert to 16-bit PCM
            audio_int16 = (audio_np * 32767).astype(np.int16)
            return audio_int16.tobytes()

        except Exception as e:
            _log.error(f"Synthesis failed: {e}")
            raise

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to audio, yielding chunks.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as bytes
        """
        if not self.is_available():
            _log.debug("TTS not available, skipping synthesis")
            return

        # Ensure model is loaded
        if not await self.load_model():
            _log.warning("Failed to load model, skipping synthesis")
            return

        try:
            # Run synthesis in thread pool
            loop = asyncio.get_running_loop()
            audio_bytes = await loop.run_in_executor(
                _executor,
                self._synthesize_sync,
                text,
            )

            # Yield in chunks for streaming
            chunk_size = 4096  # ~85ms at 24kHz 16-bit mono
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]

        except Exception as e:
            _log.error(f"Synthesis error: {e}")
            return

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Synthesize text to a single audio buffer.

        Args:
            text: Text to synthesize

        Returns:
            Complete audio as bytes
        """
        chunks = []
        async for chunk in self.synthesize(text):
            chunks.append(chunk)
        return b"".join(chunks)

    def get_info(self) -> dict:
        """Get synthesizer information."""
        return {
            "backend": "qwen3",
            "model_variant": self.model_variant,
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "loaded": self._loaded,
            "available": self.is_available(),
            "error": self._load_error,
        }

    def unload(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._loaded = False
        _log.info("Model unloaded")

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


__all__ = ["Qwen3Synthesizer"]
