"""
Audio queue for VoiceGate.

Manages queuing and streaming of audio chunks for gapless playback.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field
from collections import deque

from cathedral.shared.gate import GateLogger
from .models import AudioChunk

_log = GateLogger.get("VoiceGate.AudioQueue")


@dataclass
class QueueStats:
    """Statistics about the audio queue."""
    total_chunks: int = 0
    total_bytes: int = 0
    total_duration_ms: int = 0
    dropped_chunks: int = 0


class AudioQueue:
    """
    Queue for audio chunks with gapless playback support.

    Manages a queue of audio chunks and provides async iteration
    for streaming to clients.
    """

    # Maximum queue size to prevent memory issues
    MAX_QUEUE_SIZE = 100

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the audio queue.

        Args:
            sample_rate: Sample rate for duration calculations
        """
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._sequence = 0
        self._stats = QueueStats()
        self._closed = False

    async def enqueue(
        self,
        audio_data: bytes,
        is_final: bool = False,
    ) -> bool:
        """
        Add an audio chunk to the queue.

        Args:
            audio_data: Raw PCM audio bytes
            is_final: Whether this is the final chunk

        Returns:
            True if queued successfully, False if queue full
        """
        if self._closed:
            _log.warning("Cannot enqueue to closed queue")
            return False

        # Calculate duration (16-bit mono)
        bytes_per_sample = 2
        num_samples = len(audio_data) // bytes_per_sample
        duration_ms = int((num_samples / self.sample_rate) * 1000)

        chunk = AudioChunk(
            sequence=self._sequence,
            data=audio_data,
            is_final=is_final,
            duration_ms=duration_ms,
            sample_rate=self.sample_rate,
        )

        try:
            # Non-blocking put with timeout
            await asyncio.wait_for(
                self._queue.put(chunk),
                timeout=1.0,
            )

            self._sequence += 1
            self._stats.total_chunks += 1
            self._stats.total_bytes += len(audio_data)
            self._stats.total_duration_ms += duration_ms

            return True

        except asyncio.TimeoutError:
            self._stats.dropped_chunks += 1
            _log.warning(f"Queue full, dropped chunk {self._sequence}")
            return False

    async def stream(self) -> AsyncGenerator[AudioChunk, None]:
        """
        Yield audio chunks as they become available.

        Continues until a chunk with is_final=True is received
        or the queue is closed.
        """
        while not self._closed:
            try:
                chunk = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=5.0,  # Timeout to check for close
                )

                yield chunk

                if chunk.is_final:
                    break

            except asyncio.TimeoutError:
                # Check if we should continue waiting
                if self._closed:
                    break
                continue

    def close(self):
        """Close the queue, stopping any consumers."""
        self._closed = True

    def reset(self):
        """Reset the queue for reuse."""
        # Clear any pending items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._sequence = 0
        self._stats = QueueStats()
        self._closed = False

    @property
    def pending_count(self) -> int:
        """Number of chunks waiting in queue."""
        return self._queue.qsize()

    @property
    def stats(self) -> QueueStats:
        """Get queue statistics."""
        return self._stats

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def is_closed(self) -> bool:
        """Check if queue is closed."""
        return self._closed


class AudioQueueManager:
    """
    Manages per-thread audio queues.

    Provides queue allocation and cleanup for multiple
    concurrent voice streams.
    """

    def __init__(self):
        self._queues: dict[str, AudioQueue] = {}
        self._lock = asyncio.Lock()

    async def get_queue(self, thread_uid: str, sample_rate: int = 24000) -> AudioQueue:
        """
        Get or create an audio queue for a thread.

        Args:
            thread_uid: Thread identifier
            sample_rate: Sample rate for the queue

        Returns:
            AudioQueue for the thread
        """
        async with self._lock:
            if thread_uid not in self._queues:
                self._queues[thread_uid] = AudioQueue(sample_rate=sample_rate)
            else:
                # Reset existing queue for new stream
                self._queues[thread_uid].reset()

            return self._queues[thread_uid]

    async def close_queue(self, thread_uid: str):
        """Close and remove a thread's queue."""
        async with self._lock:
            if thread_uid in self._queues:
                self._queues[thread_uid].close()
                del self._queues[thread_uid]

    async def close_all(self):
        """Close all queues."""
        async with self._lock:
            for queue in self._queues.values():
                queue.close()
            self._queues.clear()

    @property
    def active_count(self) -> int:
        """Number of active queues."""
        return len(self._queues)


# Global queue manager
_queue_manager: Optional[AudioQueueManager] = None


def get_queue_manager() -> AudioQueueManager:
    """Get the global queue manager, creating if needed."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = AudioQueueManager()
    return _queue_manager


__all__ = [
    "AudioQueue",
    "AudioQueueManager",
    "QueueStats",
    "get_queue_manager",
]
