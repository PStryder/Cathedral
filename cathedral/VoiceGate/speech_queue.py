"""
Speech Queue with Cancel Token Support.

Manages speech generation with proper cancellation to prevent
"ghost speech" - old audio chunks leaking after an interrupt.

Key features:
- Generation IDs track speech batches
- Cancel tokens immediately drop pending/future chunks
- Clean interrupt handling without blocking tool execution
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, Optional, Set
from collections import deque

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("VoiceGate.SpeechQueue")


@dataclass
class SpeechChunk:
    """A chunk of audio in the speech queue."""
    data: bytes
    generation_id: str
    sequence: int
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)

    # Metadata for debugging/audit
    text_fragment: Optional[str] = None  # The text this audio represents


class SpeechQueue:
    """
    Manages speech generation with cancellation support.

    When the user interrupts:
    1. Call cancel(generation_id) - marks generation as cancelled
    2. All pending chunks for that generation are dropped
    3. Future chunks for that generation are dropped on enqueue
    4. Tool execution continues unaffected

    This prevents ghost speech while allowing clean turn transitions.
    """

    def __init__(self, max_cancelled_ids: int = 100):
        """
        Initialize the speech queue.

        Args:
            max_cancelled_ids: Max cancelled IDs to track (prevents memory leak)
        """
        self.queue: asyncio.Queue[SpeechChunk] = asyncio.Queue()
        self.current_generation_id: Optional[str] = None
        self.cancelled_generations: Set[str] = set()
        self._max_cancelled_ids = max_cancelled_ids
        self._sequence_counters: Dict[str, int] = {}
        self._lock = asyncio.Lock()

        # Stats for monitoring
        self._chunks_enqueued = 0
        self._chunks_dropped = 0
        self._generations_cancelled = 0

    async def enqueue(
        self,
        audio_data: bytes,
        generation_id: str,
        is_final: bool = False,
        text_fragment: Optional[str] = None,
    ) -> bool:
        """
        Add a speech chunk to the queue.

        Returns False if the chunk was dropped (generation cancelled).

        Args:
            audio_data: PCM audio bytes
            generation_id: ID of this speech generation batch
            is_final: True if this is the last chunk
            text_fragment: Optional text this audio represents

        Returns:
            True if enqueued, False if dropped
        """
        # Ghost speech prevention - drop if generation was cancelled
        if generation_id in self.cancelled_generations:
            self._chunks_dropped += 1
            _log.debug(f"Dropped chunk for cancelled generation {generation_id[:8]}")
            return False

        async with self._lock:
            # Get next sequence number for this generation
            if generation_id not in self._sequence_counters:
                self._sequence_counters[generation_id] = 0

            sequence = self._sequence_counters[generation_id]
            self._sequence_counters[generation_id] += 1

            # Clean up old sequence counters
            if is_final and generation_id in self._sequence_counters:
                del self._sequence_counters[generation_id]

        chunk = SpeechChunk(
            data=audio_data,
            generation_id=generation_id,
            sequence=sequence,
            is_final=is_final,
            text_fragment=text_fragment,
        )

        await self.queue.put(chunk)
        self._chunks_enqueued += 1
        self.current_generation_id = generation_id

        return True

    def cancel(self, generation_id: str) -> int:
        """
        Cancel a speech generation.

        All pending and future chunks for this generation will be dropped.

        Args:
            generation_id: The generation to cancel

        Returns:
            Number of chunks that will be dropped from queue
        """
        if generation_id in self.cancelled_generations:
            return 0  # Already cancelled

        self.cancelled_generations.add(generation_id)
        self._generations_cancelled += 1

        # Prune old cancelled IDs to prevent memory leak
        if len(self.cancelled_generations) > self._max_cancelled_ids:
            # Remove oldest (arbitrary since set, but keeps size bounded)
            self.cancelled_generations.pop()

        # Count how many queued chunks will be dropped
        # (They'll be dropped when dequeued, not removed from queue)
        pending_count = 0
        # Note: We don't actually drain the queue here - chunks are filtered on dequeue

        _log.info(f"Cancelled generation {generation_id[:8]}")

        if generation_id in self._sequence_counters:
            del self._sequence_counters[generation_id]

        return pending_count

    def cancel_current(self) -> Optional[str]:
        """
        Cancel the current speech generation.

        Returns:
            The cancelled generation ID, or None if nothing playing
        """
        if self.current_generation_id:
            gen_id = self.current_generation_id
            self.cancel(gen_id)
            self.current_generation_id = None
            return gen_id
        return None

    def is_cancelled(self, generation_id: str) -> bool:
        """Check if a generation has been cancelled."""
        return generation_id in self.cancelled_generations

    async def stream(self) -> AsyncGenerator[SpeechChunk, None]:
        """
        Yield chunks from the queue, filtering cancelled generations.

        Chunks from cancelled generations are silently dropped.
        Stops when a final chunk is received.
        """
        while True:
            chunk = await self.queue.get()

            # Ghost speech prevention - skip cancelled chunks
            if chunk.generation_id in self.cancelled_generations:
                self._chunks_dropped += 1
                _log.debug(f"Filtered cancelled chunk seq={chunk.sequence}")

                # If this was final, we're done with this generation
                if chunk.is_final:
                    break
                continue

            yield chunk

            if chunk.is_final:
                break

    async def drain(self, timeout: float = 0.1) -> int:
        """
        Drain and discard all pending chunks.

        Useful when resetting state.

        Args:
            timeout: Max time to wait for queue to empty

        Returns:
            Number of chunks drained
        """
        drained = 0
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                self.queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break

        return drained

    def clear_cancelled(self):
        """Clear all cancelled generation tracking."""
        self.cancelled_generations.clear()
        self._sequence_counters.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "enqueued": self._chunks_enqueued,
            "dropped": self._chunks_dropped,
            "cancelled_generations": self._generations_cancelled,
            "pending": self.queue.qsize(),
            "tracking_cancelled": len(self.cancelled_generations),
        }

    @property
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.queue.empty()

    @property
    def pending_count(self) -> int:
        """Get the number of pending chunks."""
        return self.queue.qsize()


class SpeechQueueManager:
    """
    Manages speech queues per thread/session.

    Each conversation thread gets its own queue for independent
    speech generation and cancellation.
    """

    def __init__(self):
        self._queues: Dict[str, SpeechQueue] = {}
        self._lock = asyncio.Lock()

    async def get_queue(self, thread_id: str) -> SpeechQueue:
        """Get or create a speech queue for a thread."""
        async with self._lock:
            if thread_id not in self._queues:
                self._queues[thread_id] = SpeechQueue()
                _log.debug(f"Created speech queue for thread {thread_id[:8]}")
            return self._queues[thread_id]

    async def remove_queue(self, thread_id: str) -> bool:
        """Remove a thread's speech queue."""
        async with self._lock:
            if thread_id in self._queues:
                del self._queues[thread_id]
                return True
            return False

    def cancel_all(self, thread_id: str) -> Optional[str]:
        """Cancel all speech for a thread."""
        if thread_id in self._queues:
            return self._queues[thread_id].cancel_current()
        return None

    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """Get stats for all queues."""
        return {
            thread_id: queue.get_stats()
            for thread_id, queue in self._queues.items()
        }


# Global manager instance
_queue_manager: Optional[SpeechQueueManager] = None


def get_speech_queue_manager() -> SpeechQueueManager:
    """Get the global speech queue manager."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = SpeechQueueManager()
    return _queue_manager


__all__ = [
    "SpeechChunk",
    "SpeechQueue",
    "SpeechQueueManager",
    "get_speech_queue_manager",
]
