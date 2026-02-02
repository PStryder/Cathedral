"""
Transcript Writer - Persists voice events with commit-level filtering.

Handles:
- Writing committed events to conversation thread
- Filtering events for MemoryGate ingestion
- Audit trail for tools and interrupts
- UI-friendly event formatting

Commit levels:
- EPHEMERAL: Not stored (partials, backchannels, audio chunks)
- COMMITTED: Stored in MemoryGate (final utterances, turns)
- AUDIT: Stored but hidden (tool calls, interrupts)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any

from cathedral.shared.gate import GateLogger
from .events import (
    VoiceEvent, Channel, CommitLevel, EventType,
)

_log = GateLogger.get("VoiceGate.TranscriptWriter")


@dataclass
class TranscriptEntry:
    """An entry in the transcript."""
    event: VoiceEvent
    formatted_text: Optional[str] = None  # UI-friendly format
    memory_text: Optional[str] = None     # MemoryGate format


class TranscriptWriter:
    """
    Writes voice events to conversation transcript.

    Filters events by commit level:
    - EPHEMERAL: Discarded (not stored)
    - COMMITTED: Stored, visible in UI, ingested by MemoryGate
    - AUDIT: Stored, collapsed in UI, not in MemoryGate
    """

    def __init__(
        self,
        thread_uid: str,
        on_transcript_update: Optional[Callable[[TranscriptEntry], None]] = None,
        on_memory_event: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize transcript writer.

        Args:
            thread_uid: The conversation thread ID
            on_transcript_update: Callback when transcript is updated
            on_memory_event: Callback for MemoryGate-worthy events
        """
        self.thread_uid = thread_uid
        self.on_transcript_update = on_transcript_update
        self.on_memory_event = on_memory_event

        # Storage
        self._entries: List[TranscriptEntry] = []
        self._pending_user_text: str = ""
        self._pending_agent_text: str = ""

        # Stats
        self._events_received = 0
        self._events_committed = 0
        self._events_audited = 0
        self._events_discarded = 0

    async def write(self, event: VoiceEvent) -> Optional[TranscriptEntry]:
        """
        Process and potentially store an event.

        Args:
            event: The voice event to process

        Returns:
            TranscriptEntry if event was stored, None if discarded
        """
        self._events_received += 1

        # Filter by commit level
        if event.is_ephemeral():
            self._events_discarded += 1
            # But still accumulate partial transcripts
            if event.type == EventType.PARTIAL_TRANSCRIPT:
                self._pending_user_text = event.payload.get("text", "")
            return None

        # Create transcript entry
        entry = TranscriptEntry(
            event=event,
            formatted_text=self._format_for_ui(event),
            memory_text=self._format_for_memory(event) if event.should_store_in_memory() else None,
        )

        # Store entry
        self._entries.append(entry)

        if event.commit == CommitLevel.COMMITTED:
            self._events_committed += 1
        else:  # AUDIT
            self._events_audited += 1

        # Callbacks
        if self.on_transcript_update:
            self.on_transcript_update(entry)

        if entry.memory_text and self.on_memory_event:
            self.on_memory_event(entry.memory_text)

        # Reset pending text on final events
        if event.type == EventType.FINAL_TRANSCRIPT:
            self._pending_user_text = ""
        elif event.type == EventType.SPEECH_COMPLETE:
            self._pending_agent_text = ""

        return entry

    def _format_for_ui(self, event: VoiceEvent) -> str:
        """Format event for UI display."""
        if event.type == EventType.FINAL_TRANSCRIPT:
            return f"**User**: {event.payload.get('text', '')}"

        elif event.type == EventType.SPEECH_COMPLETE:
            return f"**Assistant**: {event.payload.get('text', '')}"

        elif event.type == EventType.TOOL_CALL:
            tool = event.payload.get("tool", "unknown")
            return f"_Used tool: {tool}_"

        elif event.type == EventType.TOOL_RESULT:
            tool = event.payload.get("tool", "unknown")
            return f"_Tool {tool} completed_"

        elif event.type == EventType.INTERRUPT:
            return "_[interrupted]_"

        elif event.type == EventType.TURN_START:
            source = event.source
            return f"_[{source} turn started]_"

        elif event.type == EventType.TURN_END:
            source = event.source
            return f"_[{source} turn ended]_"

        elif event.type == EventType.ERROR:
            return f"_Error: {event.payload.get('message', 'unknown')}_"

        return f"_[{event.type.value}]_"

    def _format_for_memory(self, event: VoiceEvent) -> Optional[str]:
        """Format event for MemoryGate ingestion."""
        # Only COMMITTED events go to memory
        if event.commit != CommitLevel.COMMITTED:
            return None

        if event.type == EventType.FINAL_TRANSCRIPT:
            text = event.payload.get("text", "")
            if text:
                return f"User said: {text}"

        elif event.type == EventType.SPEECH_COMPLETE:
            text = event.payload.get("text", "")
            if text:
                return f"Assistant responded: {text}"

        # Turn boundaries don't need memory entries
        return None

    def get_transcript(
        self,
        include_audit: bool = True,
        limit: Optional[int] = None,
    ) -> List[TranscriptEntry]:
        """
        Get transcript entries.

        Args:
            include_audit: Whether to include audit entries
            limit: Max entries to return (most recent)

        Returns:
            List of transcript entries
        """
        entries = self._entries

        if not include_audit:
            entries = [e for e in entries if e.event.commit == CommitLevel.COMMITTED]

        if limit:
            entries = entries[-limit:]

        return entries

    def get_formatted_transcript(
        self,
        include_audit: bool = False,
        separator: str = "\n",
    ) -> str:
        """
        Get transcript as formatted string.

        Args:
            include_audit: Whether to include audit entries
            separator: Line separator

        Returns:
            Formatted transcript string
        """
        entries = self.get_transcript(include_audit=include_audit)
        lines = [e.formatted_text for e in entries if e.formatted_text]
        return separator.join(lines)

    def get_memory_events(self) -> List[str]:
        """
        Get all events suitable for MemoryGate.

        Returns:
            List of memory-formatted strings
        """
        return [
            e.memory_text
            for e in self._entries
            if e.memory_text
        ]

    def get_stats(self) -> Dict[str, int]:
        """Get transcript writer statistics."""
        return {
            "received": self._events_received,
            "committed": self._events_committed,
            "audited": self._events_audited,
            "discarded": self._events_discarded,
            "total_stored": len(self._entries),
        }

    def clear(self):
        """Clear all transcript entries."""
        self._entries.clear()
        self._pending_user_text = ""
        self._pending_agent_text = ""


class TranscriptManager:
    """
    Manages transcript writers for multiple threads.
    """

    def __init__(self):
        self._writers: Dict[str, TranscriptWriter] = {}

    def get_writer(
        self,
        thread_uid: str,
        on_transcript_update: Optional[Callable[[TranscriptEntry], None]] = None,
        on_memory_event: Optional[Callable[[str], None]] = None,
    ) -> TranscriptWriter:
        """
        Get or create a transcript writer for a thread.

        Args:
            thread_uid: The thread ID
            on_transcript_update: Callback for transcript updates
            on_memory_event: Callback for memory events

        Returns:
            TranscriptWriter instance
        """
        if thread_uid not in self._writers:
            self._writers[thread_uid] = TranscriptWriter(
                thread_uid=thread_uid,
                on_transcript_update=on_transcript_update,
                on_memory_event=on_memory_event,
            )
        return self._writers[thread_uid]

    def remove_writer(self, thread_uid: str) -> bool:
        """Remove a transcript writer."""
        if thread_uid in self._writers:
            del self._writers[thread_uid]
            return True
        return False

    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """Get stats for all writers."""
        return {
            thread_uid: writer.get_stats()
            for thread_uid, writer in self._writers.items()
        }


# Global manager instance
_transcript_manager: Optional[TranscriptManager] = None


def get_transcript_manager() -> TranscriptManager:
    """Get the global transcript manager."""
    global _transcript_manager
    if _transcript_manager is None:
        _transcript_manager = TranscriptManager()
    return _transcript_manager


__all__ = [
    "TranscriptEntry",
    "TranscriptWriter",
    "TranscriptManager",
    "get_transcript_manager",
]
