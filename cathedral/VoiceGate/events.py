"""
VoiceGate Event System.

Defines the canonical event envelope for all voice interactions.
Events flow through three channels (audio, transcript, tool) with
three commit levels (ephemeral, committed, audit).

This separation prevents:
- Tool calls being spoken aloud
- Partial transcripts polluting MemoryGate
- Ghost speech after interruptions
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Literal, Optional

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("VoiceGate.Events")


class Channel(str, Enum):
    """Event channels - determines routing."""
    AUDIO = "audio"           # Real-time, embodied, interruptible
    TRANSCRIPT = "transcript"  # Durable truth for humans + MemoryGate
    TOOL = "tool"             # Silent execution


class CommitLevel(str, Enum):
    """Commit levels - determines persistence."""
    EPHEMERAL = "ephemeral"   # Not stored (partials, backchannels, aborted)
    COMMITTED = "committed"   # Stored in MemoryGate (final utterances)
    AUDIT = "audit"           # Stored but hidden (tool calls, interrupts)


class EventType(str, Enum):
    """Event types for the voice system."""
    # Transcript events
    PARTIAL_TRANSCRIPT = "partial_transcript"  # ephemeral
    FINAL_TRANSCRIPT = "final_transcript"      # committed
    TEXT_TOKEN = "text_token"                  # ephemeral (streaming text)

    # Speech events
    SPEECH_CHUNK = "speech_chunk"              # ephemeral (audio frame)
    SPEECH_COMPLETE = "speech_complete"        # committed

    # Tool events
    TOOL_CALL = "tool_call"                    # audit
    TOOL_RESULT = "tool_result"                # audit

    # Control events
    TURN_START = "turn_start"                  # committed
    TURN_END = "turn_end"                      # committed
    INTERRUPT = "interrupt"                    # audit
    CANCEL = "cancel"                          # audit

    # Backchannel
    BACKCHANNEL = "backchannel"                # ephemeral

    # System
    ERROR = "error"                            # audit
    VAD = "vad"                                # ephemeral (voice activity)


# Default commit levels for each event type
DEFAULT_COMMIT_LEVELS: Dict[EventType, CommitLevel] = {
    EventType.PARTIAL_TRANSCRIPT: CommitLevel.EPHEMERAL,
    EventType.FINAL_TRANSCRIPT: CommitLevel.COMMITTED,
    EventType.SPEECH_CHUNK: CommitLevel.EPHEMERAL,
    EventType.SPEECH_COMPLETE: CommitLevel.COMMITTED,
    EventType.TOOL_CALL: CommitLevel.AUDIT,
    EventType.TOOL_RESULT: CommitLevel.AUDIT,
    EventType.TURN_START: CommitLevel.COMMITTED,
    EventType.TURN_END: CommitLevel.COMMITTED,
    EventType.INTERRUPT: CommitLevel.AUDIT,
    EventType.CANCEL: CommitLevel.AUDIT,
    EventType.BACKCHANNEL: CommitLevel.EPHEMERAL,
    EventType.ERROR: CommitLevel.AUDIT,
    EventType.VAD: CommitLevel.EPHEMERAL,
}


@dataclass
class VoiceEvent:
    """
    Canonical event envelope for all voice interactions.

    Every event flows through this structure, enabling:
    - Deterministic UI rendering
    - Trivial MemoryGate filtering
    - Replay and debugging
    - Safe streaming
    """
    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: str = ""                    # Groups related events in a turn
    generation_id: str = ""              # For cancel tokens (speech generation)

    # Routing
    source: Literal["user", "agent", "tool", "system"] = "system"
    channel: Channel = Channel.TRANSCRIPT
    type: EventType = EventType.PARTIAL_TRANSCRIPT
    commit: CommitLevel = CommitLevel.EPHEMERAL

    # Timing
    ts: float = field(default_factory=time.time)

    # Content
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Apply default commit level if not set."""
        if self.commit == CommitLevel.EPHEMERAL and self.type in DEFAULT_COMMIT_LEVELS:
            # Only override if it was the default value
            expected = DEFAULT_COMMIT_LEVELS.get(self.type)
            if expected and expected != CommitLevel.EPHEMERAL:
                self.commit = expected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "source": self.source,
            "channel": self.channel.value,
            "type": self.type.value,
            "commit": self.commit.value,
            "ts": self.ts,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceEvent":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            turn_id=data.get("turn_id", ""),
            generation_id=data.get("generation_id", ""),
            source=data.get("source", "system"),
            channel=Channel(data.get("channel", "transcript")),
            type=EventType(data.get("type", "partial_transcript")),
            commit=CommitLevel(data.get("commit", "ephemeral")),
            ts=data.get("ts", time.time()),
            payload=data.get("payload", {}),
        )

    def should_store_in_memory(self) -> bool:
        """Check if this event should be ingested by MemoryGate."""
        return self.commit == CommitLevel.COMMITTED

    def should_store_in_transcript(self) -> bool:
        """Check if this event should appear in the conversation transcript."""
        return self.commit in (CommitLevel.COMMITTED, CommitLevel.AUDIT)

    def is_ephemeral(self) -> bool:
        """Check if this event is ephemeral (not persisted)."""
        return self.commit == CommitLevel.EPHEMERAL


# Factory functions for common events

def user_turn_start(turn_id: str) -> VoiceEvent:
    """Create a user turn start event."""
    return VoiceEvent(
        turn_id=turn_id,
        source="user",
        channel=Channel.TRANSCRIPT,
        type=EventType.TURN_START,
        commit=CommitLevel.COMMITTED,
    )


def user_utterance(turn_id: str, text: str, is_final: bool = True) -> VoiceEvent:
    """Create a user utterance event."""
    return VoiceEvent(
        turn_id=turn_id,
        source="user",
        channel=Channel.TRANSCRIPT,
        type=EventType.FINAL_TRANSCRIPT if is_final else EventType.PARTIAL_TRANSCRIPT,
        commit=CommitLevel.COMMITTED if is_final else CommitLevel.EPHEMERAL,
        payload={"text": text},
    )


def agent_turn_start(turn_id: str, generation_id: str) -> VoiceEvent:
    """Create an agent turn start event."""
    return VoiceEvent(
        turn_id=turn_id,
        generation_id=generation_id,
        source="agent",
        channel=Channel.TRANSCRIPT,
        type=EventType.TURN_START,
        commit=CommitLevel.COMMITTED,
    )


def agent_speech_chunk(turn_id: str, generation_id: str, audio_data: bytes) -> VoiceEvent:
    """Create an agent speech chunk event."""
    return VoiceEvent(
        turn_id=turn_id,
        generation_id=generation_id,
        source="agent",
        channel=Channel.AUDIO,
        type=EventType.SPEECH_CHUNK,
        commit=CommitLevel.EPHEMERAL,
        payload={"audio": audio_data},
    )


def agent_speech_complete(turn_id: str, generation_id: str, text: str) -> VoiceEvent:
    """Create an agent speech complete event."""
    return VoiceEvent(
        turn_id=turn_id,
        generation_id=generation_id,
        source="agent",
        channel=Channel.TRANSCRIPT,
        type=EventType.SPEECH_COMPLETE,
        commit=CommitLevel.COMMITTED,
        payload={"text": text},
    )


def tool_call_event(turn_id: str, tool_name: str, tool_input: Dict[str, Any]) -> VoiceEvent:
    """Create a tool call event (audit, not spoken)."""
    return VoiceEvent(
        turn_id=turn_id,
        source="tool",
        channel=Channel.TOOL,
        type=EventType.TOOL_CALL,
        commit=CommitLevel.AUDIT,
        payload={"tool": tool_name, "input": tool_input},
    )


def tool_result_event(turn_id: str, tool_name: str, result: Any) -> VoiceEvent:
    """Create a tool result event (audit, not spoken)."""
    return VoiceEvent(
        turn_id=turn_id,
        source="tool",
        channel=Channel.TOOL,
        type=EventType.TOOL_RESULT,
        commit=CommitLevel.AUDIT,
        payload={"tool": tool_name, "result": result},
    )


def interrupt_event(turn_id: str, cancelled_generation_id: str) -> VoiceEvent:
    """Create an interrupt event."""
    return VoiceEvent(
        turn_id=turn_id,
        source="user",
        channel=Channel.AUDIO,
        type=EventType.INTERRUPT,
        commit=CommitLevel.AUDIT,
        payload={"cancelled_generation_id": cancelled_generation_id},
    )


def cancel_event(generation_id: str, reason: str = "user_interrupt") -> VoiceEvent:
    """Create a cancel event for speech generation."""
    return VoiceEvent(
        generation_id=generation_id,
        source="system",
        channel=Channel.AUDIO,
        type=EventType.CANCEL,
        commit=CommitLevel.AUDIT,
        payload={"reason": reason},
    )


__all__ = [
    "Channel",
    "CommitLevel",
    "EventType",
    "VoiceEvent",
    "DEFAULT_COMMIT_LEVELS",
    # Factory functions
    "user_turn_start",
    "user_utterance",
    "agent_turn_start",
    "agent_speech_chunk",
    "agent_speech_complete",
    "tool_call_event",
    "tool_result_event",
    "interrupt_event",
    "cancel_event",
]
