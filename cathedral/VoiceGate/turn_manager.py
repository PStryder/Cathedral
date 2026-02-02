"""
Turn Manager - Conversation turn state machine.

Manages the flow of conversation turns:
- Who is currently speaking (user/agent)
- Interrupt detection and handling
- Turn transitions
- Barge-in support

State machine:
    IDLE -> USER_SPEAKING -> (interrupt?) -> AGENT_SPEAKING -> IDLE
                 |                                  |
                 v                                  v
            (silence)                          (complete)
                 |                                  |
                 v                                  v
         AGENT_SPEAKING <------------------- USER_SPEAKING
                                (barge-in)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, List, Dict, Any

from cathedral.shared.gate import GateLogger
from .events import (
    VoiceEvent, Channel, CommitLevel, EventType,
    user_turn_start, user_utterance, agent_turn_start,
    interrupt_event, cancel_event,
)

_log = GateLogger.get("VoiceGate.TurnManager")


class TurnState(Enum):
    """Conversation turn states."""
    IDLE = auto()            # No one speaking
    USER_SPEAKING = auto()   # User is speaking
    AGENT_SPEAKING = auto()  # Agent is speaking
    AGENT_THINKING = auto()  # Agent processing (tools, etc.)
    INTERRUPTED = auto()     # User interrupted agent


@dataclass
class Turn:
    """Represents a conversation turn."""
    turn_id: str
    source: str  # "user" or "agent"
    started_at: float
    ended_at: Optional[float] = None
    generation_id: Optional[str] = None  # For agent turns
    text: str = ""
    was_interrupted: bool = False

    @property
    def duration(self) -> float:
        """Get turn duration in seconds."""
        end = self.ended_at or time.time()
        return end - self.started_at

    @property
    def is_active(self) -> bool:
        """Check if turn is still active."""
        return self.ended_at is None


@dataclass
class TurnManagerConfig:
    """Configuration for turn manager."""
    # Silence detection
    silence_threshold_ms: int = 500      # Silence before turn end
    barge_in_threshold_ms: int = 200     # Speech before interrupt triggers

    # Debouncing
    min_turn_duration_ms: int = 100      # Minimum turn length
    interrupt_cooldown_ms: int = 300     # Cooldown after interrupt


class TurnManager:
    """
    Manages conversation turn state.

    Handles:
    - Turn state transitions
    - Interrupt detection
    - Turn history
    - Event generation for turn boundaries
    """

    def __init__(
        self,
        config: Optional[TurnManagerConfig] = None,
        on_event: Optional[Callable[[VoiceEvent], None]] = None,
        on_interrupt: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize turn manager.

        Args:
            config: Turn manager configuration
            on_event: Callback for turn events
            on_interrupt: Callback when interrupt occurs (receives cancelled generation_id)
        """
        self.config = config or TurnManagerConfig()
        self.on_event = on_event
        self.on_interrupt = on_interrupt

        # State
        self._state = TurnState.IDLE
        self._current_turn: Optional[Turn] = None
        self._turn_history: List[Turn] = []

        # Timing
        self._last_user_audio_time: float = 0
        self._last_agent_audio_time: float = 0
        self._last_interrupt_time: float = 0

        # Voice activity
        self._user_speaking = False
        self._agent_speaking = False

        # Lock for state transitions
        self._lock = asyncio.Lock()

    @property
    def state(self) -> TurnState:
        """Get current turn state."""
        return self._state

    @property
    def current_turn(self) -> Optional[Turn]:
        """Get the current active turn."""
        return self._current_turn

    async def user_audio_received(self) -> Optional[VoiceEvent]:
        """
        Signal that user audio was received.

        May trigger:
        - Turn start (if idle)
        - Interrupt (if agent speaking)

        Returns:
            Event generated, if any
        """
        async with self._lock:
            now = time.time()
            self._last_user_audio_time = now
            self._user_speaking = True

            if self._state == TurnState.IDLE:
                # Start user turn
                return await self._start_user_turn()

            elif self._state == TurnState.AGENT_SPEAKING:
                # Check for barge-in
                time_since_interrupt = (now - self._last_interrupt_time) * 1000

                if time_since_interrupt > self.config.interrupt_cooldown_ms:
                    # Interrupt the agent
                    return await self._interrupt_agent()

            return None

    async def user_silence_detected(self) -> Optional[VoiceEvent]:
        """
        Signal that user has stopped speaking.

        May trigger turn end if silence threshold exceeded.

        Returns:
            Event generated, if any
        """
        async with self._lock:
            self._user_speaking = False

            if self._state == TurnState.USER_SPEAKING:
                # Check if silence long enough to end turn
                now = time.time()
                silence_duration = (now - self._last_user_audio_time) * 1000

                if silence_duration >= self.config.silence_threshold_ms:
                    return await self._end_user_turn()

            return None

    async def agent_speech_started(self, generation_id: str) -> Optional[VoiceEvent]:
        """
        Signal that agent has started speaking.

        Args:
            generation_id: The speech generation ID

        Returns:
            Event generated, if any
        """
        async with self._lock:
            self._agent_speaking = True
            self._last_agent_audio_time = time.time()

            if self._state in (TurnState.IDLE, TurnState.AGENT_THINKING):
                # Start agent turn
                return await self._start_agent_turn(generation_id)

            return None

    async def agent_speech_ended(self, text: str = "") -> Optional[VoiceEvent]:
        """
        Signal that agent has finished speaking.

        Args:
            text: The complete text that was spoken

        Returns:
            Event generated, if any
        """
        async with self._lock:
            self._agent_speaking = False

            if self._state == TurnState.AGENT_SPEAKING:
                return await self._end_agent_turn(text)

            return None

    async def agent_thinking_started(self) -> Optional[VoiceEvent]:
        """
        Signal that agent is processing (e.g., running tools).

        Returns:
            Event generated, if any
        """
        async with self._lock:
            if self._state == TurnState.IDLE:
                self._state = TurnState.AGENT_THINKING
                _log.debug("Agent thinking started")
            return None

    async def _start_user_turn(self) -> VoiceEvent:
        """Start a new user turn."""
        # End any existing turn
        if self._current_turn and self._current_turn.is_active:
            self._current_turn.ended_at = time.time()
            self._turn_history.append(self._current_turn)

        # Create new turn
        turn_id = str(uuid.uuid4())
        self._current_turn = Turn(
            turn_id=turn_id,
            source="user",
            started_at=time.time(),
        )
        self._state = TurnState.USER_SPEAKING

        event = user_turn_start(turn_id)

        _log.debug(f"User turn started: {turn_id[:8]}")

        if self.on_event:
            self.on_event(event)

        return event

    async def _end_user_turn(self) -> Optional[VoiceEvent]:
        """End the current user turn."""
        if not self._current_turn or self._current_turn.source != "user":
            return None

        self._current_turn.ended_at = time.time()

        # Check minimum duration
        if self._current_turn.duration * 1000 < self.config.min_turn_duration_ms:
            _log.debug("Turn too short, ignoring")
            self._current_turn = None
            self._state = TurnState.IDLE
            return None

        self._turn_history.append(self._current_turn)
        self._state = TurnState.IDLE

        _log.debug(f"User turn ended: {self._current_turn.turn_id[:8]}")

        # The actual utterance event is generated elsewhere with the transcribed text
        return None

    async def _start_agent_turn(self, generation_id: str) -> VoiceEvent:
        """Start a new agent turn."""
        # End any existing turn
        if self._current_turn and self._current_turn.is_active:
            self._current_turn.ended_at = time.time()
            self._turn_history.append(self._current_turn)

        # Create new turn
        turn_id = str(uuid.uuid4())
        self._current_turn = Turn(
            turn_id=turn_id,
            source="agent",
            started_at=time.time(),
            generation_id=generation_id,
        )
        self._state = TurnState.AGENT_SPEAKING

        event = agent_turn_start(turn_id, generation_id)

        _log.debug(f"Agent turn started: {turn_id[:8]}, gen={generation_id[:8]}")

        if self.on_event:
            self.on_event(event)

        return event

    async def _end_agent_turn(self, text: str) -> Optional[VoiceEvent]:
        """End the current agent turn."""
        if not self._current_turn or self._current_turn.source != "agent":
            return None

        self._current_turn.ended_at = time.time()
        self._current_turn.text = text
        self._turn_history.append(self._current_turn)
        self._state = TurnState.IDLE

        _log.debug(f"Agent turn ended: {self._current_turn.turn_id[:8]}")

        return None

    async def _interrupt_agent(self) -> VoiceEvent:
        """Interrupt the agent's current speech."""
        self._last_interrupt_time = time.time()

        cancelled_generation_id = None
        if self._current_turn and self._current_turn.generation_id:
            cancelled_generation_id = self._current_turn.generation_id
            self._current_turn.was_interrupted = True
            self._current_turn.ended_at = time.time()
            self._turn_history.append(self._current_turn)

        self._state = TurnState.INTERRUPTED

        # Create interrupt event
        event = interrupt_event(
            turn_id=self._current_turn.turn_id if self._current_turn else str(uuid.uuid4()),
            cancelled_generation_id=cancelled_generation_id or "",
        )

        _log.info(f"Agent interrupted, cancelled generation {cancelled_generation_id[:8] if cancelled_generation_id else 'none'}")

        if self.on_event:
            self.on_event(event)

        if self.on_interrupt and cancelled_generation_id:
            self.on_interrupt(cancelled_generation_id)

        # Immediately transition to user speaking
        return await self._start_user_turn()

    def get_turn_history(self, limit: int = 10) -> List[Turn]:
        """Get recent turn history."""
        return self._turn_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get turn manager status."""
        return {
            "state": self._state.name,
            "current_turn": {
                "turn_id": self._current_turn.turn_id,
                "source": self._current_turn.source,
                "duration": self._current_turn.duration,
                "generation_id": self._current_turn.generation_id,
            } if self._current_turn else None,
            "user_speaking": self._user_speaking,
            "agent_speaking": self._agent_speaking,
            "history_length": len(self._turn_history),
        }

    def reset(self):
        """Reset turn manager state."""
        self._state = TurnState.IDLE
        self._current_turn = None
        self._user_speaking = False
        self._agent_speaking = False
        _log.debug("Turn manager reset")


__all__ = [
    "TurnState",
    "Turn",
    "TurnManagerConfig",
    "TurnManager",
]
