"""
Cathedral Unified Memory - Type Definitions

Shared types for the unified memory interface.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class MemorySource(str, Enum):
    """Source type for unified search results."""
    CONVERSATION = "conversation"
    OBSERVATION = "observation"
    PATTERN = "pattern"
    CONCEPT = "concept"
    DOCUMENT = "document"
    SUMMARY = "summary"


class MemoryTier(str, Enum):
    """Memory tier for cold/hot storage."""
    HOT = "hot"
    COLD = "cold"


@dataclass
class SearchResult:
    """
    Unified search result from any memory source.

    Provides a common structure for results from conversation memory
    and MemoryGate (observations, patterns, concepts, documents).
    """
    source: MemorySource
    content: str
    similarity: float
    ref: str  # Format: "type:id" e.g., "observation:123", "message:uuid"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional enrichment fields
    timestamp: Optional[datetime] = None
    confidence: Optional[float] = None
    domain: Optional[str] = None
    tier: MemoryTier = MemoryTier.HOT

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source.value,
            "content": self.content,
            "similarity": self.similarity,
            "ref": self.ref,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
            "domain": self.domain,
            "tier": self.tier.value,
        }

    @classmethod
    def from_conversation_message(cls, msg: Dict[str, Any], similarity: float = 0.0) -> "SearchResult":
        """Create from conversation message dict."""
        return cls(
            source=MemorySource.CONVERSATION,
            content=msg.get("content", ""),
            similarity=similarity,
            ref=f"message:{msg.get('message_uid', '')}",
            metadata={
                "role": msg.get("role"),
                "thread_uid": msg.get("thread_uid"),
            },
            timestamp=datetime.fromisoformat(msg["timestamp"]) if msg.get("timestamp") else None,
        )

    @classmethod
    def from_conversation_summary(cls, summary: Dict[str, Any], similarity: float = 0.0) -> "SearchResult":
        """Create from conversation summary dict."""
        return cls(
            source=MemorySource.SUMMARY,
            content=summary.get("summary_text", ""),
            similarity=similarity,
            ref=f"summary:{summary.get('id', '')}",
            metadata={
                "thread_uid": summary.get("thread_uid"),
            },
            timestamp=datetime.fromisoformat(summary["created_at"]) if summary.get("created_at") else None,
        )

    # Legacy aliases
    @classmethod
    def from_loom_message(cls, msg: Dict[str, Any], similarity: float = 0.0) -> "SearchResult":
        """Backward-compatible alias for conversation messages."""
        return cls.from_conversation_message(msg, similarity=similarity)

    @classmethod
    def from_loom_summary(cls, summary: Dict[str, Any], similarity: float = 0.0) -> "SearchResult":
        """Backward-compatible alias for conversation summaries."""
        return cls.from_conversation_summary(summary, similarity=similarity)

    @classmethod
    def from_memorygate(cls, result: Dict[str, Any]) -> "SearchResult":
        """Create from MemoryGate search result."""
        source_type = result.get("source_type", "observation")
        try:
            source = MemorySource(source_type)
        except ValueError:
            source = MemorySource.OBSERVATION

        return cls(
            source=source,
            content=result.get("content", result.get("snippet", "")),
            similarity=result.get("similarity", 0.0),
            ref=result.get("ref", f"{source_type}:{result.get('id', '')}"),
            metadata={
                "domain": result.get("domain"),
                "ai_name": result.get("ai_name"),
            },
            confidence=result.get("confidence"),
            domain=result.get("domain"),
            tier=MemoryTier(result.get("tier", "hot")),
        )


@dataclass
class ThreadInfo:
    """Thread information."""
    thread_uid: str
    thread_name: str
    is_active: bool = False
    created_at: Optional[datetime] = None
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_uid": self.thread_uid,
            "thread_name": self.thread_name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "message_count": self.message_count,
        }


@dataclass
class MemoryStats:
    """Combined memory statistics."""
    # Conversation stats
    thread_count: int = 0
    message_count: int = 0
    embedded_message_count: int = 0
    summary_count: int = 0

    # MemoryGate stats
    observation_count: int = 0
    pattern_count: int = 0
    concept_count: int = 0
    document_count: int = 0

    # System stats
    conversation_available: bool = False
    memorygate_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation": {
                "threads": self.thread_count,
                "messages": self.message_count,
                "embedded": self.embedded_message_count,
                "summaries": self.summary_count,
                "available": self.conversation_available,
            },
            "knowledge": {
                "observations": self.observation_count,
                "patterns": self.pattern_count,
                "concepts": self.concept_count,
                "documents": self.document_count,
                "available": self.memorygate_available,
            }
        }
