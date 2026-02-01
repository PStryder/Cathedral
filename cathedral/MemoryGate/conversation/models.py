"""
Conversation models for MemoryGate.

Provides SQLAlchemy models for conversation memory, replacing legacy models
with a unified implementation under MemoryGate.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float,
    DateTime, ForeignKey, Index, func
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID

# Try to import pgvector
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    Vector = None

Base = declarative_base()

# Embedding dimension for OpenAI text-embedding-3-small
EMBEDDING_DIM = 1536


class ConversationThread(Base):
    """
    Conversation thread.

    A thread represents a conversation session with multiple messages.
    Maps to legacy thread tables for migration compatibility.
    """
    __tablename__ = "mg_conversation_threads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_uid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    thread_name = Column(String(255), nullable=False, default="New Thread")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=False)

    # Optional metadata
    personality_id = Column(String(100), nullable=True)
    metadata_json = Column(Text, nullable=True)  # JSON string for extensible metadata

    # Relationships
    messages = relationship(
        "ConversationMessage",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.timestamp"
    )
    summaries = relationship(
        "ConversationSummary",
        back_populates="thread",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_mg_threads_active", "is_active"),
        Index("ix_mg_threads_created", "created_at"),
        Index("ix_mg_threads_updated", "updated_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "thread_uid": self.thread_uid,
            "thread_name": self.thread_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "personality_id": self.personality_id,
            "message_count": len(self.messages) if self.messages else 0,
        }


class ConversationMessage(Base):
    """
    Chat message within a thread.

    Messages can be from user, assistant, or system roles.
    Each message can have an associated embedding for semantic search.
    """
    __tablename__ = "mg_conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_uid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    thread_uid = Column(String(36), ForeignKey("mg_conversation_threads.thread_uid"), nullable=False)
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_type = Column(String(50), default="regular")  # regular, context, summary, injected

    # Optional metadata
    model_used = Column(String(100), nullable=True)  # LLM model that generated this (for assistant)
    token_count = Column(Integer, nullable=True)
    metadata_json = Column(Text, nullable=True)

    # Relationships
    thread = relationship("ConversationThread", back_populates="messages")
    embedding = relationship(
        "ConversationEmbedding",
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_mg_messages_thread", "thread_uid"),
        Index("ix_mg_messages_timestamp", "timestamp"),
        Index("ix_mg_messages_role", "role"),
        Index("ix_mg_messages_type", "message_type"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_uid": self.message_uid,
            "thread_uid": self.thread_uid,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "message_type": self.message_type,
            "model_used": self.model_used,
            "has_embedding": self.embedding is not None,
        }


class ConversationEmbedding(Base):
    """
    Vector embedding for a message.

    Stores the semantic embedding for vector similarity search.
    Uses pgvector for efficient nearest-neighbor queries.
    """
    __tablename__ = "mg_conversation_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_uid = Column(
        String(36),
        ForeignKey("mg_conversation_messages.message_uid", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )
    embedding = Column(Vector(EMBEDDING_DIM)) if PGVECTOR_AVAILABLE else Column(Text)
    model_version = Column(String(100), default="text-embedding-3-small")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    message = relationship("ConversationMessage", back_populates="embedding")

    __table_args__ = (
        Index("ix_mg_embeddings_message", "message_uid"),
    )


class ConversationSummary(Base):
    """
    Conversation summary for context compression.

    Summaries are generated to compress older messages while
    retaining semantic meaning for context injection.
    """
    __tablename__ = "mg_conversation_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_uid = Column(
        String(36),
        ForeignKey("mg_conversation_threads.thread_uid", ondelete="CASCADE"),
        nullable=False
    )
    summary_text = Column(Text, nullable=False)
    message_range_start = Column(Integer, nullable=True)  # Starting message ID
    message_range_end = Column(Integer, nullable=True)    # Ending message ID
    message_count = Column(Integer, nullable=True)        # Number of messages summarized
    created_at = Column(DateTime, default=datetime.utcnow)

    # Embedding for semantic search over summaries
    embedding = Column(Vector(EMBEDDING_DIM)) if PGVECTOR_AVAILABLE else Column(Text)
    model_version = Column(String(100), default="text-embedding-3-small")

    # Relationships
    thread = relationship("ConversationThread", back_populates="summaries")

    __table_args__ = (
        Index("ix_mg_summaries_thread", "thread_uid"),
        Index("ix_mg_summaries_created", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "thread_uid": self.thread_uid,
            "summary_text": self.summary_text,
            "message_range_start": self.message_range_start,
            "message_range_end": self.message_range_end,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "has_embedding": self.embedding is not None,
        }


class GenericEmbedding(Base):
    """
    Generic embedding storage for Cathedral thread embeddings.

    Note: MemoryGate creates its own 'embeddings' table via alembic migrations.
    This table is for Cathedral-specific thread embeddings only.
    """
    __tablename__ = "mg_thread_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(100), nullable=False, default="cathedral")
    source_type = Column(String(50), nullable=False)  # thread, observation, pattern, concept
    source_id = Column(String(100), nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM)) if PGVECTOR_AVAILABLE else Column(Text)
    model_version = Column(String(100), default="text-embedding-3-small")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_embeddings_tenant_source", "tenant_id", "source_type"),
        Index("ix_embeddings_lookup", "tenant_id", "source_type", "source_id"),
        # Unique constraint for upsert
        Index("uq_embeddings_full", "tenant_id", "source_type", "source_id", "model_version", unique=True),
    )


# Helper function to get all models for table creation
def get_all_models():
    """Return list of all conversation models."""
    return [
        ConversationThread,
        ConversationMessage,
        ConversationEmbedding,
        ConversationSummary,
        GenericEmbedding,
    ]
