"""
Loom SQLAlchemy models with pgvector support.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float,
    DateTime, ForeignKey, Index, func
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

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


class Thread(Base):
    """Conversation thread."""
    __tablename__ = "loom_threads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_uid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    thread_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=False)

    # Relationships
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_loom_threads_active", "is_active"),
        Index("ix_loom_threads_created", "created_at"),
    )

    def to_dict(self) -> dict:
        return {
            "thread_uid": self.thread_uid,
            "thread_name": self.thread_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active
        }


class Message(Base):
    """Chat message within a thread."""
    __tablename__ = "loom_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_uid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    thread_uid = Column(String(36), ForeignKey("loom_threads.thread_uid"), nullable=False)
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_type = Column(String(50), default="regular")  # regular, context, summary

    # Relationships
    thread = relationship("Thread", back_populates="messages")
    embedding = relationship("MessageEmbedding", back_populates="message", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_loom_messages_thread", "thread_uid"),
        Index("ix_loom_messages_timestamp", "timestamp"),
        Index("ix_loom_messages_role", "role"),
    )

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class MessageEmbedding(Base):
    """Vector embedding for a message."""
    __tablename__ = "loom_message_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_uid = Column(String(36), ForeignKey("loom_messages.message_uid"), unique=True, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM)) if PGVECTOR_AVAILABLE else Column(Text)
    model_version = Column(String(100), default="text-embedding-3-small")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    message = relationship("Message", back_populates="embedding")

    __table_args__ = (
        Index("ix_loom_embeddings_message", "message_uid"),
    )


class Fact(Base):
    """Extracted fact from a conversation."""
    __tablename__ = "loom_facts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_uid = Column(String(36), nullable=True)
    pair_id = Column(String(100), nullable=True)
    fact = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Embedding for semantic search
    embedding = Column(Vector(EMBEDDING_DIM)) if PGVECTOR_AVAILABLE else Column(Text)

    __table_args__ = (
        Index("ix_loom_facts_thread", "thread_uid"),
    )


class Tag(Base):
    """Tag associated with a message pair."""
    __tablename__ = "loom_tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_uid = Column(String(36), nullable=True)
    pair_id = Column(String(100), nullable=True)
    tag = Column(String(255), nullable=False)

    __table_args__ = (
        Index("ix_loom_tags_thread", "thread_uid"),
        Index("ix_loom_tags_tag", "tag"),
    )


class UserInfo(Base):
    """Persistent user information."""
    __tablename__ = "loom_user_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=True)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_loom_user_info_user_key", "user_id", "key"),
    )


class Summary(Base):
    """Conversation summary for context compression."""
    __tablename__ = "loom_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_uid = Column(String(36), nullable=False)
    summary_text = Column(Text, nullable=False)
    message_range_start = Column(Integer, nullable=True)  # Message ID range
    message_range_end = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Embedding for semantic search
    embedding = Column(Vector(EMBEDDING_DIM)) if PGVECTOR_AVAILABLE else Column(Text)

    __table_args__ = (
        Index("ix_loom_summaries_thread", "thread_uid"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thread_uid": self.thread_uid,
            "summary_text": self.summary_text,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
