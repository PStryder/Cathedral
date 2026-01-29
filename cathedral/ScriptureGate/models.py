"""
ScriptureGate SQLAlchemy models for the content index.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, Index, JSON
)
from sqlalchemy.orm import declarative_base
import uuid

# Try to import pgvector
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    Vector = None

Base = declarative_base()

# Embedding dimension (same as conversation embeddings)
EMBEDDING_DIM = 1536


class Scripture(Base):
    """
    Index entry for a stored file/document.
    The actual file lives on disk; this is the searchable index.
    """
    __tablename__ = "scripture_index"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scripture_uid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))

    # File info
    file_path = Column(String(500), nullable=False)  # Relative path within scripture storage
    file_type = Column(String(50), nullable=False)   # document, image, audio, artifact, thread
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=True)       # Bytes

    # Metadata
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)               # List of tags
    extra_metadata = Column(JSON, nullable=True)      # Arbitrary metadata

    # Source tracking
    source = Column(String(50), nullable=False)      # upload, agent, export, generated
    source_ref = Column(String(255), nullable=True)  # e.g., agent_id, thread_uid

    # Content extraction
    extracted_text = Column(Text, nullable=True)     # Text content for search
    content_hash = Column(String(64), nullable=True) # SHA256 for dedup

    # Embedding for semantic search
    embedding = Column(Vector(EMBEDDING_DIM)) if PGVECTOR_AVAILABLE else Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)     # When embedding was generated

    # Status
    is_indexed = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)      # Soft delete

    __table_args__ = (
        Index("ix_scripture_type", "file_type"),
        Index("ix_scripture_source", "source"),
        Index("ix_scripture_created", "created_at"),
        Index("ix_scripture_indexed", "is_indexed"),
    )

    def to_dict(self) -> dict:
        """Convert to dict for API responses."""
        return {
            "uid": self.scripture_uid,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "mime_type": self.mime_type,
            "title": self.title,
            "description": self.description,
            "tags": self.tags or [],
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_indexed": self.is_indexed,
        }

    def to_ref(self) -> str:
        """Get reference string for this scripture."""
        return f"scripture:{self.file_type}/{self.scripture_uid[:8]}"

    def to_search_result(self, similarity: float = 0.0) -> dict:
        """Format for search results."""
        return {
            "ref": self.to_ref(),
            "uid": self.scripture_uid,
            "title": self.title or self.file_path.split("/")[-1],
            "type": self.file_type,
            "description": self.description[:200] if self.description else None,
            "similarity": similarity,
            "tags": self.tags or [],
        }
