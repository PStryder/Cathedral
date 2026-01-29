"""
Database connection for MemoryGate conversation service.

Uses the same PostgreSQL database as MemoryGate, with pgvector support
for semantic search over conversation embeddings.
"""

import os
from typing import Optional
from contextlib import contextmanager, asynccontextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from pathlib import Path

from .models import Base

# Load .env from project root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)


class ConversationDB:
    """Database state holder for conversation service."""

    # Sync engine (for migrations and simple operations)
    engine = None
    SessionLocal: Optional[sessionmaker] = None

    # Async engine (for async operations)
    async_engine = None
    AsyncSessionLocal: Optional[async_sessionmaker] = None

    _initialized = False


def get_database_url() -> str:
    """Get the database URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is required")
    return url


def get_async_database_url() -> str:
    """Convert sync database URL to async format."""
    url = get_database_url()
    # Convert postgresql:// to postgresql+asyncpg://
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def init_conversation_db() -> None:
    """Initialize database connection and create tables."""
    if ConversationDB._initialized:
        return

    database_url = get_database_url()
    async_database_url = get_async_database_url()

    print("[MemoryGate Conversation] Connecting to PostgreSQL...")

    # Create sync engine
    ConversationDB.engine = create_engine(database_url, pool_pre_ping=True)
    ConversationDB.SessionLocal = sessionmaker(bind=ConversationDB.engine)

    # Create async engine
    ConversationDB.async_engine = create_async_engine(async_database_url, pool_pre_ping=True)
    ConversationDB.AsyncSessionLocal = async_sessionmaker(
        bind=ConversationDB.async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    # Ensure pgvector extension exists
    with ConversationDB.engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create tables
    Base.metadata.create_all(bind=ConversationDB.engine)

    ConversationDB._initialized = True
    print("[MemoryGate Conversation] Database initialized")


def ensure_initialized() -> None:
    """Ensure database is initialized."""
    if not ConversationDB._initialized:
        init_conversation_db()


@contextmanager
def get_session():
    """Get a sync database session as context manager."""
    ensure_initialized()
    session = ConversationDB.SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_session():
    """Get an async database session as context manager."""
    ensure_initialized()
    session = ConversationDB.AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_raw_session() -> Session:
    """Get a raw sync session (caller manages lifecycle)."""
    ensure_initialized()
    return ConversationDB.SessionLocal()


async def get_raw_async_session() -> AsyncSession:
    """Get a raw async session (caller manages lifecycle)."""
    ensure_initialized()
    return ConversationDB.AsyncSessionLocal()


def is_initialized() -> bool:
    """Check if database is initialized."""
    return ConversationDB._initialized


def reset_db_state() -> None:
    """Reset database state (for testing)."""
    ConversationDB.engine = None
    ConversationDB.SessionLocal = None
    ConversationDB.async_engine = None
    ConversationDB.AsyncSessionLocal = None
    ConversationDB._initialized = False
