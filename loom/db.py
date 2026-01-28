"""
Loom PostgreSQL database initialization with pgvector support.
"""

import os
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)


class LoomDB:
    """Database state holder for Loom."""

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


def init_db() -> None:
    """Initialize database connection and create tables."""
    if LoomDB._initialized:
        return

    database_url = get_database_url()
    async_database_url = get_async_database_url()

    print("[Loom DB] Connecting to PostgreSQL...")

    # Create sync engine
    LoomDB.engine = create_engine(database_url, pool_pre_ping=True)
    LoomDB.SessionLocal = sessionmaker(bind=LoomDB.engine)

    # Create async engine
    LoomDB.async_engine = create_async_engine(async_database_url, pool_pre_ping=True)
    LoomDB.AsyncSessionLocal = async_sessionmaker(
        bind=LoomDB.async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    # Ensure pgvector extension exists
    with LoomDB.engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create tables
    from loom.models import Base
    Base.metadata.create_all(bind=LoomDB.engine)

    LoomDB._initialized = True
    print("[Loom DB] Database initialized")


def get_session() -> Session:
    """Get a sync database session."""
    if not LoomDB._initialized:
        init_db()
    return LoomDB.SessionLocal()


async def get_async_session() -> AsyncSession:
    """Get an async database session."""
    if not LoomDB._initialized:
        init_db()
    return LoomDB.AsyncSessionLocal()


def is_initialized() -> bool:
    """Check if database is initialized."""
    return LoomDB._initialized
