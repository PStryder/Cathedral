"""
Database connection for MemoryGate conversation service.

Uses the shared DB service and initializes tables on demand.
"""

from contextlib import contextmanager, asynccontextmanager

from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from cathedral.shared import db_service
from .models import Base

_tables_initialized = False


def init_conversation_db() -> None:
    """Create conversation tables using the shared engine."""
    global _tables_initialized
    if _tables_initialized:
        return

    if not db_service.is_initialized():
        raise RuntimeError("Database not initialized. Call init_db(...) before init_conversation_db().")

    engine = db_service.get_engine()

    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    Base.metadata.create_all(bind=engine)

    # Add FTS support for PostgreSQL only (after table creation)
    if engine.dialect.name == "postgresql":
        _add_fts_support(engine)

    _tables_initialized = True


def _add_fts_support(engine) -> None:
    """
    Add full-text search support to conversation messages table (PostgreSQL only).

    Creates a generated tsvector column and GIN index for hybrid search.
    This is done separately from model definition to maintain SQLite compatibility.
    """
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'mg_conversation_messages' AND column_name = 'search_vector'
        """))
        if result.fetchone() is not None:
            return  # Column already exists

        # Add generated tsvector column
        conn.execute(text("""
            ALTER TABLE mg_conversation_messages
            ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """))

        # Create GIN index for fast full-text search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_mg_messages_search
            ON mg_conversation_messages USING GIN(search_vector)
        """))

        conn.commit()


def _require_ready() -> None:
    if not db_service.is_initialized():
        raise RuntimeError("Database not initialized. Call init_db(...) before using conversation DB.")
    if not _tables_initialized:
        raise RuntimeError("Conversation tables not initialized. Call init_conversation_db() first.")


@contextmanager
def get_session():
    """Get a sync database session as context manager."""
    _require_ready()
    session = db_service.get_sessionmaker()()
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
    _require_ready()
    session = db_service.get_async_sessionmaker()()
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
    _require_ready()
    return db_service.get_sessionmaker()()


async def get_raw_async_session() -> AsyncSession:
    """Get a raw async session (caller manages lifecycle)."""
    _require_ready()
    return db_service.get_async_sessionmaker()()


def is_initialized() -> bool:
    """Check if conversation tables are initialized."""
    return _tables_initialized


def reset_db_state() -> None:
    """Reset table state (for testing)."""
    global _tables_initialized
    _tables_initialized = False
