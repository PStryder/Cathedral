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
    _tables_initialized = True


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
