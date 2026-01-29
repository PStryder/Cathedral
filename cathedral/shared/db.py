"""
Shared database utilities for Cathedral.

Thin wrappers over the canonical DB service.
"""

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from . import db_service


def init_db(database_url: str, *, echo: bool = False) -> None:
    """Initialize the shared DB service."""
    db_service.init_db(database_url, echo=echo)


def is_initialized() -> bool:
    """Check if database is initialized."""
    return db_service.is_initialized()


def ensure_initialized() -> None:
    """Require database initialization."""
    if not is_initialized():
        raise RuntimeError("Database not initialized. Call init_db(...) before using the DB.")


def get_session():
    """Get a sync database session (context manager)."""
    return db_service.get_session()


def get_async_session():
    """Get an async database session (context manager)."""
    return db_service.get_async_session()


def get_raw_session() -> Session:
    """Get a raw sync session (caller manages lifecycle)."""
    return db_service.get_raw_session()


async def get_raw_async_session() -> AsyncSession:
    """Get a raw async session (caller manages lifecycle)."""
    return await db_service.get_raw_async_session()


def get_engine():
    """Get the SQLAlchemy engine."""
    return db_service.get_engine()
