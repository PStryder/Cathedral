"""
Shared database utilities for Cathedral.

Conversation storage is provided by MemoryGate conversation service.
"""

from contextlib import contextmanager, asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from cathedral.MemoryGate.conversation import db as conv_db


def init_db() -> None:
    """Initialize the conversation database."""
    conv_db.init_conversation_db()


def is_initialized() -> bool:
    """Check if database is initialized."""
    return conv_db.is_initialized()


def ensure_initialized() -> None:
    """Ensure database is initialized."""
    if not is_initialized():
        init_db()


@contextmanager
def get_session():
    """Get a sync database session (context manager)."""
    ensure_initialized()
    with conv_db.get_session() as session:
        yield session


@asynccontextmanager
async def get_async_session():
    """Get an async database session (context manager)."""
    ensure_initialized()
    async with conv_db.get_async_session() as session:
        yield session


def get_raw_session() -> Session:
    """Get a raw sync session (caller manages lifecycle)."""
    ensure_initialized()
    return conv_db.get_raw_session()


async def get_raw_async_session() -> AsyncSession:
    """Get a raw async session (caller manages lifecycle)."""
    ensure_initialized()
    return await conv_db.get_raw_async_session()


def get_engine():
    """Get the SQLAlchemy engine."""
    ensure_initialized()
    return conv_db.ConversationDB.engine
