"""
Shared database utilities for Cathedral.

Provides backend-agnostic database access that works with either
Loom or MemoryGate conversation backend.
"""

import os
from typing import Optional
from contextlib import contextmanager, asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from cathedral.Memory import CONVERSATION_BACKEND, ConversationBackend


def _get_loom_db():
    """Get Loom database module."""
    from loom import db as loom_db
    return loom_db


def _get_mg_db():
    """Get MemoryGate conversation database module."""
    from cathedral.MemoryGate.conversation import db as mg_db
    return mg_db


def init_db() -> None:
    """Initialize the database for the configured backend."""
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        _get_loom_db().init_db()
    else:
        _get_mg_db().init_conversation_db()


def is_initialized() -> bool:
    """Check if database is initialized."""
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        return _get_loom_db().is_initialized()
    else:
        return _get_mg_db().is_initialized()


def ensure_initialized() -> None:
    """Ensure database is initialized."""
    if not is_initialized():
        init_db()


@contextmanager
def get_session():
    """Get a sync database session (context manager)."""
    ensure_initialized()
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        with _get_loom_db().get_session() as session:
            yield session
    else:
        with _get_mg_db().get_session() as session:
            yield session


@asynccontextmanager
async def get_async_session():
    """Get an async database session (context manager)."""
    ensure_initialized()
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        async with _get_loom_db().get_async_session() as session:
            yield session
    else:
        async with _get_mg_db().get_async_session() as session:
            yield session


def get_raw_session() -> Session:
    """Get a raw sync session (caller manages lifecycle)."""
    ensure_initialized()
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        return _get_loom_db().get_raw_session()
    else:
        return _get_mg_db().get_raw_session()


async def get_raw_async_session() -> AsyncSession:
    """Get a raw async session (caller manages lifecycle)."""
    ensure_initialized()
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        return await _get_loom_db().get_raw_async_session()
    else:
        return await _get_mg_db().get_raw_async_session()


def get_engine():
    """Get the SQLAlchemy engine."""
    ensure_initialized()
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        from loom.db import LoomDB
        return LoomDB.engine
    else:
        from cathedral.MemoryGate.conversation.db import ConversationDB
        return ConversationDB.engine
