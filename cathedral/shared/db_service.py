"""
Centralized database service for Cathedral.

Provides a single, explicit initialization point and shared engines/sessions.
No engines or sessions are created at import time.
"""

from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

_engine: Optional[Engine] = None
_async_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[sessionmaker] = None
_AsyncSessionLocal: Optional[async_sessionmaker] = None
_initialized = False

_database_url: Optional[str] = None
_sync_url: Optional[str] = None
_async_url: Optional[str] = None


def _build_urls(database_url: str) -> Tuple[str, str]:
    url = make_url(database_url)
    drivername = url.drivername

    if drivername.startswith("postgresql"):
        sync_driver = "postgresql"
        async_driver = "postgresql+asyncpg"
        if "+asyncpg" in drivername:
            async_url = database_url
            sync_url = url.set(drivername=sync_driver).render_as_string(hide_password=False)
        else:
            sync_url = database_url
            async_url = url.set(drivername=async_driver).render_as_string(hide_password=False)
        return sync_url, async_url

    if drivername.startswith("sqlite"):
        sync_driver = "sqlite"
        async_driver = "sqlite+aiosqlite"
        if "+aiosqlite" in drivername:
            async_url = database_url
            sync_url = url.set(drivername=sync_driver).render_as_string(hide_password=False)
        else:
            sync_url = database_url
            async_url = url.set(drivername=async_driver).render_as_string(hide_password=False)
        return sync_url, async_url

    return database_url, database_url


def _sqlite_engine_kwargs(sync_url: str) -> dict:
    url = make_url(sync_url)
    if not url.drivername.startswith("sqlite"):
        return {}

    kwargs: dict = {"connect_args": {"check_same_thread": False}}
    if url.database in (None, "", ":memory:"):
        kwargs["poolclass"] = StaticPool
    return kwargs


def init_db(database_url: str, *, echo: bool = False) -> None:
    """
    Initialize shared database engines and sessionmakers.

    Must be called explicitly at runtime startup or in tests.
    """
    global _engine, _async_engine, _SessionLocal, _AsyncSessionLocal
    global _initialized, _database_url, _sync_url, _async_url

    if _initialized:
        return

    if not database_url:
        raise RuntimeError("database_url is required to initialize the DB service")

    sync_url, async_url = _build_urls(database_url)
    sqlite_kwargs = _sqlite_engine_kwargs(sync_url)

    engine_kwargs = {"echo": echo, "pool_pre_ping": True}
    engine_kwargs.update(sqlite_kwargs)
    _engine = create_engine(sync_url, **engine_kwargs)

    async_kwargs = {"echo": echo, "pool_pre_ping": True}
    async_kwargs.update(sqlite_kwargs)
    _async_engine = create_async_engine(async_url, **async_kwargs)

    _SessionLocal = sessionmaker(bind=_engine)
    _AsyncSessionLocal = async_sessionmaker(
        bind=_async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    _database_url = database_url
    _sync_url = sync_url
    _async_url = async_url
    _initialized = True


def is_initialized() -> bool:
    """Return True if the DB service has been initialized."""
    return _initialized


def _require_initialized() -> None:
    if not _initialized:
        raise RuntimeError("Database not initialized. Call init_db(...) before using the DB service.")


def get_engine() -> Engine:
    _require_initialized()
    return _engine


def get_async_engine() -> AsyncEngine:
    _require_initialized()
    return _async_engine


def get_sessionmaker() -> sessionmaker:
    _require_initialized()
    return _SessionLocal


def get_async_sessionmaker() -> async_sessionmaker:
    _require_initialized()
    return _AsyncSessionLocal


@contextmanager
def get_session():
    """Get a sync session as a context manager."""
    _require_initialized()
    session = _SessionLocal()
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
    """Get an async session as a context manager."""
    _require_initialized()
    session = _AsyncSessionLocal()
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
    _require_initialized()
    return _SessionLocal()


async def get_raw_async_session() -> AsyncSession:
    """Get a raw async session (caller manages lifecycle)."""
    _require_initialized()
    return _AsyncSessionLocal()

