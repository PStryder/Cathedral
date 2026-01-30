"""
Shared utilities for Cathedral.

Provides access to common functionality used across Gate implementations.
"""

from cathedral.shared.db import (
    init_db,
    is_initialized,
    ensure_initialized,
    get_session,
    get_async_session,
    get_raw_session,
    get_raw_async_session,
    get_engine,
)

from cathedral.shared.embeddings import (
    is_configured as embeddings_configured,
    embed_text,
    embed_texts,
    embed_text_batch,
    cosine_similarity,
    close_client as close_embedding_client,
)

from cathedral.shared.gate import (
    GateLogger,
    GateErrorHandler,
    GateHealth,
    GateOperationResult,
    ConfigLoader,
    PathUtils,
    build_health_status,
    deep_update,
    get_logger,
)

__all__ = [
    # Database
    "init_db",
    "is_initialized",
    "ensure_initialized",
    "get_session",
    "get_async_session",
    "get_raw_session",
    "get_raw_async_session",
    "get_engine",
    # Embeddings
    "embeddings_configured",
    "embed_text",
    "embed_texts",
    "embed_text_batch",
    "cosine_similarity",
    "close_embedding_client",
    # Gate utilities
    "GateLogger",
    "GateErrorHandler",
    "GateHealth",
    "GateOperationResult",
    "ConfigLoader",
    "PathUtils",
    "build_health_status",
    "deep_update",
    "get_logger",
]
