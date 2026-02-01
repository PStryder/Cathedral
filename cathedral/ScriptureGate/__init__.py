"""
ScriptureGate - Indexed document/file library for Cathedral.

Provides:
- File storage with automatic organization
- Content extraction (text, image descriptions, audio transcription)
- Semantic search via embeddings
- Reference-based retrieval
- RAG-lite context injection

All binary content lives here, with searchable indexes in PostgreSQL.
Other systems (conversation memory, MemoryGate) reference scriptures via refs.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO

from sqlalchemy import select, update, delete, text, or_, and_
from sqlalchemy.orm import Session

from cathedral.shared.gate import GateLogger, build_health_status

# Logger for this gate
_log = GateLogger.get("ScriptureGate")

# Import submodules
from cathedral.ScriptureGate.storage import (
    store_file,
    store_content,
    read_file,
    read_text,
    delete_file,
    file_exists,
    get_full_path,
    detect_file_type,
    detect_mime_type,
    SCRIPTURE_ROOT,
)
from cathedral.ScriptureGate.models import Scripture, Base, EMBEDDING_DIM
from cathedral.ScriptureGate.indexer import (
    extract_text_content,
    generate_embedding,
    build_searchable_text,
)

# Database - use shared database utilities (backend-agnostic)
from cathedral.shared.db import (
    get_session,
    get_async_session,
    is_initialized as db_initialized,
    get_engine,
)

_tables_created = False


def _ensure_tables():
    """Ensure scripture tables exist."""
    global _tables_created
    if _tables_created:
        return

    if not db_initialized():
        raise RuntimeError("Database not initialized. Call init_db(...) before using ScriptureGate.")

    engine = get_engine()

    # Create pgvector extension if using PostgreSQL
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    Base.metadata.create_all(bind=engine)

    # Migrate embedding column from text to vector if needed (PostgreSQL only)
    if engine.dialect.name == "postgresql":
        _migrate_embedding_column(engine)

    _tables_created = True


def _migrate_embedding_column(engine):
    """Migrate embedding column from text to vector type if needed."""
    with engine.connect() as conn:
        # Check current column type
        result = conn.execute(text("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = 'scripture_index' AND column_name = 'embedding'
        """))
        row = result.fetchone()
        if row is None:
            return  # Column doesn't exist, will be created fresh

        current_type = row[0].lower()
        if current_type == "user-defined":
            # Already vector type
            return

        if current_type == "text":
            _log.info("Migrating scripture_index.embedding from text to vector type...")
            try:
                # Drop the text column and recreate as vector
                conn.execute(text("ALTER TABLE scripture_index DROP COLUMN embedding"))
                conn.execute(text(f"ALTER TABLE scripture_index ADD COLUMN embedding vector({EMBEDDING_DIM})"))
                conn.commit()
                _log.info("Successfully migrated embedding column to vector type")
            except Exception as e:
                _log.error(f"Failed to migrate embedding column: {e}")
                conn.rollback()


def init_scripture_db() -> None:
    """Initialize scripture tables (explicit)."""
    _ensure_tables()


# ==================== Health Checks ====================


def is_healthy() -> bool:
    """Check if the gate is operational."""
    try:
        return _tables_created and db_initialized() and SCRIPTURE_ROOT.exists()
    except Exception:
        return False


def get_health_status() -> Dict[str, Any]:
    """Get detailed health information."""
    checks = {
        "tables_created": _tables_created,
        "db_initialized": db_initialized(),
        "storage_dir_exists": SCRIPTURE_ROOT.exists(),
    }

    details = {
        "storage_root": str(SCRIPTURE_ROOT),
    }

    if _tables_created:
        try:
            with get_session() as session:
                count = session.execute(
                    select(Scripture).where(Scripture.is_deleted.is_(False))
                ).scalars().all()
                details["scripture_count"] = len(count)
        except Exception as e:
            details["scripture_count"] = f"error: {e}"

    return build_health_status(
        gate_name="ScriptureGate",
        initialized=_tables_created,
        dependencies=["database", "filesystem"],
        checks=checks,
        details=details,
    )


def get_dependencies() -> List[str]:
    """List external dependencies."""
    return ["database", "filesystem"]


# ==================== Store Operations ====================


async def store(
    source: Union[str, Path, BinaryIO, bytes],
    title: str = None,
    description: str = None,
    tags: List[str] = None,
    file_type: str = None,
    original_name: str = None,
    source_type: str = "upload",
    source_ref: str = None,
    metadata: Dict = None,
    auto_index: bool = True
) -> Dict:
    """
    Store a file or content in ScriptureGate.

    Args:
        source: File path, file object, or bytes content
        title: Human-readable title
        description: Description for search
        tags: List of tags
        file_type: Override type detection (document, image, audio, artifact)
        original_name: Original filename (required for bytes/file objects)
        source_type: Origin (upload, agent, export, generated)
        source_ref: Reference to source (e.g., agent_id, thread_uid)
        metadata: Additional metadata dict
        auto_index: Whether to extract text and generate embedding

    Returns:
        Scripture dict with uid, ref, etc.
    """
    _ensure_tables()

    # Handle bytes content
    if isinstance(source, bytes):
        if not original_name:
            raise ValueError("original_name required for bytes content")
        scripture_uid, rel_path, file_size, mime_type, content_hash = store_content(
            source, original_name, file_type
        )
    # Handle string content (for text)
    elif isinstance(source, str) and not Path(source).exists():
        if not original_name:
            original_name = "content.txt"
        scripture_uid, rel_path, file_size, mime_type, content_hash = store_content(
            source, original_name, file_type
        )
    else:
        # File path or file object
        scripture_uid, rel_path, file_size, mime_type, content_hash = store_file(
            source, file_type, original_name
        )

    # Determine file type if not set
    if file_type is None:
        file_type = detect_file_type(rel_path, mime_type)

    # Create index entry
    with get_session() as session:
        scripture = Scripture(
            scripture_uid=scripture_uid,
            file_path=rel_path,
            file_type=file_type,
            mime_type=mime_type,
            file_size=file_size,
            title=title,
            description=description,
            tags=tags,
            extra_metadata=metadata,
            source=source_type,
            source_ref=source_ref,
            content_hash=content_hash,
        )
        session.add(scripture)
        session.commit()
        result = scripture.to_dict()
        result["ref"] = scripture.to_ref()

    # Index asynchronously if requested
    if auto_index:
        task = asyncio.create_task(_index_scripture(scripture_uid))
        task.add_done_callback(_handle_index_task_exception)

    return result


def _handle_index_task_exception(task: asyncio.Task) -> None:
    """Handle exceptions from background indexing tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        _log.error(f"Background indexing failed: {exc}")


async def store_text(
    content: str,
    title: str,
    filename: str = None,
    description: str = None,
    tags: List[str] = None,
    source_type: str = "generated",
    source_ref: str = None,
    metadata: Dict = None
) -> Dict:
    """Convenience function to store text content."""
    if filename is None:
        filename = f"{title.lower().replace(' ', '_')}.txt"

    return await store(
        source=content,
        title=title,
        description=description,
        tags=tags,
        file_type="document",
        original_name=filename,
        source_type=source_type,
        source_ref=source_ref,
        metadata=metadata
    )


async def store_artifact(
    content: Union[str, bytes, Dict],
    title: str,
    filename: str = None,
    description: str = None,
    tags: List[str] = None,
    source_ref: str = None,
    metadata: Dict = None
) -> Dict:
    """Store an agent-generated artifact."""
    # Handle dict as JSON
    if isinstance(content, dict):
        content = json.dumps(content, indent=2)
        if filename is None:
            filename = f"{title.lower().replace(' ', '_')}.json"

    if filename is None:
        filename = f"{title.lower().replace(' ', '_')}.txt"

    return await store(
        source=content,
        title=title,
        description=description,
        tags=tags,
        file_type="artifact",
        original_name=filename,
        source_type="agent",
        source_ref=source_ref,
        metadata=metadata
    )


# ==================== Index Operations ====================


async def _index_scripture(scripture_uid: str) -> bool:
    """Index a scripture (extract text, generate embedding)."""
    _ensure_tables()

    async with get_async_session() as session:
        result = await session.execute(
            select(Scripture).where(Scripture.scripture_uid == scripture_uid)
        )
        scripture = result.scalar_one_or_none()

        if not scripture:
            return False

        # Extract text content
        extracted = await extract_text_content(
            scripture.file_path,
            scripture.file_type,
            scripture.mime_type
        )

        if extracted:
            scripture.extracted_text = extracted

        # Build searchable text
        searchable = build_searchable_text(
            title=scripture.title,
            description=scripture.description,
            extracted_text=extracted,
            tags=scripture.tags
        )

        # Generate embedding
        if searchable:
            embedding = await generate_embedding(searchable)
            if embedding:
                scripture.embedding = embedding
                scripture.is_indexed = True
                scripture.indexed_at = datetime.utcnow()

        await session.commit()
        return scripture.is_indexed


async def reindex(scripture_uid: str) -> bool:
    """Re-index a scripture (regenerate text and embedding)."""
    return await _index_scripture(scripture_uid)


async def backfill_index(batch_size: int = 20) -> int:
    """Index scriptures that haven't been indexed yet."""
    _ensure_tables()

    async with get_async_session() as session:
        result = await session.execute(
            select(Scripture.scripture_uid)
            .where(Scripture.is_indexed.is_(False))
            .where(Scripture.is_deleted.is_(False))
            .limit(batch_size)
        )
        unindexed = result.scalars().all()

    count = 0
    for uid in unindexed:
        if await _index_scripture(uid):
            count += 1

    return count


# ==================== Search Operations ====================


async def search(
    query: str,
    limit: int = 10,
    file_type: str = None,
    tags: List[str] = None,
    min_similarity: float = 0.3
) -> List[Dict]:
    """
    Semantic search across scriptures.

    Args:
        query: Search query
        limit: Max results
        file_type: Filter by type (document, image, audio, artifact)
        tags: Filter by tags (any match)
        min_similarity: Minimum similarity threshold

    Returns:
        List of search results with similarity scores
    """
    _ensure_tables()

    # Generate query embedding
    query_embedding = await generate_embedding(query)
    if query_embedding is None:
        return []

    async with get_async_session() as session:
        # Build query with filters
        filters = ["is_deleted = false", "is_indexed = true"]
        # Format embedding as pgvector-compatible string (no spaces after commas)
        # Embed directly in SQL since asyncpg CAST doesn't work well with vector type
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        params = {"limit": limit}

        if file_type:
            filters.append("file_type = :file_type")
            params["file_type"] = file_type

        filter_clause = " AND ".join(filters)

        sql = text(f"""
            SELECT scripture_uid, file_path, file_type, title, description, tags,
                   1 - (embedding <=> '{embedding_str}'::vector) as similarity
            FROM scripture_index
            WHERE {filter_clause}
              AND embedding IS NOT NULL
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT :limit
        """)

        result = await session.execute(sql, params)
        rows = result.fetchall()

        results = []
        for row in rows:
            similarity = float(row[6]) if row[6] else 0.0
            if similarity >= min_similarity:
                results.append({
                    "uid": row[0],
                    "ref": f"scripture:{row[2]}/{row[0][:8]}",
                    "file_path": row[1],
                    "file_type": row[2],
                    "title": row[3] or row[1].split("/")[-1],
                    "description": row[4][:200] if row[4] else None,
                    "tags": row[5] or [],
                    "similarity": similarity,
                })

        # Filter by tags if specified
        if tags and results:
            results = [
                r for r in results
                if any(t in (r.get("tags") or []) for t in tags)
            ]

        return results


def search_by_ref(ref: str) -> Optional[Dict]:
    """
    Get scripture by reference.

    Ref format: scripture:<type>/<uid_prefix>
    """
    _ensure_tables()

    if not ref.startswith("scripture:"):
        return None

    parts = ref[10:].split("/", 1)
    if len(parts) != 2:
        return None

    file_type, uid_prefix = parts

    with get_session() as session:
        scripture = session.execute(
            select(Scripture)
            .where(Scripture.scripture_uid.like(f"{uid_prefix}%"))
            .where(Scripture.file_type == file_type)
            .where(Scripture.is_deleted.is_(False))
        ).scalar_one_or_none()

        if scripture:
            result = scripture.to_dict()
            result["ref"] = scripture.to_ref()
            return result

    return None


def get(scripture_uid: str) -> Optional[Dict]:
    """Get scripture by UID."""
    _ensure_tables()

    with get_session() as session:
        scripture = session.execute(
            select(Scripture)
            .where(Scripture.scripture_uid == scripture_uid)
            .where(Scripture.is_deleted.is_(False))
        ).scalar_one_or_none()

        if scripture:
            result = scripture.to_dict()
            result["ref"] = scripture.to_ref()
            return result

    return None


def list_scriptures(
    file_type: str = None,
    tags: List[str] = None,
    source: str = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict]:
    """List scriptures with optional filters."""
    _ensure_tables()

    with get_session() as session:
        query = select(Scripture).where(Scripture.is_deleted.is_(False))

        if file_type:
            query = query.where(Scripture.file_type == file_type)
        if source:
            query = query.where(Scripture.source == source)

        query = query.order_by(Scripture.created_at.desc())
        query = query.offset(offset).limit(limit)

        scriptures = session.execute(query).scalars().all()
        results = []
        for s in scriptures:
            result = s.to_dict()
            result["ref"] = s.to_ref()
            results.append(result)

        # Filter by tags if specified
        if tags and results:
            results = [
                r for r in results
                if any(t in (r.get("tags") or []) for t in tags)
            ]

        return results


# ==================== Read Operations ====================


def read(scripture_uid: str, as_text: bool = False) -> Union[bytes, str, None]:
    """Read file content by UID."""
    _ensure_tables()

    with get_session() as session:
        scripture = session.execute(
            select(Scripture.file_path)
            .where(Scripture.scripture_uid == scripture_uid)
            .where(Scripture.is_deleted.is_(False))
        ).scalar_one_or_none()

        if not scripture:
            return None

    if as_text:
        return read_text(scripture)
    return read_file(scripture)


def get_path(scripture_uid: str) -> Optional[Path]:
    """Get full file path by UID."""
    _ensure_tables()

    with get_session() as session:
        rel_path = session.execute(
            select(Scripture.file_path)
            .where(Scripture.scripture_uid == scripture_uid)
            .where(Scripture.is_deleted.is_(False))
        ).scalar_one_or_none()

        if rel_path:
            return get_full_path(rel_path)

    return None


# ==================== Delete Operations ====================


def remove(scripture_uid: str, hard_delete: bool = False) -> bool:
    """
    Remove a scripture.

    Args:
        scripture_uid: Scripture UID
        hard_delete: If True, delete file from disk too

    Returns:
        True if removed
    """
    _ensure_tables()

    with get_session() as session:
        scripture = session.execute(
            select(Scripture)
            .where(Scripture.scripture_uid == scripture_uid)
        ).scalar_one_or_none()

        if not scripture:
            return False

        if hard_delete:
            # Delete file from disk
            delete_file(scripture.file_path)
            session.delete(scripture)
        else:
            # Soft delete
            scripture.is_deleted = True

        session.commit()
        return True


# ==================== RAG Context Building ====================


async def build_context(
    query: str,
    limit: int = 3,
    file_types: List[str] = None,
    min_similarity: float = 0.4
) -> str:
    """
    Build context string from relevant scriptures for RAG injection.

    Args:
        query: User query to match against
        limit: Max scriptures to include
        file_types: Filter by types
        min_similarity: Minimum relevance threshold

    Returns:
        Formatted context string for prompt injection
    """
    results = []

    if file_types:
        for ft in file_types:
            ft_results = await search(query, limit=limit, file_type=ft, min_similarity=min_similarity)
            results.extend(ft_results)
        # Sort by similarity and take top
        results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)[:limit]
    else:
        results = await search(query, limit=limit, min_similarity=min_similarity)

    if not results:
        return ""

    lines = ["[RELEVANT DOCUMENTS]"]
    for r in results:
        ref = r.get("ref", "?")
        title = r.get("title", "Untitled")
        desc = r.get("description", "")[:150] if r.get("description") else ""
        sim = r.get("similarity", 0)
        lines.append(f"- [{ref}] {title} (relevance: {sim:.2f})")
        if desc:
            lines.append(f"  {desc}")

    return "\n".join(lines)


# ==================== Legacy Compatibility ====================

# These maintain compatibility with the old ScriptureGate API

def export_thread(thread_data: list, name: str):
    """Export thread to scripture (legacy compat)."""
    import asyncio
    content = json.dumps({"thread": thread_data}, indent=2)

    # Handle both sync and async contexts
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - schedule as task
        loop.create_task(store(
            source=content,
            title=name,
            original_name=f"{name}.thread.json",
            file_type="thread",
            source_type="export",
            tags=["thread", "export"]
        ))
    except RuntimeError:
        # No running loop - create one (sync context)
        asyncio.run(store(
            source=content,
            title=name,
            original_name=f"{name}.thread.json",
            file_type="thread",
            source_type="export",
            tags=["thread", "export"]
        ))


def import_bios(path: str) -> str:
    """Import bios file (legacy compat)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def import_glyph(path: str) -> dict:
    """Import glyph file (legacy compat)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================== Stats ====================


def stats() -> Dict:
    """Get scripture statistics."""
    _ensure_tables()

    with get_session() as session:
        total = session.execute(
            select(Scripture).where(Scripture.is_deleted.is_(False))
        ).scalars().all()

        by_type = {}
        indexed_count = 0
        total_size = 0

        for s in total:
            by_type[s.file_type] = by_type.get(s.file_type, 0) + 1
            if s.is_indexed:
                indexed_count += 1
            if s.file_size:
                total_size += s.file_size

        return {
            "total": len(total),
            "indexed": indexed_count,
            "by_type": by_type,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


__all__ = [
    # Initialization
    "init_scripture_db",
    # Health
    "is_healthy",
    "get_health_status",
    "get_dependencies",
    # Storage
    "store",
    "store_text",
    "store_artifact",
    # Retrieval
    "get",
    "get_by_ref",
    "list_scriptures",
    "search",
    # Read
    "read",
    "get_path",
    # Delete
    "remove",
    # RAG
    "build_context",
    # Stats
    "stats",
    # Legacy
    "export_thread",
    "import_bios",
    "import_glyph",
    # Models
    "Scripture",
    "SCRIPTURE_ROOT",
]
