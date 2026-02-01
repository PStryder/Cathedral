"""
MemoryGate Conversation Service

Provides conversation memory management, replacing the legacy system with a unified
implementation under MemoryGate. Supports:
- Thread management (create, switch, list, delete)
- Message operations (append, recall, clear)
- Semantic search over conversations
- Automatic embedding generation
- Context composition for LLM prompts
- Summarization for context compression
"""

import uuid
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy import select, update, delete, func, text
from sqlalchemy.orm import Session, selectinload

from .db import (
    init_conversation_db,
    get_session,
    get_async_session,
    get_raw_session,
    get_raw_async_session,
    is_initialized,
)
from .models import (
    ConversationThread,
    ConversationMessage,
    ConversationEmbedding,
    ConversationSummary,
    EMBEDDING_DIM,
)
from .embeddings import (
    embed_text,
    embed_texts,
    embed_text_batch,
    is_configured as embeddings_configured,
)

# Optional: LoomMirror for local summarization (lazy loaded)
_summarizer = None
_summarizer_checked = False
SUMMARIZER_AVAILABLE = False


def _format_embedding(embedding: List[float]) -> str:
    """Format embedding list as pgvector-compatible string."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _get_summarizer():
    """Lazy-load the summarizer on first use."""
    global _summarizer, _summarizer_checked, SUMMARIZER_AVAILABLE

    if _summarizer_checked:
        return _summarizer

    _summarizer_checked = True
    try:
        from cathedral.Config import get as get_config
        from .loom_mirror import LoomMirror
        model_path = get_config(
            "LOOMMIRROR_MODEL_PATH",
            "./models/memory/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        )
        _summarizer = LoomMirror(
            model_path=model_path,
            model_name="TinyLlama-1.1B",
            n_ctx=2048
        )
        SUMMARIZER_AVAILABLE = True
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"LoomMirror not available: {e}")
        _summarizer = None
        SUMMARIZER_AVAILABLE = False

    return _summarizer


# Knowledge discovery integration (optional, may not be initialized)
_discovery_enabled = False
_discovery_queue_func = None


def enable_discovery(queue_func):
    """Enable knowledge discovery by providing the queue function."""
    global _discovery_enabled, _discovery_queue_func
    _discovery_enabled = True
    _discovery_queue_func = queue_func


def disable_discovery():
    """Disable knowledge discovery."""
    global _discovery_enabled, _discovery_queue_func
    _discovery_enabled = False
    _discovery_queue_func = None


def truncate_to_fit(messages: List[Dict], token_limit: int) -> List[Dict]:
    """Truncate messages to fit within token limit, keeping most recent."""
    result = []
    token_count = 0
    for message in reversed(messages):
        tokens = len(message.get('content', '').split())
        if token_count + tokens > token_limit:
            break
        result.insert(0, message)
        token_count += tokens
    return result


class ConversationService:
    """
    Main conversation service providing thread and message management.

    This replaces the legacy conversation functionality with a MemoryGate-integrated
    implementation using the same database patterns.
    """

    def __init__(self):
        """Initialize the conversation service."""
        self.active_thread_uid: Optional[str] = None

    def _resolve_thread_uid(self, thread_uid: Optional[str]) -> str:
        """Resolve thread UID, ensuring a default thread exists."""
        if thread_uid:
            return thread_uid
        if not self.active_thread_uid:
            self._ensure_default_thread()
        if not self.active_thread_uid:
            raise ValueError("No thread UID provided or active")
        return self.active_thread_uid

    def _ensure_default_thread(self):
        """Ensure a default thread exists and set it as active."""
        with get_session() as session:
            # Check for active thread
            active = session.execute(
                select(ConversationThread).where(ConversationThread.is_active.is_(True))
            ).scalar_one_or_none()

            if active:
                self.active_thread_uid = active.thread_uid
                return

            # Check for any thread named 'default'
            default = session.execute(
                select(ConversationThread).where(ConversationThread.thread_name == "default")
            ).scalar_one_or_none()

            if default:
                default.is_active = True
                self.active_thread_uid = default.thread_uid
                return

            # Create new default thread
            new_thread = ConversationThread(
                thread_uid=str(uuid.uuid4()),
                thread_name="default",
                is_active=True
            )
            session.add(new_thread)
            self.active_thread_uid = new_thread.thread_uid

    # ==========================================================================
    # Thread Management
    # ==========================================================================

    def create_thread(self, thread_name: str = "New Thread") -> str:
        """
        Create a new thread and set it as active.

        Args:
            thread_name: Name for the thread (will be made unique if exists)

        Returns:
            Thread UID
        """
        with get_session() as session:
            base_name = thread_name or "New Thread"
            final_name = base_name
            i = 1

            # Find unique name
            while True:
                existing = session.execute(
                    select(ConversationThread).where(ConversationThread.thread_name == final_name)
                ).scalar_one_or_none()
                if not existing:
                    break
                i += 1
                final_name = f"{base_name} {i}"

            # Deactivate all threads
            session.execute(update(ConversationThread).values(is_active=False))

            # Create new thread
            thread_uid = str(uuid.uuid4())
            new_thread = ConversationThread(
                thread_uid=thread_uid,
                thread_name=final_name,
                is_active=True
            )
            session.add(new_thread)

            self.active_thread_uid = thread_uid
            return thread_uid

    def switch_thread(self, thread_uid: str) -> bool:
        """
        Switch to an existing thread.

        Args:
            thread_uid: Thread UID to switch to

        Returns:
            True if successful, False if thread not found
        """
        with get_session() as session:
            thread = session.execute(
                select(ConversationThread).where(ConversationThread.thread_uid == thread_uid)
            ).scalar_one_or_none()

            if not thread:
                return False

            session.execute(update(ConversationThread).values(is_active=False))
            session.execute(
                update(ConversationThread)
                .where(ConversationThread.thread_uid == thread_uid)
                .values(is_active=True)
            )

            self.active_thread_uid = thread_uid
            return True

    def list_threads(self) -> List[Dict]:
        """List all threads, ordered by most recent first."""
        with get_session() as session:
            threads = session.execute(
                select(ConversationThread).order_by(ConversationThread.updated_at.desc())
            ).scalars().all()
            return [t.to_dict() for t in threads]

    def get_thread(self, thread_uid: str) -> Optional[Dict]:
        """Get a specific thread by UID."""
        with get_session() as session:
            thread = session.execute(
                select(ConversationThread).where(ConversationThread.thread_uid == thread_uid)
            ).scalar_one_or_none()
            return thread.to_dict() if thread else None

    def get_active_thread(self) -> Optional[Dict]:
        """Get the currently active thread."""
        with get_session() as session:
            thread = session.execute(
                select(ConversationThread).where(ConversationThread.is_active.is_(True))
            ).scalar_one_or_none()
            return thread.to_dict() if thread else None

    def get_active_thread_uid(self) -> str:
        """Get the active thread UID."""
        if not self.active_thread_uid:
            self._ensure_default_thread()
        return self.active_thread_uid

    def delete_thread(self, thread_uid: str) -> bool:
        """
        Delete a thread and all its messages.

        Args:
            thread_uid: Thread to delete

        Returns:
            True if deleted, False if not found
        """
        with get_session() as session:
            thread = session.execute(
                select(ConversationThread).where(ConversationThread.thread_uid == thread_uid)
            ).scalar_one_or_none()

            if not thread:
                return False

            was_active = thread.is_active
            session.delete(thread)

            # If deleted thread was active, activate another
            if was_active:
                other = session.execute(
                    select(ConversationThread).limit(1)
                ).scalar_one_or_none()
                if other:
                    other.is_active = True
                    self.active_thread_uid = other.thread_uid
                else:
                    self.active_thread_uid = None

            return True

    def rename_thread(self, thread_uid: str, new_name: str) -> bool:
        """Rename a thread."""
        with get_session() as session:
            result = session.execute(
                update(ConversationThread)
                .where(ConversationThread.thread_uid == thread_uid)
                .values(thread_name=new_name, updated_at=datetime.utcnow())
            )
            return result.rowcount > 0

    # ==========================================================================
    # Message Operations
    # ==========================================================================

    def recall(self, thread_uid: str = None) -> List[Dict]:
        """
        Get message history for a thread (sync).

        Args:
            thread_uid: Thread to recall from (uses active if None)

        Returns:
            List of message dicts
        """
        thread_uid = self._resolve_thread_uid(thread_uid)

        with get_session() as session:
            messages = session.execute(
                select(ConversationMessage)
                .where(ConversationMessage.thread_uid == thread_uid)
                .order_by(ConversationMessage.timestamp.asc())
            ).scalars().all()
            return [m.to_dict() for m in messages]

    async def recall_async(self, thread_uid: str = None) -> List[Dict]:
        """
        Get message history for a thread (async).

        Args:
            thread_uid: Thread to recall from (uses active if None)

        Returns:
            List of message dicts
        """
        thread_uid = self._resolve_thread_uid(thread_uid)

        async with get_async_session() as session:
            result = await session.execute(
                select(ConversationMessage)
                .options(selectinload(ConversationMessage.embedding))
                .where(ConversationMessage.thread_uid == thread_uid)
                .order_by(ConversationMessage.timestamp.asc())
            )
            messages = result.scalars().all()
            return [m.to_dict() for m in messages]

    def append(self, role: str, content: str, thread_uid: str = None) -> str:
        """
        Append a message to a thread (sync).

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            thread_uid: Target thread (uses active if None)

        Returns:
            Message UID
        """
        thread_uid = self._resolve_thread_uid(thread_uid)

        message_uid = str(uuid.uuid4())

        with get_session() as session:
            message = ConversationMessage(
                message_uid=message_uid,
                thread_uid=thread_uid,
                role=role,
                content=content
            )
            session.add(message)

            # Update thread timestamp
            session.execute(
                update(ConversationThread)
                .where(ConversationThread.thread_uid == thread_uid)
                .values(updated_at=datetime.utcnow())
            )

        return message_uid

    async def append_async(
        self,
        role: str,
        content: str,
        thread_uid: str = None,
        generate_embedding: bool = True
    ) -> str:
        """
        Append a message and optionally generate embedding (async).

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            thread_uid: Target thread (uses active if None)
            generate_embedding: Whether to generate embedding in background

        Returns:
            Message UID
        """
        thread_uid = self._resolve_thread_uid(thread_uid)

        message_uid = str(uuid.uuid4())

        async with get_async_session() as session:
            message = ConversationMessage(
                message_uid=message_uid,
                thread_uid=thread_uid,
                role=role,
                content=content
            )
            session.add(message)

            # Update thread timestamp
            await session.execute(
                update(ConversationThread)
                .where(ConversationThread.thread_uid == thread_uid)
                .values(updated_at=datetime.utcnow())
            )

        # Generate embedding asynchronously
        if generate_embedding and embeddings_configured():
            task = asyncio.create_task(self._embed_message(message_uid, thread_uid, content))
            task.add_done_callback(self._handle_task_exception)

        return message_uid

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        """Handle exceptions from background tasks to prevent silent failures."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            import logging
            logging.getLogger(__name__).error(f"Background task failed: {exc}", exc_info=exc)

    async def _embed_message(self, message_uid: str, thread_uid: str, content: str) -> None:
        """Generate and store embedding for a message, then queue discovery."""
        embedding = await embed_text(content)
        if embedding is None:
            return

        async with get_async_session() as session:
            msg_embedding = ConversationEmbedding(
                message_uid=message_uid,
                embedding=embedding
            )
            session.add(msg_embedding)

        # Queue for knowledge discovery (non-blocking)
        if _discovery_enabled and _discovery_queue_func:
            try:
                await _discovery_queue_func(message_uid, thread_uid, embedding)
            except Exception:
                # Don't let discovery failures affect message flow
                pass

    def clear(self, thread_uid: str = None) -> int:
        """
        Clear all messages from a thread.

        Args:
            thread_uid: Thread to clear (uses active if None)

        Returns:
            Number of messages deleted
        """
        thread_uid = self._resolve_thread_uid(thread_uid)

        with get_session() as session:
            result = session.execute(
                delete(ConversationMessage).where(ConversationMessage.thread_uid == thread_uid)
            )
            return result.rowcount

    # ==========================================================================
    # Semantic Search
    # ==========================================================================

    async def semantic_search(
        self,
        query: str,
        thread_uid: str = None,
        limit: int = 5,
        include_all_threads: bool = False
    ) -> List[Dict]:
        """
        Search messages semantically using vector similarity.

        Args:
            query: Search query
            thread_uid: Limit to specific thread (None for active)
            limit: Maximum results
            include_all_threads: Search across all threads

        Returns:
            Messages ranked by relevance
        """
        if not embeddings_configured():
            return []

        query_embedding = await embed_text(query)
        if query_embedding is None:
            return []

        async with get_async_session() as session:
            # Use raw SQL for pgvector operations
            # Embed vector directly in SQL since asyncpg CAST doesn't work well with vector type
            embedding_str = _format_embedding(query_embedding)
            if thread_uid and not include_all_threads:
                sql = text(f"""
                    SELECT m.message_uid, m.role, m.content, m.timestamp, m.thread_uid,
                           1 - (e.embedding <=> '{embedding_str}'::vector) as similarity
                    FROM mg_conversation_messages m
                    JOIN mg_conversation_embeddings e ON m.message_uid = e.message_uid
                    WHERE m.thread_uid = :thread_uid
                    ORDER BY e.embedding <=> '{embedding_str}'::vector
                    LIMIT :limit
                """)
                result = await session.execute(sql, {
                    "thread_uid": thread_uid,
                    "limit": limit
                })
            else:
                sql = text(f"""
                    SELECT m.message_uid, m.role, m.content, m.timestamp, m.thread_uid,
                           1 - (e.embedding <=> '{embedding_str}'::vector) as similarity
                    FROM mg_conversation_messages m
                    JOIN mg_conversation_embeddings e ON m.message_uid = e.message_uid
                    ORDER BY e.embedding <=> '{embedding_str}'::vector
                    LIMIT :limit
                """)
                result = await session.execute(sql, {
                    "limit": limit
                })

            rows = result.fetchall()
            return [
                {
                    "message_uid": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3].isoformat() if row[3] else None,
                    "thread_uid": row[4],
                    "similarity": float(row[5]) if row[5] else 0.0
                }
                for row in rows
            ]

    async def search_summaries(self, query: str, limit: int = 3) -> List[Dict]:
        """Search summaries semantically."""
        if not embeddings_configured():
            return []

        query_embedding = await embed_text(query)
        if query_embedding is None:
            return []

        async with get_async_session() as session:
            embedding_str = _format_embedding(query_embedding)
            sql = text(f"""
                SELECT id, thread_uid, summary_text, created_at,
                       1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM mg_conversation_summaries
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT :limit
            """)
            result = await session.execute(sql, {
                "limit": limit
            })

            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "thread_uid": row[1],
                    "summary_text": row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                    "similarity": float(row[4]) if row[4] else 0.0
                }
                for row in rows
            ]

    async def hybrid_search(
        self,
        query: str,
        thread_uid: str = None,
        limit: int = 5,
        include_all_threads: bool = False,
        fts_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> List[Dict]:
        """
        Hybrid search combining PostgreSQL full-text search with vector similarity.

        Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
        FTS catches keyword matches, semantic catches conceptual similarity.

        Falls back to semantic-only search on non-PostgreSQL backends.

        Args:
            query: Search query
            thread_uid: Limit to specific thread (None for active)
            limit: Maximum results
            include_all_threads: Search across all threads
            fts_weight: Weight for full-text search component (default 0.4)
            semantic_weight: Weight for semantic component (default 0.6)

        Returns:
            Messages ranked by combined relevance
        """
        # Check if we're on PostgreSQL (required for hybrid search)
        from cathedral.shared import db_service
        engine = db_service.get_engine()
        is_postgres = engine.dialect.name == "postgresql"

        if not is_postgres:
            # Fall back to semantic-only search on non-PostgreSQL
            return await self.semantic_search(query, thread_uid, limit, include_all_threads)

        if not embeddings_configured():
            # Fall back to FTS-only if embeddings not configured
            return await self._fts_search(query, thread_uid, limit, include_all_threads)

        query_embedding = await embed_text(query)
        if query_embedding is None:
            return await self._fts_search(query, thread_uid, limit, include_all_threads)

        async with get_async_session() as session:
            # Hybrid query using RRF scoring
            # RRF formula: score = Σ (weight / (k + rank))
            # k=60 is the standard smoothing constant
            # Embed vector directly in SQL since asyncpg CAST doesn't work well with vector type
            embedding_str = _format_embedding(query_embedding)

            if thread_uid and not include_all_threads:
                sql = text(f"""
                    WITH fts_results AS (
                        SELECT message_uid,
                               ROW_NUMBER() OVER (ORDER BY ts_rank(search_vector, plainto_tsquery('english', :query)) DESC) as fts_rank
                        FROM mg_conversation_messages
                        WHERE thread_uid = :thread_uid
                          AND search_vector @@ plainto_tsquery('english', :query)
                        LIMIT :search_limit
                    ),
                    semantic_results AS (
                        SELECT m.message_uid,
                               ROW_NUMBER() OVER (ORDER BY e.embedding <=> '{embedding_str}'::vector) as sem_rank
                        FROM mg_conversation_messages m
                        JOIN mg_conversation_embeddings e ON m.message_uid = e.message_uid
                        WHERE m.thread_uid = :thread_uid
                        LIMIT :search_limit
                    ),
                    combined AS (
                        SELECT COALESCE(f.message_uid, s.message_uid) as message_uid,
                               COALESCE(:fts_weight / (60.0 + f.fts_rank), 0) +
                               COALESCE(:sem_weight / (60.0 + s.sem_rank), 0) as rrf_score
                        FROM fts_results f
                        FULL OUTER JOIN semantic_results s ON f.message_uid = s.message_uid
                    )
                    SELECT m.message_uid, m.role, m.content, m.timestamp, m.thread_uid,
                           c.rrf_score
                    FROM combined c
                    JOIN mg_conversation_messages m ON c.message_uid = m.message_uid
                    ORDER BY c.rrf_score DESC
                    LIMIT :limit
                """)
                result = await session.execute(sql, {
                    "query": query,
                    "thread_uid": thread_uid,
                    "fts_weight": fts_weight,
                    "sem_weight": semantic_weight,
                    "search_limit": limit * 3,  # Get more candidates for fusion
                    "limit": limit
                })
            else:
                sql = text(f"""
                    WITH fts_results AS (
                        SELECT message_uid,
                               ROW_NUMBER() OVER (ORDER BY ts_rank(search_vector, plainto_tsquery('english', :query)) DESC) as fts_rank
                        FROM mg_conversation_messages
                        WHERE search_vector @@ plainto_tsquery('english', :query)
                        LIMIT :search_limit
                    ),
                    semantic_results AS (
                        SELECT m.message_uid,
                               ROW_NUMBER() OVER (ORDER BY e.embedding <=> '{embedding_str}'::vector) as sem_rank
                        FROM mg_conversation_messages m
                        JOIN mg_conversation_embeddings e ON m.message_uid = e.message_uid
                        LIMIT :search_limit
                    ),
                    combined AS (
                        SELECT COALESCE(f.message_uid, s.message_uid) as message_uid,
                               COALESCE(:fts_weight / (60.0 + f.fts_rank), 0) +
                               COALESCE(:sem_weight / (60.0 + s.sem_rank), 0) as rrf_score
                        FROM fts_results f
                        FULL OUTER JOIN semantic_results s ON f.message_uid = s.message_uid
                    )
                    SELECT m.message_uid, m.role, m.content, m.timestamp, m.thread_uid,
                           c.rrf_score
                    FROM combined c
                    JOIN mg_conversation_messages m ON c.message_uid = m.message_uid
                    ORDER BY c.rrf_score DESC
                    LIMIT :limit
                """)
                result = await session.execute(sql, {
                    "query": query,
                    "fts_weight": fts_weight,
                    "sem_weight": semantic_weight,
                    "search_limit": limit * 3,
                    "limit": limit
                })

            rows = result.fetchall()
            return [
                {
                    "message_uid": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3].isoformat() if row[3] else None,
                    "thread_uid": row[4],
                    "score": float(row[5]) if row[5] else 0.0,
                    "search_type": "hybrid"
                }
                for row in rows
            ]

    async def _fts_search(
        self,
        query: str,
        thread_uid: str = None,
        limit: int = 5,
        include_all_threads: bool = False
    ) -> List[Dict]:
        """Full-text search only fallback (PostgreSQL only)."""
        # FTS requires PostgreSQL
        from cathedral.shared import db_service
        engine = db_service.get_engine()
        if engine.dialect.name != "postgresql":
            return []

        async with get_async_session() as session:
            if thread_uid and not include_all_threads:
                sql = text("""
                    SELECT message_uid, role, content, timestamp, thread_uid,
                           ts_rank(search_vector, plainto_tsquery('english', :query)) as score
                    FROM mg_conversation_messages
                    WHERE thread_uid = :thread_uid
                      AND search_vector @@ plainto_tsquery('english', :query)
                    ORDER BY score DESC
                    LIMIT :limit
                """)
                result = await session.execute(sql, {
                    "query": query,
                    "thread_uid": thread_uid,
                    "limit": limit
                })
            else:
                sql = text("""
                    SELECT message_uid, role, content, timestamp, thread_uid,
                           ts_rank(search_vector, plainto_tsquery('english', :query)) as score
                    FROM mg_conversation_messages
                    WHERE search_vector @@ plainto_tsquery('english', :query)
                    ORDER BY score DESC
                    LIMIT :limit
                """)
                result = await session.execute(sql, {
                    "query": query,
                    "limit": limit
                })

            rows = result.fetchall()
            return [
                {
                    "message_uid": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3].isoformat() if row[3] else None,
                    "thread_uid": row[4],
                    "score": float(row[5]) if row[5] else 0.0,
                    "search_type": "fts"
                }
                for row in rows
            ]

    # ==========================================================================
    # Context Composition
    # ==========================================================================

    async def compose_context(
        self,
        user_input: str,
        thread_uid: str,
        preserve_last_n: int = 20,
        include_semantic: bool = True,
        semantic_limit: int = 3,
        max_tokens: int = 4096,
        use_hybrid_search: bool = True,
        use_context_gate: bool = True,
        gate_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Build enriched context for LLM prompt.

        Includes thread history with summarization and optional
        hybrid search results from other conversations.

        Args:
            user_input: Current user input (used for search)
            thread_uid: Thread to build context for
            preserve_last_n: Number of recent messages to keep verbatim
            include_semantic: Include search results from other conversations
            semantic_limit: Max search results
            max_tokens: Token budget for context
            use_hybrid_search: Use hybrid FTS+semantic search (default True)
            use_context_gate: Use ContextGate heuristics (default True)
            gate_threshold: ContextGate score threshold (default 0.5)

        Returns:
            Message list ready for LLM
        """
        from cathedral.MemoryGate.context_gate import ContextGate

        # Get thread history with summarization
        history = await self.recall_with_summary_async(
            preserve_last_n=preserve_last_n,
            thread_uid=thread_uid
        )

        result = []

        # Check if we should include search results
        should_search = include_semantic
        if should_search and use_context_gate:
            should_inject, score, reason = ContextGate.decide(user_input, threshold=gate_threshold)
            should_search = should_inject

        # Add search context if enabled and gating passed
        if should_search:
            if use_hybrid_search:
                search_results = await self.hybrid_search(
                    query=user_input,
                    thread_uid=thread_uid,
                    limit=semantic_limit,
                    include_all_threads=True
                )
                # Filter by score (RRF scores are typically small, use lower threshold)
                search_results = [r for r in search_results if r.get("score", 0) > 0.005]
            else:
                search_results = await self.semantic_search(
                    query=user_input,
                    thread_uid=thread_uid,
                    limit=semantic_limit,
                    include_all_threads=True
                )
                search_results = [r for r in search_results if r.get("similarity", 0) > 0.5]

            summary_results = await self.search_summaries(query=user_input, limit=2)
            summary_results = [s for s in summary_results if s.get("similarity", 0) > 0.5]

            context_lines = []
            for r in search_results:
                context_lines.append(f"- [{r['role']}] {r['content'][:200]}")

            for s in summary_results:
                context_lines.append(f"- [summary] {s['summary_text'][:200]}")

            if context_lines:
                result.append({
                    "role": "system",
                    "content": "[RELEVANT CONTEXT]\n" + "\n".join(context_lines)
                })

        # Add thread history
        result.extend(history)

        # Truncate to fit context window
        max_content_tokens = max_tokens - 256  # Leave room for response
        return truncate_to_fit(result, max_content_tokens)

    # ==========================================================================
    # Summarization
    # ==========================================================================

    def recall_with_summary(
        self,
        preserve_last_n: int = 20,
        thread_uid: str = None
    ) -> List[Dict]:
        """Get history with early messages summarized (sync)."""
        thread_uid = self._resolve_thread_uid(thread_uid)

        history = self.recall(thread_uid)

        if len(history) <= preserve_last_n:
            return history

        early = history[:-preserve_last_n]
        recent = history[-preserve_last_n:]

        summary = self._summarize_messages(early)
        summary_block = {
            "role": "system",
            "content": f"[EARLIER CONVERSATION SUMMARY]\n{summary}"
        }

        return [summary_block] + recent

    async def recall_with_summary_async(
        self,
        preserve_last_n: int = 20,
        thread_uid: str = None
    ) -> List[Dict]:
        """Get history with early messages summarized (async)."""
        thread_uid = self._resolve_thread_uid(thread_uid)

        history = await self.recall_async(thread_uid)

        if len(history) <= preserve_last_n:
            return history

        early = history[:-preserve_last_n]
        recent = history[-preserve_last_n:]

        summary = self._summarize_messages(early)
        summary_block = {
            "role": "system",
            "content": f"[EARLIER CONVERSATION SUMMARY]\n{summary}"
        }

        return [summary_block] + recent

    def _summarize_messages(self, messages: List[Dict]) -> str:
        """Summarize a list of messages."""
        if not messages:
            return "No earlier messages."

        summarizer = _get_summarizer()
        if summarizer is None:
            # Fallback: just show last few messages
            lines = [f"{m['role']}: {m['content'][:100]}" for m in messages[-5:]]
            return "Earlier conversation:\n" + "\n".join(lines)

        # Use local summarizer
        chunk_size = 10
        chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
        summaries = []

        for chunk in chunks:
            lines = [f"{m['role'].upper()}: {m['content']}" for m in chunk]
            dialogue_text = "\n".join(lines)
            prompt = f"""Summarize this conversation in 2-3 sentences:

{dialogue_text}

SUMMARY:"""
            try:
                summary = summarizer.run(prompt, max_tokens=128)
                summaries.append(summary.strip())
            except Exception:
                summaries.append(f"[{len(chunk)} messages]")

        return " ".join(summaries)

    async def create_summary(
        self,
        thread_uid: str,
        messages: List[Dict]
    ) -> Optional[int]:
        """Create and store a summary with embedding."""
        if not messages:
            return None

        summary_text = self._summarize_messages(messages)

        # Generate embedding for summary
        embedding = None
        if embeddings_configured():
            embedding = await embed_text(summary_text)

        async with get_async_session() as session:
            summary = ConversationSummary(
                thread_uid=thread_uid,
                summary_text=summary_text,
                message_count=len(messages),
                embedding=embedding
            )
            session.add(summary)
            await session.flush()
            return summary.id

    # ==========================================================================
    # Backfill Embeddings
    # ==========================================================================

    async def backfill_embeddings(self, batch_size: int = 50) -> int:
        """
        Generate embeddings for messages that don't have them.

        Args:
            batch_size: Number of messages per batch

        Returns:
            Number of embeddings generated
        """
        if not embeddings_configured():
            return 0

        async with get_async_session() as session:
            # Find messages without embeddings
            sql = text("""
                SELECT m.message_uid, m.content
                FROM mg_conversation_messages m
                LEFT JOIN mg_conversation_embeddings e ON m.message_uid = e.message_uid
                WHERE e.id IS NULL
                LIMIT :limit
            """)
            result = await session.execute(sql, {"limit": batch_size})
            rows = result.fetchall()

            if not rows:
                return 0

            # Generate embeddings in batch
            message_uids = [row[0] for row in rows]
            contents = [row[1] for row in rows]

            embeddings = await embed_text_batch(contents)

            # Store embeddings
            count = 0
            for message_uid, embedding in zip(message_uids, embeddings):
                if embedding is not None:
                    msg_embedding = ConversationEmbedding(
                        message_uid=message_uid,
                        embedding=embedding
                    )
                    session.add(msg_embedding)
                    count += 1

            return count

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        with get_session() as session:
            thread_count = session.execute(
                select(func.count(ConversationThread.id))
            ).scalar() or 0

            message_count = session.execute(
                select(func.count(ConversationMessage.id))
            ).scalar() or 0

            embedded_count = session.execute(
                select(func.count(ConversationEmbedding.id))
            ).scalar() or 0

            summary_count = session.execute(
                select(func.count(ConversationSummary.id))
            ).scalar() or 0

            # Trigger lazy check for summarizer availability
            _get_summarizer()

            return {
                "threads": thread_count,
                "messages": message_count,
                "embedded": embedded_count,
                "summaries": summary_count,
                "summarizer_available": SUMMARIZER_AVAILABLE,
                "embeddings_configured": embeddings_configured(),
            }


# ==========================================================================
# Module-level singleton and convenience functions
# ==========================================================================

_service: Optional[ConversationService] = None


def get_conversation_service() -> ConversationService:
    """Get or create global ConversationService instance."""
    global _service
    if _service is None:
        _service = ConversationService()
    return _service


def reset_service() -> None:
    """Reset the global service instance (for testing)."""
    global _service
    _service = None


# Export key classes and functions
__all__ = [
    "ConversationService",
    "get_conversation_service",
    "reset_service",
    "enable_discovery",
    "disable_discovery",
    "ConversationThread",
    "ConversationMessage",
    "ConversationEmbedding",
    "ConversationSummary",
    "init_conversation_db",
    "is_initialized",
    "embeddings_configured",
]
