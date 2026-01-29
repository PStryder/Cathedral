"""
Loom - Conversation memory and semantic search system.
PostgreSQL + pgvector backend with async embedding support.
"""

import os
import uuid
import time
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy import select, update, delete, func, text
from sqlalchemy.orm import Session

from loom.db import init_db, get_session, get_async_session, LoomDB, is_initialized
from loom.models import Thread, Message, MessageEmbedding, Fact, Tag, Summary, EMBEDDING_DIM
from loom.embeddings import embed_text, embed_texts, embed_text_batch, is_configured as embeddings_configured
from loom.LoomMirror import LoomMirror

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

# Initialize LoomMirror for local summarization (optional, may fail if model not present)
try:
    model_path = os.environ.get(
        "LOOMMIRROR_MODEL_PATH",
        "./models/memory/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    )
    memory_llm = LoomMirror(
        model_path=model_path,
        model_name="TinyLlama-1.1B",
        n_ctx=2048
    )
    LOOM_MIRROR_AVAILABLE = True
except Exception as e:
    print(f"[Loom] LoomMirror not available: {e}")
    memory_llm = None
    LOOM_MIRROR_AVAILABLE = False


def estimate_tokens(messages: List[Dict]) -> int:
    """Estimate token count from messages."""
    return sum(len(m.get('content', '').split()) for m in messages)


def truncate_to_fit(messages: List[Dict], token_limit: int) -> List[Dict]:
    """Truncate messages to fit within token limit."""
    result = []
    token_count = 0
    for message in reversed(messages):
        tokens = len(message.get('content', '').split())
        if token_count + tokens > token_limit:
            break
        result.insert(0, message)
        token_count += tokens
    return result


class Loom:
    """
    Main Loom interface for conversation memory.
    Provides both sync and async methods.
    """

    def __init__(self):
        """Initialize Loom with database connection."""
        init_db()
        self.active_thread_uid: Optional[str] = None
        self._ensure_default_thread()

    def _ensure_default_thread(self):
        """Ensure a default thread exists and set it as active."""
        with get_session() as session:
            # Check for active thread
            active = session.execute(
                select(Thread).where(Thread.is_active == True)
            ).scalar_one_or_none()

            if active:
                self.active_thread_uid = active.thread_uid
                return

            # Check for any thread named 'default'
            default = session.execute(
                select(Thread).where(Thread.thread_name == "default")
            ).scalar_one_or_none()

            if default:
                default.is_active = True
                session.commit()
                self.active_thread_uid = default.thread_uid
                return

            # Create new default thread
            new_thread = Thread(
                thread_uid=str(uuid.uuid4()),
                thread_name="default",
                is_active=True
            )
            session.add(new_thread)
            session.commit()
            self.active_thread_uid = new_thread.thread_uid

    # ==================== Thread Management ====================

    def create_new_thread(self, thread_name: str = "New Thread") -> str:
        """Create a new thread and set it as active."""
        with get_session() as session:
            base_name = thread_name or "New Thread"
            final_name = base_name
            i = 1

            # Find unique name
            while True:
                existing = session.execute(
                    select(Thread).where(Thread.thread_name == final_name)
                ).scalar_one_or_none()
                if not existing:
                    break
                i += 1
                final_name = f"{base_name} {i}"

            # Deactivate all threads
            session.execute(update(Thread).values(is_active=False))

            # Create new thread
            thread_uid = str(uuid.uuid4())
            new_thread = Thread(
                thread_uid=thread_uid,
                thread_name=final_name,
                is_active=True
            )
            session.add(new_thread)
            session.commit()

            self.active_thread_uid = thread_uid
            return thread_uid

    def switch_to_thread(self, thread_uid: str) -> None:
        """Switch to an existing thread."""
        with get_session() as session:
            session.execute(update(Thread).values(is_active=False))
            session.execute(
                update(Thread).where(Thread.thread_uid == thread_uid).values(is_active=True)
            )
            session.commit()
            self.active_thread_uid = thread_uid

    def list_all_threads(self) -> List[Dict]:
        """List all threads."""
        with get_session() as session:
            threads = session.execute(
                select(Thread).order_by(Thread.created_at.desc())
            ).scalars().all()
            return [t.to_dict() for t in threads]

    def get_active_thread_metadata(self) -> Dict:
        """Get metadata for the active thread."""
        with get_session() as session:
            thread = session.execute(
                select(Thread).where(Thread.is_active == True)
            ).scalar_one_or_none()
            if not thread:
                raise ValueError("No active thread found")
            return {"thread_uid": thread.thread_uid, "thread_name": thread.thread_name}

    # ==================== Message Operations ====================

    def recall(self, thread_uid: str = None) -> List[Dict]:
        """Get message history for a thread (sync)."""
        if not thread_uid:
            thread_uid = self.active_thread_uid
        if not thread_uid:
            raise ValueError("No thread UID provided or active")

        with get_session() as session:
            messages = session.execute(
                select(Message)
                .where(Message.thread_uid == thread_uid)
                .order_by(Message.timestamp.asc())
            ).scalars().all()
            return [m.to_dict() for m in messages]

    async def recall_async(self, thread_uid: str = None) -> List[Dict]:
        """Get message history for a thread (async)."""
        if not thread_uid:
            thread_uid = self.active_thread_uid
        if not thread_uid:
            raise ValueError("No thread UID provided or active")

        async with await get_async_session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.thread_uid == thread_uid)
                .order_by(Message.timestamp.asc())
            )
            messages = result.scalars().all()
            return [m.to_dict() for m in messages]

    def append(self, role: str, content: str, thread_uid: str = None) -> str:
        """Append a message to a thread (sync). Returns message_uid."""
        if not thread_uid:
            thread_uid = self.active_thread_uid
        if not thread_uid:
            raise ValueError("No thread UID provided or active")

        message_uid = str(uuid.uuid4())

        with get_session() as session:
            message = Message(
                message_uid=message_uid,
                thread_uid=thread_uid,
                role=role,
                content=content
            )
            session.add(message)
            session.commit()

        return message_uid

    async def append_async(self, role: str, content: str, thread_uid: str = None) -> str:
        """Append a message and generate embedding (async). Returns message_uid."""
        if not thread_uid:
            thread_uid = self.active_thread_uid
        if not thread_uid:
            raise ValueError("No thread UID provided or active")

        message_uid = str(uuid.uuid4())

        async with await get_async_session() as session:
            message = Message(
                message_uid=message_uid,
                thread_uid=thread_uid,
                role=role,
                content=content
            )
            session.add(message)
            await session.commit()

        # Generate embedding asynchronously (also triggers discovery)
        if embeddings_configured():
            asyncio.create_task(self._embed_message(message_uid, thread_uid, content))

        return message_uid

    async def _embed_message(self, message_uid: str, thread_uid: str, content: str) -> None:
        """Generate and store embedding for a message, then queue for discovery."""
        embedding = await embed_text(content)
        if embedding is None:
            return

        async with await get_async_session() as session:
            msg_embedding = MessageEmbedding(
                message_uid=message_uid,
                embedding=embedding
            )
            session.add(msg_embedding)
            await session.commit()

        # Queue for knowledge discovery (non-blocking)
        if _discovery_enabled and _discovery_queue_func:
            try:
                await _discovery_queue_func(message_uid, thread_uid, embedding)
            except Exception as e:
                # Don't let discovery failures affect message flow
                pass

    def clear(self, thread_uid: str = None) -> None:
        """Clear all messages from a thread."""
        if not thread_uid:
            thread_uid = self.active_thread_uid
        if not thread_uid:
            raise ValueError("No thread UID provided or active")

        with get_session() as session:
            session.execute(delete(Message).where(Message.thread_uid == thread_uid))
            session.commit()

    # ==================== Semantic Search ====================

    async def semantic_search(
        self,
        query: str,
        thread_uid: str = None,
        limit: int = 5,
        include_all_threads: bool = False
    ) -> List[Dict]:
        """
        Search messages semantically using vector similarity.
        Returns messages ranked by relevance.
        """
        if not embeddings_configured():
            return []

        query_embedding = await embed_text(query)
        if query_embedding is None:
            return []

        async with await get_async_session() as session:
            # Build query with pgvector cosine distance
            embedding_col = MessageEmbedding.embedding

            # Use raw SQL for vector operations
            if thread_uid and not include_all_threads:
                sql = text("""
                    SELECT m.message_uid, m.role, m.content, m.timestamp,
                           1 - (e.embedding <=> :query_embedding::vector) as similarity
                    FROM loom_messages m
                    JOIN loom_message_embeddings e ON m.message_uid = e.message_uid
                    WHERE m.thread_uid = :thread_uid
                    ORDER BY e.embedding <=> :query_embedding::vector
                    LIMIT :limit
                """)
                result = await session.execute(sql, {
                    "query_embedding": str(query_embedding),
                    "thread_uid": thread_uid,
                    "limit": limit
                })
            else:
                sql = text("""
                    SELECT m.message_uid, m.role, m.content, m.timestamp, m.thread_uid,
                           1 - (e.embedding <=> :query_embedding::vector) as similarity
                    FROM loom_messages m
                    JOIN loom_message_embeddings e ON m.message_uid = e.message_uid
                    ORDER BY e.embedding <=> :query_embedding::vector
                    LIMIT :limit
                """)
                result = await session.execute(sql, {
                    "query_embedding": str(query_embedding),
                    "limit": limit
                })

            rows = result.fetchall()
            return [
                {
                    "message_uid": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3].isoformat() if row[3] else None,
                    "thread_uid": row[4] if len(row) > 4 else thread_uid,
                    "similarity": float(row[-1]) if row[-1] else 0.0
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

        async with await get_async_session() as session:
            sql = text("""
                SELECT id, thread_uid, summary_text, created_at,
                       1 - (embedding <=> :query_embedding::vector) as similarity
                FROM loom_summaries
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :query_embedding::vector
                LIMIT :limit
            """)
            result = await session.execute(sql, {
                "query_embedding": str(query_embedding),
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

    # ==================== Context Building ====================

    async def compose_prompt_context_async(
        self,
        user_input: str,
        thread_uid: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Build enriched context for LLM prompt (async).
        Includes thread history, semantic search results, and summaries.
        """
        # Get thread history with summarization
        history = await self.recall_with_summary_async(thread_uid=thread_uid)

        # Search for semantically relevant messages
        semantic_results = await self.semantic_search(
            query=user_input,
            thread_uid=thread_uid,
            limit=top_k,
            include_all_threads=True  # Search across all threads for relevant context
        )

        # Search summaries too
        summary_results = await self.search_summaries(query=user_input, limit=2)

        # Build context
        result = []

        # Add semantic context if found
        if semantic_results or summary_results:
            context_lines = ["[RELEVANT CONTEXT]"]

            for r in semantic_results:
                if r.get("similarity", 0) > 0.5:  # Only include if reasonably similar
                    context_lines.append(f"- [{r['role']}] {r['content'][:200]}")

            for s in summary_results:
                if s.get("similarity", 0) > 0.5:
                    context_lines.append(f"- [summary] {s['summary_text'][:200]}")

            if len(context_lines) > 1:
                result.append({"role": "system", "content": "\n".join(context_lines)})

        # Add thread history
        result.extend(history)

        # Truncate to fit context window
        max_tokens = 4096 - 256  # Leave room for response
        return truncate_to_fit(result, max_tokens)

    def compose_prompt_context(self, user_input: str, thread_uid: str, top_k: int = 3) -> List[Dict]:
        """Build enriched context (sync wrapper)."""
        # For sync usage, use basic recall without semantic search
        history = self.recall_with_summary(thread_uid=thread_uid)
        max_tokens = 4096 - 256
        return truncate_to_fit(history, max_tokens)

    # ==================== Summarization ====================

    def recall_with_summary(self, preserve_last_n: int = 20, thread_uid: str = None) -> List[Dict]:
        """Get history with early messages summarized (sync)."""
        if not thread_uid:
            thread_uid = self.active_thread_uid

        history = self.recall(thread_uid)

        if len(history) <= preserve_last_n:
            return history

        early = history[:-preserve_last_n]
        recent = history[-preserve_last_n:]

        summary = self._summarize_messages(early)
        summary_block = {"role": "system", "content": f"[EARLIER CONVERSATION SUMMARY]\n{summary}"}

        return [summary_block] + recent

    async def recall_with_summary_async(self, preserve_last_n: int = 20, thread_uid: str = None) -> List[Dict]:
        """Get history with early messages summarized (async)."""
        if not thread_uid:
            thread_uid = self.active_thread_uid

        history = await self.recall_async(thread_uid)

        if len(history) <= preserve_last_n:
            return history

        early = history[:-preserve_last_n]
        recent = history[-preserve_last_n:]

        summary = self._summarize_messages(early)
        summary_block = {"role": "system", "content": f"[EARLIER CONVERSATION SUMMARY]\n{summary}"}

        return [summary_block] + recent

    def _summarize_messages(self, messages: List[Dict]) -> str:
        """Summarize a list of messages using LoomMirror."""
        if not LOOM_MIRROR_AVAILABLE or not messages:
            # Fallback: just concatenate last few
            lines = [f"{m['role']}: {m['content'][:100]}" for m in messages[-5:]]
            return "Earlier conversation:\n" + "\n".join(lines)

        # Use LoomMirror for summarization
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
                summary = memory_llm.run(prompt, max_tokens=128)
                summaries.append(summary.strip())
            except Exception as e:
                summaries.append(f"[{len(chunk)} messages]")

        return " ".join(summaries)

    async def create_summary(self, thread_uid: str, messages: List[Dict]) -> Optional[int]:
        """Create and store a summary with embedding."""
        if not messages:
            return None

        summary_text = self._summarize_messages(messages)

        # Generate embedding for summary
        embedding = None
        if embeddings_configured():
            embedding = await embed_text(summary_text)

        async with await get_async_session() as session:
            summary = Summary(
                thread_uid=thread_uid,
                summary_text=summary_text,
                embedding=embedding
            )
            session.add(summary)
            await session.commit()
            return summary.id

    # ==================== Backfill Embeddings ====================

    async def backfill_embeddings(self, batch_size: int = 50) -> int:
        """Generate embeddings for messages that don't have them."""
        if not embeddings_configured():
            return 0

        async with await get_async_session() as session:
            # Find messages without embeddings
            sql = text("""
                SELECT m.message_uid, m.content
                FROM loom_messages m
                LEFT JOIN loom_message_embeddings e ON m.message_uid = e.message_uid
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
                    msg_embedding = MessageEmbedding(
                        message_uid=message_uid,
                        embedding=embedding
                    )
                    session.add(msg_embedding)
                    count += 1

            await session.commit()
            return count

    # ==================== Utility ====================

    def set_active_thread_uid(self, thread_uid: str) -> None:
        """Set the active thread UID."""
        self.active_thread_uid = thread_uid

    def get_active_thread_uid(self) -> str:
        """Get the active thread UID."""
        if not self.active_thread_uid:
            raise ValueError("No active thread UID set")
        return self.active_thread_uid
