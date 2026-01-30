"""
Knowledge Discovery Service

Discovers relationships between memory entities through embedding similarity.
Runs asynchronously in the background to avoid blocking the agent.

Discovery types:
- Messages → Observations (what facts relate to this message?)
- Messages → Patterns (what patterns does this message reflect?)
- Messages → Concepts (what concepts is this message about?)
- Messages ↔ Messages (similar discussions in other threads)
- Threads → Concepts (what is this thread about?)
- Threads → Messages (similar messages in other threads)
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _get_conversation_tables() -> Dict[str, str]:
    """
    Get conversation table names.

    Returns dict with keys: messages, embeddings, threads
    """
    return {
        "messages": "mg_conversation_messages",
        "embeddings": "mg_conversation_embeddings",
        "threads": "mg_conversation_threads",
    }


@dataclass
class DiscoveryConfig:
    """Configuration for knowledge discovery."""
    enabled: bool = True
    min_similarity: float = 0.6
    top_k: int = 5

    # Background processing
    thread_embedding_interval: int = 5  # Compute thread embedding every N messages
    discovery_batch_size: int = 10

    # Relationship metadata
    relationship_tag: str = "discovered"
    relationship_method: str = "embedding_similarity"


@dataclass
class DiscoveredRelationship:
    """A discovered relationship between entities."""
    from_ref: str
    to_ref: str
    rel_type: str
    similarity: float
    discovered_at: datetime


class KnowledgeDiscoveryService:
    """
    Discovers relationships through embedding similarity.

    Uses pgvector cosine distance to find related entities across:
    - Conversation tables (messages, threads, summaries)
    - MemoryGate tables (observations, patterns, concepts)

    Uses MemoryGate conversation tables.
    """

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._message_counts: Dict[str, int] = {}  # thread_uid -> message count since last thread embedding

    async def start(self):
        """Start the background discovery worker."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._worker())
        logger.info("[Discovery] Background worker started")

    async def stop(self):
        """Stop the background discovery worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Discovery] Background worker stopped")

    async def queue_message_discovery(
        self,
        message_uid: str,
        thread_uid: str,
        embedding: List[float]
    ):
        """Queue a message for background discovery."""
        await self._queue.put({
            "type": "message",
            "message_uid": message_uid,
            "thread_uid": thread_uid,
            "embedding": embedding
        })

        # Track message count for thread embedding updates
        count = self._message_counts.get(thread_uid, 0) + 1
        self._message_counts[thread_uid] = count

        # Queue thread embedding update if threshold reached
        if count >= self.config.thread_embedding_interval:
            await self._queue.put({
                "type": "thread_embedding",
                "thread_uid": thread_uid
            })
            self._message_counts[thread_uid] = 0

    async def queue_thread_discovery(self, thread_uid: str):
        """Queue a thread for discovery (uses thread embedding)."""
        await self._queue.put({
            "type": "thread",
            "thread_uid": thread_uid
        })

    async def _worker(self):
        """Background worker that processes the discovery queue."""
        while self._running:
            try:
                # Wait for item with timeout to allow clean shutdown
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if item["type"] == "message":
                    await self._discover_for_message(
                        item["message_uid"],
                        item["thread_uid"],
                        item["embedding"]
                    )
                elif item["type"] == "thread_embedding":
                    await self._update_thread_embedding(item["thread_uid"])
                elif item["type"] == "thread":
                    await self._discover_for_thread(item["thread_uid"])

                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"[Discovery] Worker error: {e}")

    async def _discover_for_message(
        self,
        message_uid: str,
        thread_uid: str,
        embedding: List[float]
    ):
        """
        Discover relationships for a message.

        Finds:
        - Similar observations
        - Similar patterns
        - Related concepts
        - Similar messages in OTHER threads
        """
        if not self.config.enabled:
            return

        from cathedral.shared.db import get_async_session
        from cathedral import MemoryGate

        message_ref = f"message:{message_uid}"
        relationships = []

        async with get_async_session() as session:
            # 1. Find similar observations
            obs_results = await self._search_memorygate_embeddings(
                session, embedding, "observation", self.config.top_k
            )
            for obs_id, similarity in obs_results:
                if similarity >= self.config.min_similarity:
                    relationships.append(DiscoveredRelationship(
                        from_ref=message_ref,
                        to_ref=f"observation:{obs_id}",
                        rel_type="relates_to",
                        similarity=similarity,
                        discovered_at=datetime.utcnow()
                    ))

            # 2. Find similar patterns
            pattern_results = await self._search_memorygate_embeddings(
                session, embedding, "pattern", self.config.top_k
            )
            for pattern_id, similarity in pattern_results:
                if similarity >= self.config.min_similarity:
                    relationships.append(DiscoveredRelationship(
                        from_ref=message_ref,
                        to_ref=f"pattern:{pattern_id}",
                        rel_type="reflects",
                        similarity=similarity,
                        discovered_at=datetime.utcnow()
                    ))

            # 3. Find related concepts
            concept_results = await self._search_memorygate_embeddings(
                session, embedding, "concept", self.config.top_k
            )
            for concept_id, similarity in concept_results:
                if similarity >= self.config.min_similarity:
                    relationships.append(DiscoveredRelationship(
                        from_ref=message_ref,
                        to_ref=f"concept:{concept_id}",
                        rel_type="about",
                        similarity=similarity,
                        discovered_at=datetime.utcnow()
                    ))

            # 4. Find similar messages in OTHER threads
            message_results = await self._search_conversation_messages(
                session, embedding, thread_uid, self.config.top_k
            )
            for other_uid, similarity in message_results:
                if similarity >= self.config.min_similarity:
                    relationships.append(DiscoveredRelationship(
                        from_ref=message_ref,
                        to_ref=f"message:{other_uid}",
                        rel_type="similar_to",
                        similarity=similarity,
                        discovered_at=datetime.utcnow()
                    ))

        # Store relationships in MemoryGate
        for rel in relationships:
            try:
                MemoryGate.add_relationship(
                    from_ref=rel.from_ref,
                    to_ref=rel.to_ref,
                    rel_type=rel.rel_type,
                    weight=rel.similarity,
                    description=f"Discovered via {self.config.relationship_method}"
                )
            except Exception as e:
                logger.debug(f"[Discovery] Failed to store relationship: {e}")

        if relationships:
            logger.info(f"[Discovery] Message {message_uid[:8]}: found {len(relationships)} relationships")

    async def _discover_for_thread(self, thread_uid: str):
        """
        Discover relationships for a thread using its aggregate embedding.

        Finds:
        - Related concepts
        - Similar messages in other threads
        """
        if not self.config.enabled:
            return

        from cathedral.shared.db import get_async_session
        from cathedral import MemoryGate

        async with get_async_session() as session:
            # Get thread embedding
            embedding = await self._get_thread_embedding(session, thread_uid)
            if not embedding:
                logger.debug(f"[Discovery] No embedding for thread {thread_uid[:8]}")
                return

            thread_ref = f"thread:{thread_uid}"
            relationships = []

            # 1. Find related concepts
            concept_results = await self._search_memorygate_embeddings(
                session, embedding, "concept", self.config.top_k
            )
            for concept_id, similarity in concept_results:
                if similarity >= self.config.min_similarity:
                    relationships.append(DiscoveredRelationship(
                        from_ref=thread_ref,
                        to_ref=f"concept:{concept_id}",
                        rel_type="about",
                        similarity=similarity,
                        discovered_at=datetime.utcnow()
                    ))

            # 2. Find similar messages in OTHER threads
            message_results = await self._search_conversation_messages(
                session, embedding, thread_uid, self.config.top_k
            )
            for other_uid, similarity in message_results:
                if similarity >= self.config.min_similarity:
                    relationships.append(DiscoveredRelationship(
                        from_ref=thread_ref,
                        to_ref=f"message:{other_uid}",
                        rel_type="similar_to",
                        similarity=similarity,
                        discovered_at=datetime.utcnow()
                    ))

        # Store relationships
        for rel in relationships:
            try:
                MemoryGate.add_relationship(
                    from_ref=rel.from_ref,
                    to_ref=rel.to_ref,
                    rel_type=rel.rel_type,
                    weight=rel.similarity,
                    description=f"Discovered via {self.config.relationship_method}"
                )
            except Exception as e:
                logger.debug(f"[Discovery] Failed to store relationship: {e}")

        if relationships:
            logger.info(f"[Discovery] Thread {thread_uid[:8]}: found {len(relationships)} relationships")

    async def _update_thread_embedding(self, thread_uid: str):
        """
        Compute and store thread-level embedding.

        Uses weighted average of message embeddings (recent messages weighted higher).
        """
        from cathedral.shared.db import get_async_session
        from cathedral import MemoryGate

        tables = _get_conversation_tables()

        async with get_async_session() as session:
            # Get all message embeddings for thread, ordered by timestamp
            query = text(f"""
                SELECT me.embedding
                FROM {tables['embeddings']} me
                JOIN {tables['messages']} m ON m.message_uid = me.message_uid
                WHERE m.thread_uid = :thread_uid
                ORDER BY m.timestamp ASC
            """)

            result = await session.execute(query, {"thread_uid": thread_uid})
            rows = result.fetchall()

            if not rows:
                return

            # Compute weighted average (recent messages weighted higher)
            embeddings = [row[0] for row in rows if row[0] is not None]
            if not embeddings:
                return

            # Weights: linear increase, recent = 2x oldest
            n = len(embeddings)
            if n == 0:
                return

            weights = [1.0 + (i / n) for i in range(n)]
            total_weight = sum(weights)

            # Safety check for division by zero (shouldn't happen but defensive)
            if total_weight == 0:
                return

            # Weighted average
            dim = len(embeddings[0])
            avg_embedding = [0.0] * dim
            for emb, weight in zip(embeddings, weights):
                for i in range(dim):
                    avg_embedding[i] += emb[i] * weight

            avg_embedding = [v / total_weight for v in avg_embedding]

            # Store in MemoryGate embeddings table
            # Use direct SQL since we need to handle upsert
            try:
                upsert_query = text("""
                    INSERT INTO embeddings (tenant_id, source_type, source_id, embedding, model_version, created_at)
                    VALUES ('cathedral', 'thread', :thread_uid, :embedding, 'text-embedding-3-small', NOW())
                    ON CONFLICT (tenant_id, source_type, source_id, model_version)
                    DO UPDATE SET embedding = :embedding, created_at = NOW()
                """)

                await session.execute(upsert_query, {
                    "thread_uid": thread_uid,
                    "embedding": avg_embedding
                })
                await session.commit()

                logger.info(f"[Discovery] Updated thread embedding for {thread_uid[:8]} ({n} messages)")
            except Exception as e:
                logger.error(f"[Discovery] Failed to store thread embedding: {e}")

    async def _get_thread_embedding(
        self,
        session: AsyncSession,
        thread_uid: str
    ) -> Optional[List[float]]:
        """Get stored thread embedding from MemoryGate."""
        query = text("""
            SELECT embedding
            FROM embeddings
            WHERE tenant_id = 'cathedral'
              AND source_type = 'thread'
              AND source_id = :thread_uid
        """)

        result = await session.execute(query, {"thread_uid": thread_uid})
        row = result.fetchone()

        if row and row[0]:
            return list(row[0])
        return None

    def _format_embedding_for_pgvector(self, embedding: List[float]) -> str:
        """Format embedding list as pgvector-compatible string."""
        return "[" + ",".join(str(v) for v in embedding) + "]"

    async def _search_memorygate_embeddings(
        self,
        session: AsyncSession,
        embedding: List[float],
        source_type: str,
        limit: int
    ) -> List[Tuple[int, float]]:
        """
        Search MemoryGate embeddings table for similar items.

        Returns list of (source_id, similarity) tuples.
        """
        embedding_str = self._format_embedding_for_pgvector(embedding)

        query = text("""
            SELECT source_id, 1 - (embedding <=> :embedding::vector) as similarity
            FROM embeddings
            WHERE tenant_id = 'cathedral'
              AND source_type = :source_type
              AND embedding IS NOT NULL
            ORDER BY embedding <=> :embedding::vector
            LIMIT :limit
        """)

        result = await session.execute(query, {
            "embedding": embedding_str,
            "source_type": source_type,
            "limit": limit
        })

        return [(int(row[0]), float(row[1])) for row in result.fetchall()]

    async def _search_conversation_messages(
        self,
        session: AsyncSession,
        embedding: List[float],
        exclude_thread_uid: str,
        limit: int
    ) -> List[Tuple[str, float]]:
        """
        Search conversation messages for similar content in OTHER threads.

        Returns list of (message_uid, similarity) tuples.
        Uses MemoryGate conversation tables.
        """
        tables = _get_conversation_tables()
        embedding_str = self._format_embedding_for_pgvector(embedding)

        query = text(f"""
            SELECT me.message_uid, 1 - (me.embedding <=> :embedding::vector) as similarity
            FROM {tables['embeddings']} me
            JOIN {tables['messages']} m ON m.message_uid = me.message_uid
            WHERE m.thread_uid != :exclude_thread_uid
              AND me.embedding IS NOT NULL
            ORDER BY me.embedding <=> :embedding::vector
            LIMIT :limit
        """)

        result = await session.execute(query, {
            "embedding": embedding_str,
            "exclude_thread_uid": exclude_thread_uid,
            "limit": limit
        })

        return [(row[0], float(row[1])) for row in result.fetchall()]

    # ==========================================
    # Manual Discovery API
    # ==========================================

    async def discover_now(
        self,
        ref: str,
        embedding: Optional[List[float]] = None
    ) -> List[DiscoveredRelationship]:
        """
        Run discovery immediately for a given reference.

        If embedding not provided, fetches it from storage.
        """
        ref_type, ref_id = ref.split(":", 1)

        from cathedral.shared.db import get_async_session

        async with get_async_session() as session:
            # Get embedding if not provided
            if embedding is None:
                embedding = await self._get_embedding_for_ref(session, ref_type, ref_id)
                if not embedding:
                    return []

            relationships = []

            # Determine what to search based on ref type
            if ref_type == "message":
                # Get thread_uid for the message
                thread_uid = await self._get_thread_for_message(session, ref_id)

                # Search observations
                for obs_id, sim in await self._search_memorygate_embeddings(
                    session, embedding, "observation", self.config.top_k
                ):
                    if sim >= self.config.min_similarity:
                        relationships.append(DiscoveredRelationship(
                            from_ref=ref, to_ref=f"observation:{obs_id}",
                            rel_type="relates_to", similarity=sim,
                            discovered_at=datetime.utcnow()
                        ))

                # Search patterns
                for pat_id, sim in await self._search_memorygate_embeddings(
                    session, embedding, "pattern", self.config.top_k
                ):
                    if sim >= self.config.min_similarity:
                        relationships.append(DiscoveredRelationship(
                            from_ref=ref, to_ref=f"pattern:{pat_id}",
                            rel_type="reflects", similarity=sim,
                            discovered_at=datetime.utcnow()
                        ))

                # Search concepts
                for concept_id, sim in await self._search_memorygate_embeddings(
                    session, embedding, "concept", self.config.top_k
                ):
                    if sim >= self.config.min_similarity:
                        relationships.append(DiscoveredRelationship(
                            from_ref=ref, to_ref=f"concept:{concept_id}",
                            rel_type="about", similarity=sim,
                            discovered_at=datetime.utcnow()
                        ))

                # Search other messages (cross-thread only)
                if thread_uid:
                    for msg_uid, sim in await self._search_conversation_messages(
                        session, embedding, thread_uid, self.config.top_k
                    ):
                        if sim >= self.config.min_similarity:
                            relationships.append(DiscoveredRelationship(
                                from_ref=ref, to_ref=f"message:{msg_uid}",
                                rel_type="similar_to", similarity=sim,
                                discovered_at=datetime.utcnow()
                            ))

            elif ref_type == "thread":
                # Search concepts
                for concept_id, sim in await self._search_memorygate_embeddings(
                    session, embedding, "concept", self.config.top_k
                ):
                    if sim >= self.config.min_similarity:
                        relationships.append(DiscoveredRelationship(
                            from_ref=ref, to_ref=f"concept:{concept_id}",
                            rel_type="about", similarity=sim,
                            discovered_at=datetime.utcnow()
                        ))

                # Search messages in other threads
                for msg_uid, sim in await self._search_conversation_messages(
                    session, embedding, ref_id, self.config.top_k
                ):
                    if sim >= self.config.min_similarity:
                        relationships.append(DiscoveredRelationship(
                            from_ref=ref, to_ref=f"message:{msg_uid}",
                            rel_type="similar_to", similarity=sim,
                            discovered_at=datetime.utcnow()
                        ))

            elif ref_type in ("observation", "pattern", "concept"):
                # Search messages
                for msg_uid, sim in await self._search_all_conversation_messages(
                    session, embedding, self.config.top_k
                ):
                    if sim >= self.config.min_similarity:
                        relationships.append(DiscoveredRelationship(
                            from_ref=ref, to_ref=f"message:{msg_uid}",
                            rel_type="mentioned_in",
                            similarity=sim,
                            discovered_at=datetime.utcnow()
                        ))

            return relationships

    async def _get_embedding_for_ref(
        self,
        session: AsyncSession,
        ref_type: str,
        ref_id: str
    ) -> Optional[List[float]]:
        """Get embedding for a reference."""
        tables = _get_conversation_tables()

        if ref_type == "message":
            query = text(f"""
                SELECT embedding FROM {tables['embeddings']}
                WHERE message_uid = :ref_id
            """)
        elif ref_type == "thread":
            query = text("""
                SELECT embedding FROM embeddings
                WHERE tenant_id = 'cathedral'
                  AND source_type = 'thread'
                  AND source_id = :ref_id
            """)
        else:
            query = text("""
                SELECT embedding FROM embeddings
                WHERE tenant_id = 'cathedral'
                  AND source_type = :ref_type
                  AND source_id = :ref_id
            """)

        result = await session.execute(query, {"ref_type": ref_type, "ref_id": ref_id})
        row = result.fetchone()

        if row and row[0]:
            return list(row[0])
        return None

    async def _get_thread_for_message(
        self,
        session: AsyncSession,
        message_uid: str
    ) -> Optional[str]:
        """Get thread_uid for a message."""
        tables = _get_conversation_tables()
        query = text(f"SELECT thread_uid FROM {tables['messages']} WHERE message_uid = :uid")
        result = await session.execute(query, {"uid": message_uid})
        row = result.fetchone()
        return row[0] if row else None

    async def _search_all_conversation_messages(
        self,
        session: AsyncSession,
        embedding: List[float],
        limit: int
    ) -> List[Tuple[str, float]]:
        """Search all conversation messages (no exclusion)."""
        tables = _get_conversation_tables()
        embedding_str = self._format_embedding_for_pgvector(embedding)

        query = text(f"""
            SELECT me.message_uid, 1 - (me.embedding <=> :embedding::vector) as similarity
            FROM {tables['embeddings']} me
            WHERE me.embedding IS NOT NULL
            ORDER BY me.embedding <=> :embedding::vector
            LIMIT :limit
        """)

        result = await session.execute(query, {"embedding": embedding_str, "limit": limit})
        return [(row[0], float(row[1])) for row in result.fetchall()]


# Global instance
_discovery_service: Optional[KnowledgeDiscoveryService] = None


def get_discovery_service(config: Optional[DiscoveryConfig] = None) -> KnowledgeDiscoveryService:
    """Get or create the global discovery service."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = KnowledgeDiscoveryService(config)
    return _discovery_service


async def start_discovery():
    """Start the background discovery worker."""
    service = get_discovery_service()
    await service.start()


async def stop_discovery():
    """Stop the background discovery worker."""
    service = get_discovery_service()
    await service.stop()


async def queue_message_discovery(message_uid: str, thread_uid: str, embedding: List[float]):
    """Queue a message for discovery."""
    service = get_discovery_service()
    await service.queue_message_discovery(message_uid, thread_uid, embedding)


async def discover_for_ref(ref: str) -> List[DiscoveredRelationship]:
    """Run discovery immediately for a reference."""
    service = get_discovery_service()
    return await service.discover_now(ref)
