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
    min_similarity_cross_thread: float = 0.7  # Higher bar for message→message across threads
    top_k: int = 5

    # Background processing
    thread_embedding_interval: int = 5  # Compute thread embedding every N messages
    discovery_batch_size: int = 10

    # Relationship metadata
    relationship_tag: str = "discovered"
    relationship_method: str = "embedding_similarity"

    # Hybrid search settings
    use_hybrid_search: bool = True
    fts_weight: float = 0.4
    semantic_weight: float = 0.6
    overlap_boost: float = 1.5  # Multiplier when item appears in both FTS and semantic results
    rrf_threshold: float = 0.008  # Higher RRF threshold for cross-thread (was 0.005)

    # Profile domain boosting
    profile_domain_boost: float = 1.3  # Boost for domain='profile' in searches

    # Anchor-based expansion
    anchor_neighbor_limit: int = 3  # Max neighbors to pull for dual-hit anchors
    non_anchor_neighbor_limit: int = 1  # Max neighbors for single-hit results


@dataclass
class DiscoveredRelationship:
    """A discovered relationship between entities."""
    from_ref: str
    to_ref: str
    rel_type: str
    similarity: float
    discovered_at: datetime


@dataclass
class HybridSearchResult:
    """Result from hybrid search with anchor metadata."""
    message_uid: str
    score: float
    is_anchor: bool  # True if item appeared in both FTS and semantic results
    fts_rank: Optional[int] = None
    semantic_rank: Optional[int] = None


class KnowledgeDiscoveryService:
    """
    Discovers relationships through embedding similarity.

    Uses pgvector cosine distance to find related entities across:
    - Conversation tables (messages, threads, summaries)
    - MemoryGate tables (observations, patterns, concepts)

    Supports hybrid search (BM25/FTS + embedding similarity) with overlap boosting
    for cross-thread message discovery when PostgreSQL is available.

    Uses MemoryGate conversation tables.
    """

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._message_counts: Dict[str, int] = {}  # thread_uid -> message count since last thread embedding
        self._is_postgres: Optional[bool] = None  # Cached PostgreSQL detection

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
        embedding: List[float],
        content: Optional[str] = None
    ):
        """
        Queue a message for background discovery.

        Args:
            message_uid: The message UID
            thread_uid: The thread containing the message
            embedding: The message's embedding vector
            content: Optional message content (enables hybrid search without re-fetching)
        """
        await self._queue.put({
            "type": "message",
            "message_uid": message_uid,
            "thread_uid": thread_uid,
            "embedding": embedding,
            "content": content
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
                        item["embedding"],
                        item.get("content")
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
        embedding: List[float],
        content: Optional[str] = None
    ):
        """
        Discover relationships for a message.

        Finds:
        - Similar observations
        - Similar patterns
        - Related concepts
        - Similar messages in OTHER threads (uses hybrid search when available)

        Args:
            message_uid: The message to discover relationships for
            thread_uid: The thread containing the message
            embedding: The message's embedding vector
            content: Optional message content for hybrid FTS search
        """
        if not self.config.enabled:
            return

        from cathedral.shared.db import get_async_session
        from cathedral import MemoryGate

        message_ref = f"message:{message_uid}"
        relationships = []

        async with get_async_session() as session:
            # Get message content if not provided (needed for hybrid search)
            query_text = ""
            if content:
                query_text = self._extract_query_text(content)
            elif self.config.use_hybrid_search:
                # Fetch content from database for hybrid search
                tables = _get_conversation_tables()
                content_query = text(f"""
                    SELECT content FROM {tables['messages']}
                    WHERE message_uid = :message_uid
                """)
                result = await session.execute(content_query, {"message_uid": message_uid})
                row = result.fetchone()
                if row and row[0]:
                    query_text = self._extract_query_text(row[0])

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
            # Use hybrid search (FTS + vector with overlap boost) when configured and available
            if self.config.use_hybrid_search and query_text:
                message_results = await self._search_conversation_messages_hybrid(
                    session, embedding, query_text, thread_uid, self.config.top_k
                )
                discovery_method = "hybrid_rrf"
            else:
                message_results = await self._search_conversation_messages(
                    session, embedding, thread_uid, self.config.top_k
                )
                discovery_method = "embedding_similarity"

            for other_uid, score in message_results:
                # For hybrid search, scores are RRF-based (typically 0.005-0.02 range)
                # For vector search, scores are similarity (0-1 range)
                # Use higher threshold for cross-thread message links
                if discovery_method == "hybrid_rrf":
                    # RRF threshold from config (default 0.008)
                    if score >= self.config.rrf_threshold:
                        relationships.append(DiscoveredRelationship(
                            from_ref=message_ref,
                            to_ref=f"message:{other_uid}",
                            rel_type="similar_to",
                            similarity=score,
                            discovered_at=datetime.utcnow()
                        ))
                else:
                    # Vector similarity: use higher cross-thread threshold
                    if score >= self.config.min_similarity_cross_thread:
                        relationships.append(DiscoveredRelationship(
                            from_ref=message_ref,
                            to_ref=f"message:{other_uid}",
                            rel_type="similar_to",
                            similarity=score,
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
        - Similar messages in other threads (uses hybrid search when available)
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

            # Get representative query text from recent messages for hybrid search
            query_text = ""
            if self.config.use_hybrid_search:
                tables = _get_conversation_tables()
                # Get last few messages to build a representative query
                recent_query = text(f"""
                    SELECT content FROM {tables['messages']}
                    WHERE thread_uid = :thread_uid
                    ORDER BY timestamp DESC
                    LIMIT 3
                """)
                result = await session.execute(recent_query, {"thread_uid": thread_uid})
                rows = result.fetchall()
                if rows:
                    # Combine recent messages for FTS query
                    combined = " ".join(row[0] for row in rows if row[0])
                    query_text = self._extract_query_text(combined)

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
            # Use hybrid search when configured and we have query text
            if self.config.use_hybrid_search and query_text:
                message_results = await self._search_conversation_messages_hybrid(
                    session, embedding, query_text, thread_uid, self.config.top_k
                )
                discovery_method = "hybrid_rrf"
            else:
                message_results = await self._search_conversation_messages(
                    session, embedding, thread_uid, self.config.top_k
                )
                discovery_method = "embedding_similarity"

            for other_uid, score in message_results:
                # Apply appropriate threshold based on search method
                # Use higher threshold for cross-thread links
                if discovery_method == "hybrid_rrf":
                    if score >= self.config.rrf_threshold:
                        relationships.append(DiscoveredRelationship(
                            from_ref=thread_ref,
                            to_ref=f"message:{other_uid}",
                            rel_type="similar_to",
                            similarity=score,
                            discovered_at=datetime.utcnow()
                        ))
                else:
                    if score >= self.config.min_similarity_cross_thread:
                        relationships.append(DiscoveredRelationship(
                            from_ref=thread_ref,
                            to_ref=f"message:{other_uid}",
                            rel_type="similar_to",
                            similarity=score,
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

    def _is_postgresql(self) -> bool:
        """Check if the database backend is PostgreSQL (cached)."""
        if self._is_postgres is None:
            try:
                from cathedral.shared import db_service
                engine = db_service.get_engine()
                self._is_postgres = engine.dialect.name == "postgresql"
            except Exception:
                self._is_postgres = False
        return self._is_postgres

    def _extract_query_text(self, content: str) -> str:
        """
        Extract searchable text from message content.

        Handles various content formats and extracts meaningful text
        for full-text search queries.
        """
        if not content:
            return ""

        # If content is already plain text, use it directly
        # Truncate very long content for FTS efficiency
        text = content.strip()

        # Remove common prefixes that don't add search value
        prefixes_to_remove = [
            "[SYSTEM]", "[USER]", "[ASSISTANT]",
            "[EARLIER CONVERSATION SUMMARY]", "[RELEVANT CONTEXT]"
        ]
        for prefix in prefixes_to_remove:
            if text.upper().startswith(prefix.upper()):
                text = text[len(prefix):].strip()

        # Truncate for FTS (long text rarely improves search)
        max_fts_length = 1000
        if len(text) > max_fts_length:
            text = text[:max_fts_length]

        return text

    async def _search_memorygate_embeddings(
        self,
        session: AsyncSession,
        embedding: List[float],
        source_type: str,
        limit: int,
        boost_profile_domain: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Search MemoryGate embeddings table for similar items.

        Args:
            session: Database session
            embedding: Query embedding vector
            source_type: Type of source to search (observation, pattern, concept)
            limit: Maximum results
            boost_profile_domain: Apply boost to items with domain='profile'

        Returns:
            List of (source_id, similarity) tuples, with profile items boosted.
        """
        embedding_str = self._format_embedding_for_pgvector(embedding)

        # For observations, we can boost profile domain items
        if source_type == "observation" and boost_profile_domain and self._is_postgresql():
            # Join with observations table to get domain, apply boost
            query = text(f"""
                SELECT e.source_id,
                       (1 - (e.embedding <=> '{embedding_str}'::vector)) *
                       CASE WHEN o.domain = 'profile' THEN :profile_boost ELSE 1.0 END as similarity
                FROM embeddings e
                LEFT JOIN observations o ON e.source_id::text = o.id::text
                WHERE e.tenant_id = 'cathedral'
                  AND e.source_type = :source_type
                  AND e.embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT :limit
            """)

            result = await session.execute(query, {
                "source_type": source_type,
                "limit": limit,
                "profile_boost": self.config.profile_domain_boost
            })
        else:
            # Standard search without profile boost
            query = text(f"""
                SELECT source_id, 1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM embeddings
                WHERE tenant_id = 'cathedral'
                  AND source_type = :source_type
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT :limit
            """)

            result = await session.execute(query, {
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
            SELECT me.message_uid, 1 - (me.embedding <=> '{embedding_str}'::vector) as similarity
            FROM {tables['embeddings']} me
            JOIN {tables['messages']} m ON m.message_uid = me.message_uid
            WHERE m.thread_uid != :exclude_thread_uid
              AND me.embedding IS NOT NULL
            ORDER BY me.embedding <=> '{embedding_str}'::vector
            LIMIT :limit
        """)

        result = await session.execute(query, {
            "exclude_thread_uid": exclude_thread_uid,
            "limit": limit
        })

        return [(row[0], float(row[1])) for row in result.fetchall()]

    async def _search_conversation_messages_hybrid(
        self,
        session: AsyncSession,
        embedding: List[float],
        query_text: str,
        exclude_thread_uid: str,
        limit: int,
        fts_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        overlap_boost: Optional[float] = None,
        return_anchors: bool = False,
    ) -> List[Any]:  # Returns List[Tuple[str, float]] or List[HybridSearchResult]
        """
        Hybrid search for conversation messages using FTS + vector similarity with RRF.

        Combines PostgreSQL full-text search with pgvector semantic similarity,
        applying overlap boost for items appearing in both result sets.

        Falls back to vector-only search if:
        - Not PostgreSQL
        - No query text provided
        - FTS query fails

        Args:
            session: Database session
            embedding: Query embedding vector
            query_text: Original query text for FTS
            exclude_thread_uid: Thread to exclude from results
            limit: Maximum results to return
            fts_weight: Weight for FTS component (default from config)
            semantic_weight: Weight for semantic component (default from config)
            overlap_boost: Multiplier for items in both result sets (default from config)
            return_anchors: If True, return HybridSearchResult with anchor metadata

        Returns:
            List of (message_uid, score) tuples, or HybridSearchResult objects if return_anchors=True
        """
        # Use config defaults if not specified
        fts_weight = fts_weight or self.config.fts_weight
        semantic_weight = semantic_weight or self.config.semantic_weight
        overlap_boost = overlap_boost or self.config.overlap_boost

        # Fallback conditions: not PostgreSQL or no query text
        if not self._is_postgresql() or not query_text or not query_text.strip():
            results = await self._search_conversation_messages(
                session, embedding, exclude_thread_uid, limit
            )
            if return_anchors:
                return [HybridSearchResult(
                    message_uid=uid, score=score, is_anchor=False
                ) for uid, score in results]
            return results

        tables = _get_conversation_tables()
        embedding_str = self._format_embedding_for_pgvector(embedding)

        # Clean query text for FTS (plainto_tsquery handles most escaping)
        clean_query = query_text.strip()

        # Hybrid RRF query with overlap boost
        # k=60 is the standard RRF smoothing constant
        # search_limit is larger than limit to get good candidate pool for fusion
        search_limit = limit * 3

        # Updated query returns anchor info (is_dual_hit) and individual ranks
        query = text(f"""
            WITH fts_results AS (
                SELECT m.message_uid,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank(m.search_vector, plainto_tsquery('english', :query)) DESC
                       ) as fts_rank
                FROM {tables['messages']} m
                WHERE m.thread_uid != :exclude_thread_uid
                  AND m.search_vector @@ plainto_tsquery('english', :query)
                LIMIT :search_limit
            ),
            semantic_results AS (
                SELECT m.message_uid,
                       ROW_NUMBER() OVER (
                           ORDER BY e.embedding <=> '{embedding_str}'::vector
                       ) as sem_rank
                FROM {tables['messages']} m
                JOIN {tables['embeddings']} e ON m.message_uid = e.message_uid
                WHERE m.thread_uid != :exclude_thread_uid
                LIMIT :search_limit
            ),
            combined AS (
                SELECT COALESCE(f.message_uid, s.message_uid) as message_uid,
                       (COALESCE(:fts_weight / (60.0 + f.fts_rank), 0) +
                        COALESCE(:sem_weight / (60.0 + s.sem_rank), 0)) *
                       CASE WHEN f.message_uid IS NOT NULL AND s.message_uid IS NOT NULL
                            THEN :overlap_boost ELSE 1.0 END as rrf_score,
                       f.message_uid IS NOT NULL AND s.message_uid IS NOT NULL as is_dual_hit,
                       f.fts_rank,
                       s.sem_rank
                FROM fts_results f
                FULL OUTER JOIN semantic_results s ON f.message_uid = s.message_uid
            )
            SELECT message_uid, rrf_score, is_dual_hit, fts_rank, sem_rank
            FROM combined
            ORDER BY rrf_score DESC
            LIMIT :limit
        """)

        try:
            result = await session.execute(query, {
                "query": clean_query,
                "exclude_thread_uid": exclude_thread_uid,
                "fts_weight": fts_weight,
                "sem_weight": semantic_weight,
                "overlap_boost": overlap_boost,
                "search_limit": search_limit,
                "limit": limit
            })

            rows = result.fetchall()

            if return_anchors:
                return [HybridSearchResult(
                    message_uid=row[0],
                    score=float(row[1]),
                    is_anchor=bool(row[2]),
                    fts_rank=int(row[3]) if row[3] else None,
                    semantic_rank=int(row[4]) if row[4] else None
                ) for row in rows]
            else:
                return [(row[0], float(row[1])) for row in rows]

        except Exception as e:
            # Fallback to vector-only on any error (e.g., missing tsvector column)
            logger.debug(f"[Discovery] Hybrid search failed, falling back to vector-only: {e}")
            results = await self._search_conversation_messages(
                session, embedding, exclude_thread_uid, limit
            )
            if return_anchors:
                return [HybridSearchResult(
                    message_uid=uid, score=score, is_anchor=False
                ) for uid, score in results]
            return results

    # ==========================================
    # Manual Discovery API
    # ==========================================

    async def discover_now(
        self,
        ref: str,
        embedding: Optional[List[float]] = None,
        content: Optional[str] = None
    ) -> List[DiscoveredRelationship]:
        """
        Run discovery immediately for a given reference.

        If embedding not provided, fetches it from storage.
        Uses hybrid search (FTS + vector) for cross-thread message discovery
        when PostgreSQL is available and content is provided.

        Args:
            ref: Reference in format "type:id" (e.g., "message:abc123")
            embedding: Optional pre-computed embedding vector
            content: Optional content for hybrid FTS search

        Returns:
            List of discovered relationships
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
                # Get thread_uid and content for the message
                thread_uid = await self._get_thread_for_message(session, ref_id)

                # Get query text for hybrid search
                query_text = ""
                if content:
                    query_text = self._extract_query_text(content)
                elif self.config.use_hybrid_search:
                    # Fetch content from database
                    tables = _get_conversation_tables()
                    content_query = text(f"""
                        SELECT content FROM {tables['messages']}
                        WHERE message_uid = :message_uid
                    """)
                    result = await session.execute(content_query, {"message_uid": ref_id})
                    row = result.fetchone()
                    if row and row[0]:
                        query_text = self._extract_query_text(row[0])

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
                # Use hybrid search when configured and available
                if thread_uid:
                    if self.config.use_hybrid_search and query_text:
                        message_results = await self._search_conversation_messages_hybrid(
                            session, embedding, query_text, thread_uid, self.config.top_k
                        )
                        # RRF scores: use config threshold
                        for msg_uid, score in message_results:
                            if score >= self.config.rrf_threshold:
                                relationships.append(DiscoveredRelationship(
                                    from_ref=ref, to_ref=f"message:{msg_uid}",
                                    rel_type="similar_to", similarity=score,
                                    discovered_at=datetime.utcnow()
                                ))
                    else:
                        for msg_uid, sim in await self._search_conversation_messages(
                            session, embedding, thread_uid, self.config.top_k
                        ):
                            if sim >= self.config.min_similarity_cross_thread:
                                relationships.append(DiscoveredRelationship(
                                    from_ref=ref, to_ref=f"message:{msg_uid}",
                                    rel_type="similar_to", similarity=sim,
                                    discovered_at=datetime.utcnow()
                                ))

            elif ref_type == "thread":
                # Get representative query text from recent messages
                query_text = ""
                if self.config.use_hybrid_search:
                    tables = _get_conversation_tables()
                    recent_query = text(f"""
                        SELECT content FROM {tables['messages']}
                        WHERE thread_uid = :thread_uid
                        ORDER BY timestamp DESC
                        LIMIT 3
                    """)
                    result = await session.execute(recent_query, {"thread_uid": ref_id})
                    rows = result.fetchall()
                    if rows:
                        combined = " ".join(row[0] for row in rows if row[0])
                        query_text = self._extract_query_text(combined)

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

                # Search messages in other threads (hybrid when available)
                if self.config.use_hybrid_search and query_text:
                    message_results = await self._search_conversation_messages_hybrid(
                        session, embedding, query_text, ref_id, self.config.top_k
                    )
                    for msg_uid, score in message_results:
                        if score >= self.config.rrf_threshold:
                            relationships.append(DiscoveredRelationship(
                                from_ref=ref, to_ref=f"message:{msg_uid}",
                                rel_type="similar_to", similarity=score,
                                discovered_at=datetime.utcnow()
                            ))
                else:
                    for msg_uid, sim in await self._search_conversation_messages(
                        session, embedding, ref_id, self.config.top_k
                    ):
                        if sim >= self.config.min_similarity_cross_thread:
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
            SELECT me.message_uid, 1 - (me.embedding <=> '{embedding_str}'::vector) as similarity
            FROM {tables['embeddings']} me
            WHERE me.embedding IS NOT NULL
            ORDER BY me.embedding <=> '{embedding_str}'::vector
            LIMIT :limit
        """)

        result = await session.execute(query, {"limit": limit})
        return [(row[0], float(row[1])) for row in result.fetchall()]

    async def expand_with_neighbors(
        self,
        search_results: List[HybridSearchResult],
        anchor_limit: Optional[int] = None,
        non_anchor_limit: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """
        Expand search results by following graph edges, with more expansion for anchors.

        Anchors (dual-hit items from both FTS and semantic search) get more neighbors
        because they're more likely to be highly relevant.

        Args:
            search_results: Results from hybrid search with anchor metadata
            anchor_limit: Max neighbors per anchor (default from config)
            non_anchor_limit: Max neighbors per non-anchor (default from config)

        Returns:
            Dict mapping message_uid -> list of neighbor refs (e.g., observation:123)
        """
        from cathedral import MemoryGate

        anchor_limit = anchor_limit or self.config.anchor_neighbor_limit
        non_anchor_limit = non_anchor_limit or self.config.non_anchor_neighbor_limit

        expansion: Dict[str, List[str]] = {}

        for result in search_results:
            limit = anchor_limit if result.is_anchor else non_anchor_limit
            if limit == 0:
                expansion[result.message_uid] = []
                continue

            # Get related items via MemoryGate relationship graph
            message_ref = f"message:{result.message_uid}"
            try:
                related = MemoryGate.list_relationships(
                    ref=message_ref,
                    limit=limit
                )
                # Extract neighbor refs from relationships
                neighbors = []
                for rel in related:
                    # Get the "other" side of the relationship
                    other_ref = rel.get("to_ref") if rel.get("from_ref") == message_ref else rel.get("from_ref")
                    if other_ref and other_ref != message_ref:
                        neighbors.append(other_ref)
                expansion[result.message_uid] = neighbors[:limit]
            except Exception as e:
                logger.debug(f"[Discovery] Failed to expand neighbors for {message_ref}: {e}")
                expansion[result.message_uid] = []

        return expansion

    async def search_with_expansion(
        self,
        query_text: str,
        embedding: List[float],
        exclude_thread_uid: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Run hybrid search and expand results with neighbors based on anchor status.

        This is the main entry point for context composition - returns both
        search results and their expanded neighbors.

        Args:
            query_text: Search query text for FTS
            embedding: Query embedding vector
            exclude_thread_uid: Thread to exclude from results
            limit: Maximum search results

        Returns:
            Dict with 'results' (HybridSearchResult list) and 'expansion' (neighbors dict)
        """
        from cathedral.shared.db import get_async_session

        async with get_async_session() as session:
            results = await self._search_conversation_messages_hybrid(
                session, embedding, query_text, exclude_thread_uid, limit,
                return_anchors=True
            )

        # Expand with neighbors
        expansion = await self.expand_with_neighbors(results)

        return {
            "results": results,
            "expansion": expansion,
            "anchors": [r for r in results if r.is_anchor],
            "non_anchors": [r for r in results if not r.is_anchor],
        }


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


async def queue_message_discovery(
    message_uid: str,
    thread_uid: str,
    embedding: List[float],
    content: Optional[str] = None
):
    """Queue a message for discovery."""
    service = get_discovery_service()
    await service.queue_message_discovery(message_uid, thread_uid, embedding, content)


async def discover_for_ref(
    ref: str,
    content: Optional[str] = None
) -> List[DiscoveredRelationship]:
    """Run discovery immediately for a reference."""
    service = get_discovery_service()
    return await service.discover_now(ref, content=content)
