"""
Cathedral Unified Memory

Provides a single interface for:
- Conversation memory (threads, messages, context)
- Knowledge memory (observations, patterns, concepts) via MemoryGate
- Cross-system search and context composition

Conversation storage is provided by the MemoryGate conversation service.
"""

import asyncio
from typing import Any, Dict, List, Optional

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("UnifiedMemory")

from .types import (
    MemorySource,
    MemoryTier,
    SearchResult,
    ThreadInfo,
    MemoryStats,
)

MemoryGate = None


def _get_conversation_service():
    """Get the MemoryGate conversation service."""
    from cathedral.MemoryGate.conversation import get_conversation_service
    return get_conversation_service()


def _get_memorygate():
    """Get the MemoryGate module."""
    global MemoryGate
    if MemoryGate is None:
        from cathedral import MemoryGate as _MemoryGate
        MemoryGate = _MemoryGate
    return MemoryGate


class UnifiedMemory:
    """
    Unified memory interface for Cathedral.

    Combines conversation context with knowledge memory (MemoryGate)
    into a single coherent interface.

    Usage:
        memory = UnifiedMemory()

        # Conversation operations
        thread_uid = await memory.create_thread("New Conversation")
        await memory.append_message("user", "Hello", thread_uid)
        context = await memory.compose_context("How are you?", thread_uid)

        # Knowledge operations
        await memory.store_observation("User prefers concise answers")
        results = await memory.search_knowledge("user preferences")

        # Unified search
        all_results = await memory.unified_search("important topic")
    """

    def __init__(self, conversation: Any = None):
        """
        Initialize UnifiedMemory.

        Args:
            conversation: Optional conversation service override (for testing)
        """
        self._conversation = conversation or _get_conversation_service()

        # Initialize MemoryGate for knowledge operations
        MemoryGate = _get_memorygate()
        self._mg_initialized = MemoryGate.is_initialized()
        if not self._mg_initialized:
            self._mg_initialized = MemoryGate.initialize()

    # ==========================================================================
    # Conversation Layer
    # ==========================================================================

    async def create_thread(
        self,
        thread_name: str = None,
        metadata: Dict = None
    ) -> str:
        """
        Create a new conversation thread.

        Args:
            thread_name: Name for the thread (auto-generated if None)
            metadata: Optional metadata dict (reserved for future use)

        Returns:
            Thread UID
        """
        return self._conversation.create_thread(thread_name)

    async def append_message(
        self,
        role: str,
        content: str,
        thread_uid: str = None,
        extract_memory: bool = False
    ) -> str:
        """
        Append a message to a conversation thread.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            thread_uid: Target thread (uses active thread if None)
            extract_memory: If True and role="assistant", extract observations

        Returns:
            Message UID
        """
        message_uid = await self._conversation.append_async(role, content, thread_uid)

        # Optionally extract observations from assistant responses
        if extract_memory and role == "assistant" and self._mg_initialized:
            task = asyncio.create_task(
                self._extract_observations(content, thread_uid, message_uid)
            )
            task.add_done_callback(_handle_extraction_task_exception)

        return message_uid


def _handle_extraction_task_exception(task: asyncio.Task) -> None:
    """Handle exceptions from background extraction tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        _log.error(f" Background extraction failed: {exc}")

    async def recall_conversation(
        self,
        thread_uid: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Recall conversation messages from a thread.

        Args:
            thread_uid: Thread to recall from (uses active if None)
            limit: Maximum messages to return

        Returns:
            List of message dicts with role, content, timestamp
        """
        messages = await self._conversation.recall_async(thread_uid)

        if limit and len(messages) > limit:
            messages = messages[-limit:]
        return messages

    async def search_conversations(
        self,
        query: str,
        thread_uid: str = None,
        limit: int = 5,
        include_all_threads: bool = False,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Semantic search across conversation messages.

        Args:
            query: Search query
            thread_uid: Limit to specific thread (None for active thread)
            limit: Maximum results
            include_all_threads: Search across all threads
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        results = await self._conversation.semantic_search(
            query,
            thread_uid=thread_uid,
            limit=limit,
            include_all_threads=include_all_threads
        )

        search_results = []
        for r in results:
            similarity = r.get("similarity", 0.0)
            if similarity >= min_similarity:
                search_results.append(
                    SearchResult.from_conversation_message(r, similarity)
                )

        return search_results

    async def search_summaries(
        self,
        query: str,
        limit: int = 3,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search conversation summaries.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        results = await self._conversation.search_summaries(query, limit=limit)

        search_results = []
        for r in results:
            similarity = r.get("similarity", 0.0)
            if similarity >= min_similarity:
                search_results.append(
                    SearchResult.from_conversation_summary(r, similarity)
                )

        return search_results

    # ==========================================================================
    # Knowledge Layer (MemoryGate wrapper)
    # ==========================================================================

    async def store_observation(
        self,
        observation: str,
        confidence: float = 0.8,
        domain: str = None,
        evidence: List[str] = None
    ) -> Optional[Dict]:
        """
        Store an observation in knowledge memory.

        Args:
            observation: The observation text
            confidence: Confidence level 0.0-1.0
            domain: Category/domain tag
            evidence: List of supporting evidence

        Returns:
            Stored observation dict with ID, or None if failed
        """
        if not self._mg_initialized:
            return None

        MemoryGate = _get_memorygate()
        return MemoryGate.store_observation(
            observation,
            confidence=confidence,
            domain=domain,
            evidence=evidence
        )

    async def store_pattern(
        self,
        category: str,
        name: str,
        text: str,
        confidence: float = 0.8,
        evidence_ids: List[int] = None
    ) -> Optional[Dict]:
        """
        Create or update a pattern (synthesized understanding).

        Args:
            category: Pattern category
            name: Pattern name (unique within category)
            text: Pattern description
            confidence: Confidence level
            evidence_ids: List of observation IDs supporting this pattern

        Returns:
            Pattern dict or None if failed
        """
        if not self._mg_initialized:
            return None

        MemoryGate = _get_memorygate()
        return MemoryGate.store_pattern(
            category=category,
            name=name,
            text=text,
            confidence=confidence,
            evidence_ids=evidence_ids
        )

    async def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.0,
        domain: str = None,
        include_cold: bool = False
    ) -> List[SearchResult]:
        """
        Search knowledge memory (observations, patterns, concepts).

        Args:
            query: Search query
            limit: Maximum results
            min_confidence: Minimum confidence threshold
            domain: Filter by domain
            include_cold: Include cold-tier archived memories

        Returns:
            List of SearchResult objects
        """
        if not self._mg_initialized:
            return []

        MemoryGate = _get_memorygate()
        results = MemoryGate.search(
            query,
            limit=limit,
            min_confidence=min_confidence,
            domain=domain,
            include_cold=include_cold
        )

        return [SearchResult.from_memorygate(r) for r in (results or [])]

    async def recall_observations(
        self,
        domain: str = None,
        limit: int = 10,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """
        Recall recent observations from knowledge memory.

        Args:
            domain: Filter by domain
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of observation dicts
        """
        if not self._mg_initialized:
            return []

        MemoryGate = _get_memorygate()
        return MemoryGate.recall(
            domain=domain,
            limit=limit,
            min_confidence=min_confidence
        )

    # ==========================================================================
    # Unified Operations
    # ==========================================================================

    async def unified_search(
        self,
        query: str,
        sources: List[MemorySource] = None,
        limit_per_source: int = 3,
        min_similarity: float = 0.3
    ) -> List[SearchResult]:
        """
        Search across all memory sources.

        This is the primary cross-pollination method, searching both
        conversation history and knowledge memory simultaneously.

        Args:
            query: Search query
            sources: Which sources to search (default: all)
            limit_per_source: Max results per source type
            min_similarity: Minimum similarity threshold

        Returns:
            Combined results sorted by similarity
        """
        if sources is None:
            sources = list(MemorySource)

        results = []

        # Search conversations
        if MemorySource.CONVERSATION in sources:
            conv_results = await self.search_conversations(
                query,
                limit=limit_per_source,
                include_all_threads=True,
                min_similarity=min_similarity
            )
            results.extend(conv_results)

        # Search summaries
        if MemorySource.SUMMARY in sources:
            summary_results = await self.search_summaries(
                query,
                limit=limit_per_source,
                min_similarity=min_similarity
            )
            results.extend(summary_results)

        # Search knowledge memory
        knowledge_sources = {
            MemorySource.OBSERVATION,
            MemorySource.PATTERN,
            MemorySource.CONCEPT,
            MemorySource.DOCUMENT
        }
        if knowledge_sources & set(sources):
            kg_results = await self.search_knowledge(
                query,
                limit=limit_per_source * 2  # Get more, filter by type
            )
            # Filter by requested source types
            for r in kg_results:
                if r.source in sources and r.similarity >= min_similarity:
                    results.append(r)

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)

        return results

    async def compose_context(
        self,
        user_input: str,
        thread_uid: str,
        max_tokens: int = 4096,
        include_knowledge: bool = True,
        knowledge_limit: int = 3,
        knowledge_min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        Compose full prompt context combining conversation and knowledge.

        This is the main context-building method that enriches conversation
        history with relevant knowledge from MemoryGate.

        Args:
            user_input: Current user input (used for semantic search)
            thread_uid: Thread to compose context for
            max_tokens: Maximum token budget for context
            include_knowledge: Whether to inject knowledge context
            knowledge_limit: Max knowledge items to inject
            knowledge_min_similarity: Minimum similarity for knowledge

        Returns:
            Message list ready for LLM (role/content dicts)
        """
        # Get conversation context
        context = await self._conversation.compose_context(
            user_input, thread_uid
        )

        # Inject relevant knowledge if enabled
        if include_knowledge and self._mg_initialized:
            knowledge = await self.search_knowledge(
                user_input,
                limit=knowledge_limit
            )

            # Filter by similarity threshold
            relevant = [k for k in knowledge if k.similarity >= knowledge_min_similarity]

            if relevant:
                knowledge_text = self._format_knowledge_context(relevant)
                # Insert after system prompt (index 1) or at start
                insert_pos = 1 if context and context[0].get("role") == "system" else 0
                context.insert(insert_pos, {
                    "role": "system",
                    "content": knowledge_text
                })

        return context

    # ==========================================================================
    # Thread Management
    # ==========================================================================

    def list_threads(self) -> List[Dict]:
        """List all conversation threads."""
        return self._conversation.list_threads()

    def switch_thread(self, thread_uid: str) -> None:
        """Switch active thread."""
        self._conversation.switch_thread(thread_uid)

    def get_active_thread(self) -> str:
        """Get active thread UID."""
        return self._conversation.get_active_thread_uid()

    def get_active_thread_metadata(self) -> Dict:
        """Get metadata for the active thread."""
        return self._conversation.get_active_thread() or {}

    def clear_thread(self, thread_uid: str = None) -> None:
        """Clear messages from a thread."""
        self._conversation.clear(thread_uid)

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_stats(self) -> MemoryStats:
        """
        Get combined memory statistics.

        Returns:
            MemoryStats with counts from both systems
        """
        stats = MemoryStats(
            conversation_available=True,
            memorygate_available=self._mg_initialized
        )

        # Get conversation stats
        conv_stats = self._conversation.get_stats()
        stats.thread_count = conv_stats.get("threads", 0)
        stats.message_count = conv_stats.get("messages", 0)
        stats.embedded_message_count = conv_stats.get("embedded", 0)
        stats.summary_count = conv_stats.get("summaries", 0)

        # Get MemoryGate stats
        if self._mg_initialized:
            MemoryGate = _get_memorygate()
            mg_stats = MemoryGate.get_stats()
            if mg_stats:
                counts = mg_stats.get("counts", {})
                stats.observation_count = counts.get("observations", 0)
                stats.pattern_count = counts.get("patterns", 0)
                stats.concept_count = counts.get("concepts", 0)
                stats.document_count = counts.get("documents", 0)

        return stats

    # ==========================================================================
    # Memory Extraction (Cross-Pollination)
    # ==========================================================================

    async def _extract_observations(
        self,
        content: str,
        thread_uid: str = None,
        message_uid: str = None
    ) -> List[str]:
        """
        Extract observations from content and store in MemoryGate.

        This enables automatic cross-pollination from conversations
        to knowledge memory.

        Args:
            content: Text content to extract from
            thread_uid: Source thread
            message_uid: Source message

        Returns:
            List of created observation refs
        """
        if not self._mg_initialized:
            return []

        try:
            from cathedral.MemoryGate.auto_memory import extract_from_exchange

            # Extract observations
            observations = extract_from_exchange(
                user_input="",  # Not needed for response extraction
                assistant_response=content
            )

            refs = []
            for obs in observations:
                evidence = []
                if thread_uid:
                    evidence.append(f"thread:{thread_uid}")
                if message_uid:
                    evidence.append(f"message:{message_uid}")

                result = await self.store_observation(
                    observation=obs.get("text", obs) if isinstance(obs, dict) else str(obs),
                    confidence=obs.get("confidence", 0.7) if isinstance(obs, dict) else 0.7,
                    domain=obs.get("domain", "extracted") if isinstance(obs, dict) else "extracted",
                    evidence=evidence if evidence else None
                )

                if result:
                    refs.append(f"observation:{result.get('id')}")

            return refs

        except ImportError:
            return []
        except Exception as e:
            # Don't let extraction failures break the main flow
            _log.error(f" Extraction error: {e}")
            return []

    def _format_knowledge_context(self, results: List[SearchResult]) -> str:
        """
        Format knowledge search results as context injection.

        Args:
            results: List of SearchResult objects

        Returns:
            Formatted string for system message injection
        """
        if not results:
            return ""

        lines = ["[RELEVANT KNOWLEDGE]"]

        for r in results:
            source_label = r.source.value.upper()
            content_preview = r.content[:200]
            if len(r.content) > 200:
                content_preview += "..."

            confidence_str = f" ({r.confidence:.0%})" if r.confidence else ""
            lines.append(f"- [{source_label}]{confidence_str} {content_preview}")

        return "\n".join(lines)

    # ==========================================================================
    # Backfill and Maintenance
    # ==========================================================================

    async def backfill_embeddings(self, batch_size: int = 50) -> int:
        """
        Generate embeddings for messages without them.

        Args:
            batch_size: Number of messages to process per batch

        Returns:
            Number of embeddings generated
        """
        return await self._conversation.backfill_embeddings(batch_size)


# ==========================================================================
# Module-level singleton and convenience functions
# ==========================================================================

_memory: Optional[UnifiedMemory] = None


def get_memory(conversation: Any = None) -> UnifiedMemory:
    """
    Get or create global UnifiedMemory instance.

    Args:
        conversation: Optional conversation service override (for testing)
    """
    global _memory
    if _memory is None:
        _memory = UnifiedMemory(conversation)
    return _memory


def reset_memory() -> None:
    """Reset the global memory instance (mainly for testing)."""
    global _memory
    _memory = None


# Convenience async functions
async def compose_context(user_input: str, thread_uid: str, **kwargs) -> List[Dict]:
    """Compose context for LLM prompt."""
    return await get_memory().compose_context(user_input, thread_uid, **kwargs)


async def append_message(role: str, content: str, thread_uid: str = None, **kwargs) -> str:
    """Append message to conversation."""
    return await get_memory().append_message(role, content, thread_uid, **kwargs)


async def unified_search(query: str, **kwargs) -> List[SearchResult]:
    """Search across all memory sources."""
    return await get_memory().unified_search(query, **kwargs)


async def search_knowledge(query: str, **kwargs) -> List[SearchResult]:
    """Search knowledge memory."""
    return await get_memory().search_knowledge(query, **kwargs)


async def store_observation(observation: str, **kwargs) -> Optional[Dict]:
    """Store an observation."""
    return await get_memory().store_observation(observation, **kwargs)


# Export types
__all__ = [
    "UnifiedMemory",
    "get_memory",
    "reset_memory",
    "compose_context",
    "append_message",
    "unified_search",
    "search_knowledge",
    "store_observation",
    "MemorySource",
    "MemoryTier",
    "SearchResult",
    "ThreadInfo",
    "MemoryStats",
]
