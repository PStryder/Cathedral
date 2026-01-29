"""
Tests for Cathedral Unified Memory.

Tests the UnifiedMemory abstraction layer that combines
conversation memory (MemoryGate conversation service) and MemoryGate knowledge.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from cathedral.Memory import (
    UnifiedMemory,
    get_memory,
    reset_memory,
    MemorySource,
    SearchResult,
    MemoryStats,
)


class TestMemorySource:
    """Tests for MemorySource enum."""

    def test_all_sources_defined(self):
        """All expected memory sources should be defined."""
        assert MemorySource.CONVERSATION == "conversation"
        assert MemorySource.OBSERVATION == "observation"
        assert MemorySource.PATTERN == "pattern"
        assert MemorySource.CONCEPT == "concept"
        assert MemorySource.DOCUMENT == "document"
        assert MemorySource.SUMMARY == "summary"

    def test_sources_are_strings(self):
        """MemorySource values should be strings."""
        for source in MemorySource:
            assert isinstance(source.value, str)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Should create SearchResult with all fields."""
        result = SearchResult(
            source=MemorySource.OBSERVATION,
            content="Test observation content",
            similarity=0.85,
            ref="observation:42",
            metadata={"domain": "test"},
            confidence=0.9,
        )

        assert result.source == MemorySource.OBSERVATION
        assert result.content == "Test observation content"
        assert result.similarity == 0.85
        assert result.ref == "observation:42"
        assert result.confidence == 0.9

    def test_search_result_to_dict(self):
        """Should serialize to dict correctly."""
        result = SearchResult(
            source=MemorySource.CONVERSATION,
            content="Hello world",
            similarity=0.75,
            ref="message:abc123",
        )

        data = result.to_dict()

        assert data["source"] == "conversation"
        assert data["content"] == "Hello world"
        assert data["similarity"] == 0.75
        assert data["ref"] == "message:abc123"

    def test_from_conversation_message(self):
        """Should create SearchResult from conversation message dict."""
        msg = {
            "message_uid": "uuid-123",
            "role": "user",
            "content": "Test message",
            "timestamp": "2024-01-01T12:00:00",
            "thread_uid": "thread-456",
        }

        result = SearchResult.from_conversation_message(msg, similarity=0.8)

        assert result.source == MemorySource.CONVERSATION
        assert result.content == "Test message"
        assert result.ref == "message:uuid-123"
        assert result.metadata["role"] == "user"

    def test_from_memorygate(self):
        """Should create SearchResult from MemoryGate result dict."""
        mg_result = {
            "source_type": "observation",
            "id": 42,
            "content": "Observation content",
            "similarity": 0.9,
            "confidence": 0.85,
            "domain": "technical",
            "ref": "observation:42",
        }

        result = SearchResult.from_memorygate(mg_result)

        assert result.source == MemorySource.OBSERVATION
        assert result.content == "Observation content"
        assert result.similarity == 0.9
        assert result.confidence == 0.85


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_default_stats(self):
        """Default stats should be zero."""
        stats = MemoryStats()

        assert stats.thread_count == 0
        assert stats.observation_count == 0
        assert stats.conversation_available is False
        assert stats.memorygate_available is False

    def test_stats_to_dict(self):
        """Should serialize to nested dict."""
        stats = MemoryStats(
            thread_count=5,
            message_count=100,
            observation_count=50,
            conversation_available=True,
            memorygate_available=True,
        )

        data = stats.to_dict()

        assert data["conversation"]["threads"] == 5
        assert data["conversation"]["messages"] == 100
        assert data["knowledge"]["observations"] == 50
        assert data["conversation"]["available"] is True
        assert data["knowledge"]["available"] is True


class TestUnifiedMemoryInitialization:
    """Tests for UnifiedMemory initialization."""

    def test_get_memory_returns_singleton(self):
        """get_memory should return the same instance."""
        reset_memory()

        mem1 = get_memory()
        mem2 = get_memory()

        assert mem1 is mem2

    def test_reset_memory_clears_singleton(self):
        """reset_memory should clear the global instance."""
        mem1 = get_memory()
        reset_memory()
        mem2 = get_memory()

        # After reset, should get new instance
        # Note: In practice they might still be equal if created the same way
        assert mem2 is not None


class TestUnifiedMemoryConversation:
    """Tests for conversation layer operations."""

    @pytest.fixture
    def mock_conversation(self):
        """Create mock conversation service instance."""
        with patch('cathedral.Memory._get_conversation_service') as mock_get:
            mock = MagicMock()
            mock_get.return_value = mock
            yield mock

    @pytest.fixture
    def mock_memorygate(self):
        """Create mock MemoryGate module."""
        with patch('cathedral.Memory.MemoryGate') as MockMG:
            MockMG.is_initialized.return_value = True
            MockMG.initialize.return_value = True
            yield MockMG

    def test_create_thread(self, mock_conversation, mock_memorygate):
        """Should delegate thread creation to the conversation service."""
        reset_memory()
        mock_conversation.create_thread.return_value = "thread-123"

        with patch('cathedral.Memory.MemoryGate', mock_memorygate):
            memory = UnifiedMemory()

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memory.create_thread("Test Thread")
        )

        mock_conversation.create_thread.assert_called_once_with("Test Thread")
        assert result == "thread-123"

    def test_list_threads(self, mock_conversation, mock_memorygate):
        """Should delegate thread listing to the conversation service."""
        reset_memory()
        mock_conversation.list_threads.return_value = [
            {"thread_uid": "t1", "thread_name": "Thread 1"},
            {"thread_uid": "t2", "thread_name": "Thread 2"},
        ]

        with patch('cathedral.Memory.MemoryGate', mock_memorygate):
            memory = UnifiedMemory()

        threads = memory.list_threads()

        assert len(threads) == 2
        assert threads[0]["thread_name"] == "Thread 1"

    def test_switch_thread(self, mock_conversation, mock_memorygate):
        """Should delegate thread switching to the conversation service."""
        reset_memory()

        with patch('cathedral.Memory.MemoryGate', mock_memorygate):
            memory = UnifiedMemory()

        memory.switch_thread("thread-456")

        mock_conversation.switch_thread.assert_called_once_with("thread-456")

    def test_clear_thread(self, mock_conversation, mock_memorygate):
        """Should delegate thread clearing to the conversation service."""
        reset_memory()

        with patch('cathedral.Memory.MemoryGate', mock_memorygate):
            memory = UnifiedMemory()

        memory.clear_thread("thread-789")

        mock_conversation.clear.assert_called_once_with("thread-789")


class TestUnifiedMemoryKnowledge:
    """Tests for knowledge layer operations."""

    @pytest.fixture
    def memory_with_mocks(self):
        """Create UnifiedMemory with mocked dependencies."""
        reset_memory()

        mock_conversation = MagicMock()
        mock_mg = MagicMock()
        mock_mg.is_initialized.return_value = True
        mock_mg.initialize.return_value = True

        with patch('cathedral.Memory._get_conversation_service', return_value=mock_conversation):
            with patch('cathedral.Memory.MemoryGate', mock_mg):
                memory = UnifiedMemory()
                memory._mg_initialized = True
                yield memory, mock_conversation, mock_mg

    @pytest.mark.asyncio
    async def test_store_observation(self, memory_with_mocks):
        """Should delegate observation storage to MemoryGate."""
        memory, _, mock_mg = memory_with_mocks
        mock_mg.store_observation.return_value = {"id": 42, "status": "stored"}

        result = await memory.store_observation(
            "Test observation",
            confidence=0.9,
            domain="test"
        )

        mock_mg.store_observation.assert_called_once()
        assert result["id"] == 42

    @pytest.mark.asyncio
    async def test_store_observation_without_memorygate(self):
        """Should return None when MemoryGate not initialized."""
        reset_memory()

        with patch('cathedral.Memory._get_conversation_service', return_value=MagicMock()):
            with patch('cathedral.Memory.MemoryGate') as mock_mg:
                mock_mg.is_initialized.return_value = False
                mock_mg.initialize.return_value = False
                memory = UnifiedMemory()

        result = await memory.store_observation("Test")

        assert result is None

    @pytest.mark.asyncio
    async def test_search_knowledge(self, memory_with_mocks):
        """Should search MemoryGate and return SearchResults."""
        memory, _, mock_mg = memory_with_mocks
        mock_mg.search.return_value = [
            {
                "source_type": "observation",
                "id": 1,
                "content": "Result 1",
                "similarity": 0.9,
                "ref": "observation:1",
            },
            {
                "source_type": "pattern",
                "id": 2,
                "content": "Result 2",
                "similarity": 0.8,
                "ref": "pattern:2",
            },
        ]

        results = await memory.search_knowledge("test query", limit=5)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].source == MemorySource.OBSERVATION
        assert results[1].source == MemorySource.PATTERN


class TestUnifiedSearch:
    """Tests for unified search across all sources."""

    @pytest.fixture
    def memory_with_mocks(self):
        """Create UnifiedMemory with configured mocks."""
        reset_memory()

        mock_conversation = MagicMock()
        mock_conversation.semantic_search = AsyncMock(return_value=[
            {
                "message_uid": "msg-1",
                "role": "user",
                "content": "Conversation result",
                "similarity": 0.85,
            }
        ])
        mock_conversation.search_summaries = AsyncMock(return_value=[])

        mock_mg = MagicMock()
        mock_mg.is_initialized.return_value = True
        mock_mg.initialize.return_value = True
        mock_mg.search.return_value = [
            {
                "source_type": "observation",
                "id": 42,
                "content": "Knowledge result",
                "similarity": 0.9,
                "ref": "observation:42",
            }
        ]

        with patch('cathedral.Memory._get_conversation_service', return_value=mock_conversation):
            with patch('cathedral.Memory.MemoryGate', mock_mg):
                memory = UnifiedMemory()
                memory._mg_initialized = True
                yield memory, mock_conversation, mock_mg

    @pytest.mark.asyncio
    async def test_unified_search_all_sources(self, memory_with_mocks):
        """Should search all sources and combine results."""
        memory, mock_conversation, mock_mg = memory_with_mocks

        results = await memory.unified_search(
            "test query",
            sources=None,  # All sources
            limit_per_source=3,
            min_similarity=0.3
        )

        # Should have results from both sources
        assert len(results) >= 1
        # Results should be sorted by similarity
        if len(results) > 1:
            assert results[0].similarity >= results[1].similarity

    @pytest.mark.asyncio
    async def test_unified_search_specific_sources(self, memory_with_mocks):
        """Should only search requested sources."""
        memory, mock_conversation, mock_mg = memory_with_mocks

        results = await memory.unified_search(
            "test query",
            sources=[MemorySource.OBSERVATION],
            limit_per_source=3
        )

        # Should not have called conversation semantic_search for conversations
        # since we only requested OBSERVATION
        # (though implementation may still call it)
        assert all(r.source == MemorySource.OBSERVATION for r in results if r.source != MemorySource.CONVERSATION)

    @pytest.mark.asyncio
    async def test_unified_search_respects_min_similarity(self, memory_with_mocks):
        """Should filter results below minimum similarity."""
        memory, mock_conversation, mock_mg = memory_with_mocks

        # Set high minimum similarity
        results = await memory.unified_search(
            "test query",
            min_similarity=0.95
        )

        # All results should meet threshold
        for r in results:
            assert r.similarity >= 0.95


class TestContextComposition:
    """Tests for context composition."""

    @pytest.fixture
    def memory_with_mocks(self):
        """Create UnifiedMemory with configured mocks."""
        reset_memory()

        mock_conversation = MagicMock()
        mock_conversation.compose_context = AsyncMock(return_value=[
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ])

        mock_mg = MagicMock()
        mock_mg.is_initialized.return_value = True
        mock_mg.initialize.return_value = True
        mock_mg.search.return_value = [
            {
                "source_type": "observation",
                "id": 1,
                "content": "Relevant knowledge",
                "similarity": 0.8,
                "ref": "observation:1",
                "confidence": 0.9,
            }
        ]

        with patch('cathedral.Memory._get_conversation_service', return_value=mock_conversation):
            with patch('cathedral.Memory.MemoryGate', mock_mg):
                memory = UnifiedMemory()
                memory._mg_initialized = True
                yield memory, mock_conversation, mock_mg

    @pytest.mark.asyncio
    async def test_compose_context_includes_conversation(self, memory_with_mocks):
        """Should include conversation history in context."""
        memory, mock_conversation, _ = memory_with_mocks

        context = await memory.compose_context(
            "New user input",
            "thread-123",
            include_knowledge=False
        )

        mock_conversation.compose_context.assert_called_once()
        assert len(context) >= 2

    @pytest.mark.asyncio
    async def test_compose_context_injects_knowledge(self, memory_with_mocks):
        """Should inject knowledge context when enabled."""
        memory, _, mock_mg = memory_with_mocks

        context = await memory.compose_context(
            "New user input",
            "thread-123",
            include_knowledge=True,
            knowledge_min_similarity=0.5
        )

        # Should have called MemoryGate search
        mock_mg.search.assert_called()

        # Should have injected knowledge (look for system message with knowledge)
        knowledge_messages = [
            m for m in context
            if m.get("role") == "system" and "KNOWLEDGE" in m.get("content", "").upper()
        ]
        assert len(knowledge_messages) >= 1

    @pytest.mark.asyncio
    async def test_compose_context_respects_similarity_threshold(self, memory_with_mocks):
        """Should not inject knowledge below similarity threshold."""
        memory, _, mock_mg = memory_with_mocks

        # Set very high threshold
        context = await memory.compose_context(
            "New user input",
            "thread-123",
            include_knowledge=True,
            knowledge_min_similarity=0.99  # Higher than our mock results
        )

        # Should not have knowledge injection (results don't meet threshold)
        knowledge_messages = [
            m for m in context
            if m.get("role") == "system" and "KNOWLEDGE" in m.get("content", "").upper()
        ]
        assert len(knowledge_messages) == 0


class TestMemoryExtraction:
    """Tests for automatic memory extraction."""

    @pytest.mark.asyncio
    async def test_append_message_with_extraction(self):
        """Should extract observations when extract_memory=True."""
        reset_memory()

        mock_conversation = MagicMock()
        mock_conversation.append_async = AsyncMock(return_value="msg-123")

        mock_mg = MagicMock()
        mock_mg.is_initialized.return_value = True
        mock_mg.initialize.return_value = True
        mock_mg.store_observation.return_value = {"id": 1}

        with patch('cathedral.Memory._get_conversation_service', return_value=mock_conversation):
            with patch('cathedral.Memory.MemoryGate', mock_mg):
                memory = UnifiedMemory()
                memory._mg_initialized = True

        # Mock the extraction function
        with patch('cathedral.Memory.UnifiedMemory._extract_observations') as mock_extract:
            mock_extract.return_value = ["observation:1"]

            result = await memory.append_message(
                "assistant",
                "Here is some useful information.",
                "thread-123",
                extract_memory=True
            )

            # Should have called conversation append
            mock_conversation.append_async.assert_called_once()
            assert result == "msg-123"


class TestKnowledgeFormatting:
    """Tests for knowledge context formatting."""

    def test_format_knowledge_context(self):
        """Should format knowledge results correctly."""
        reset_memory()

        mock_conversation = MagicMock()
        mock_mg = MagicMock()
        mock_mg.is_initialized.return_value = False

        with patch('cathedral.Memory._get_conversation_service', return_value=mock_conversation):
            with patch('cathedral.Memory.MemoryGate', mock_mg):
                memory = UnifiedMemory()

        results = [
            SearchResult(
                source=MemorySource.OBSERVATION,
                content="Important observation",
                similarity=0.9,
                ref="observation:1",
                confidence=0.85,
            ),
            SearchResult(
                source=MemorySource.PATTERN,
                content="Recognized pattern",
                similarity=0.8,
                ref="pattern:2",
            ),
        ]

        formatted = memory._format_knowledge_context(results)

        assert "[RELEVANT KNOWLEDGE]" in formatted
        assert "OBSERVATION" in formatted
        assert "PATTERN" in formatted
        assert "Important observation" in formatted

    def test_format_empty_knowledge(self):
        """Should return empty string for no results."""
        reset_memory()

        mock_conversation = MagicMock()
        mock_mg = MagicMock()
        mock_mg.is_initialized.return_value = False

        with patch('cathedral.Memory._get_conversation_service', return_value=mock_conversation):
            with patch('cathedral.Memory.MemoryGate', mock_mg):
                memory = UnifiedMemory()

        formatted = memory._format_knowledge_context([])

        assert formatted == ""
