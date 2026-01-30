"""
Tests for ScriptureGate document storage and retrieval.
"""

import asyncio
import pytest
from pathlib import Path

# Check for database dependencies
try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

# Skip entire module if aiosqlite not available
pytestmark = pytest.mark.skipif(
    not HAS_AIOSQLITE,
    reason="aiosqlite not installed - database tests require async SQLite support"
)

from cathedral.ScriptureGate import (
    init_scripture_db,
    store,
    store_text,
    store_artifact,
    get,
    list_scriptures,
    remove,
    read,
    stats,
    is_healthy,
    get_health_status,
    get_dependencies,
)


class TestScriptureGateInitialization:
    """Tests for ScriptureGate initialization."""

    def test_init_scripture_db(self):
        """Init should create tables without error."""
        # Already initialized by conftest, just verify no error
        init_scripture_db()

    def test_is_healthy(self):
        """Health check should return True after init."""
        assert is_healthy() is True

    def test_get_health_status(self):
        """Health status should return proper structure."""
        status = get_health_status()

        assert "gate" in status
        assert status["gate"] == "ScriptureGate"
        assert "healthy" in status
        assert "initialized" in status
        assert "checks" in status
        assert "tables_created" in status["checks"]

    def test_get_dependencies(self):
        """Should return list of dependencies."""
        deps = get_dependencies()
        assert isinstance(deps, list)
        assert "database" in deps
        assert "filesystem" in deps


class TestScriptureStorage:
    """Tests for storing scriptures."""

    @pytest.mark.asyncio
    async def test_store_text_content(self):
        """Should store text content directly."""
        result = await store(
            source="This is test content for ScriptureGate.",
            title="Test Document",
            description="A test document",
            tags=["test", "unit-test"],
            file_type="document",
            original_name="test.txt",
            source_type="test",
            auto_index=False,  # Skip embedding generation
        )

        assert result is not None
        assert "uid" in result
        assert result["title"] == "Test Document"
        assert result["file_type"] == "document"
        assert "ref" in result

    @pytest.mark.asyncio
    async def test_store_bytes_content(self):
        """Should store bytes content."""
        content = b"Binary content for testing"

        result = await store(
            source=content,
            title="Binary Test",
            original_name="binary.bin",
            auto_index=False,
        )

        assert result is not None
        assert "uid" in result

    @pytest.mark.asyncio
    async def test_store_bytes_requires_original_name(self):
        """Should raise error if original_name not provided for bytes."""
        content = b"Binary content"

        with pytest.raises(ValueError, match="original_name required"):
            await store(source=content, title="No Name")

    @pytest.mark.asyncio
    async def test_store_text_convenience(self):
        """store_text should work correctly."""
        result = await store_text(
            content="Text content via convenience function",
            title="Convenience Test",
            description="Testing store_text",
            tags=["convenience"],
        )

        assert result is not None
        assert result["title"] == "Convenience Test"
        assert result["file_type"] == "document"

    @pytest.mark.asyncio
    async def test_store_artifact(self):
        """store_artifact should work with dict content."""
        artifact_data = {"key": "value", "count": 42}

        result = await store_artifact(
            content=artifact_data,
            title="JSON Artifact",
            description="A JSON artifact",
            tags=["artifact", "json"],
        )

        assert result is not None
        assert result["title"] == "JSON Artifact"
        assert result["file_type"] == "artifact"


class TestScriptureRetrieval:
    """Tests for retrieving scriptures."""

    @pytest.fixture
    async def stored_scripture(self):
        """Create a scripture for retrieval tests."""
        result = await store(
            source="Content for retrieval testing",
            title="Retrieval Test Doc",
            description="Document for testing retrieval",
            tags=["retrieval", "test"],
            original_name="retrieval_test.txt",
            auto_index=False,
        )
        return result

    @pytest.mark.asyncio
    async def test_get_by_uid(self, stored_scripture):
        """Should retrieve scripture by UID."""
        uid = stored_scripture["uid"]

        result = get(uid)

        assert result is not None
        assert result["uid"] == uid
        assert result["title"] == "Retrieval Test Doc"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Should return None for nonexistent UID."""
        result = get("nonexistent-uid-12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_scriptures(self, stored_scripture):
        """Should list scriptures."""
        results = list_scriptures(limit=50)

        assert isinstance(results, list)
        # Should have at least our test document
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_list_scriptures_by_type(self, stored_scripture):
        """Should filter by file type."""
        results = list_scriptures(file_type="document", limit=50)

        for r in results:
            assert r["file_type"] == "document"

    @pytest.mark.asyncio
    async def test_read_content(self, stored_scripture):
        """Should read file content."""
        uid = stored_scripture["uid"]

        content = read(uid, as_text=True)

        assert content is not None
        assert "Content for retrieval testing" in content


class TestScriptureDeletion:
    """Tests for deleting scriptures."""

    @pytest.mark.asyncio
    async def test_soft_delete(self):
        """Should soft delete (mark as deleted)."""
        # Create a scripture to delete
        result = await store(
            source="Content to be deleted",
            title="Delete Me",
            original_name="delete_me.txt",
            auto_index=False,
        )
        uid = result["uid"]

        # Soft delete
        success = remove(uid, hard_delete=False)
        assert success is True

        # Should not be retrievable
        retrieved = get(uid)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_hard_delete(self):
        """Should hard delete (remove file)."""
        # Create a scripture to delete
        result = await store(
            source="Content to be hard deleted",
            title="Hard Delete Me",
            original_name="hard_delete.txt",
            auto_index=False,
        )
        uid = result["uid"]

        # Hard delete
        success = remove(uid, hard_delete=True)
        assert success is True

        # Should not be retrievable
        retrieved = get(uid)
        assert retrieved is None

    def test_delete_nonexistent(self):
        """Should return False for nonexistent UID."""
        success = remove("nonexistent-delete-uid")
        assert success is False


class TestScriptureStats:
    """Tests for scripture statistics."""

    @pytest.mark.asyncio
    async def test_stats(self):
        """Should return valid stats."""
        # Ensure at least one scripture exists
        await store(
            source="Stats test content",
            title="Stats Test",
            original_name="stats.txt",
            auto_index=False,
        )

        result = stats()

        assert result is not None
        assert "total" in result
        assert "indexed" in result
        assert "by_type" in result
        assert "total_size_mb" in result
        assert isinstance(result["by_type"], dict)
