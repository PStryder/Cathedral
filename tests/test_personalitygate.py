"""
Tests for PersonalityGate personality management.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from cathedral.PersonalityGate import (
    PersonalityManager,
    load,
    get_default,
    create,
    list_all,
    exists,
)
from cathedral.PersonalityGate.models import Personality, LLMConfig


class TestPersonalityManagerInitialization:
    """Tests for PersonalityManager initialization."""

    def test_initialize_creates_builtins(self, temp_data_dir):
        """Initialize should create builtin personalities."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            # Should have created default personality
            assert (personalities_dir / "default.json").exists()

    def test_initialize_is_idempotent(self, temp_data_dir):
        """Multiple initialize calls should be safe."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}

            PersonalityManager.initialize()
            PersonalityManager.initialize()
            PersonalityManager.initialize()

            assert PersonalityManager._initialized is True


class TestPersonalityLoading:
    """Tests for personality loading."""

    def test_load_builtin_personality(self, temp_data_dir):
        """Should load builtin personalities."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            default = PersonalityManager.load("default")

            assert default is not None
            assert default.id == "default"

    def test_load_nonexistent_returns_none(self, temp_data_dir):
        """Should return None for nonexistent personality."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            result = PersonalityManager.load("nonexistent_12345")

            assert result is None

    def test_load_caches_personality(self, temp_data_dir):
        """Should cache loaded personalities."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            PersonalityManager.load("default")

            assert "default" in PersonalityManager._cache

    def test_get_default(self, temp_data_dir):
        """Should return default personality."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}

            default = PersonalityManager.get_default()

            assert default is not None
            assert isinstance(default, Personality)


class TestPersonalityCreation:
    """Tests for personality creation."""

    def test_create_personality(self, temp_data_dir):
        """Should create a new personality."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            personality = PersonalityManager.create(
                name="Test Bot",
                description="A test personality",
                system_prompt="You are a test bot.",
                temperature=0.5
            )

            assert personality is not None
            assert personality.name == "Test Bot"
            assert personality.llm_config.temperature == 0.5
            assert (personalities_dir / f"{personality.id}.json").exists()

    def test_create_generates_unique_id(self, temp_data_dir):
        """Should generate unique IDs for personalities."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            p1 = PersonalityManager.create(name="Same Name")
            p2 = PersonalityManager.create(name="Same Name")

            assert p1.id != p2.id

    def test_create_with_custom_id(self, temp_data_dir):
        """Should use provided ID."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            personality = PersonalityManager.create(
                name="Custom ID Bot",
                personality_id="my_custom_id"
            )

            assert personality.id == "my_custom_id"


class TestPersonalityUpdate:
    """Tests for personality updates."""

    def test_update_personality(self, temp_data_dir):
        """Should update personality fields."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            personality = PersonalityManager.create(name="Update Me")
            updated = PersonalityManager.update(
                personality.id,
                {"description": "Updated description"}
            )

            assert updated is not None
            assert updated.description == "Updated description"

    def test_update_nested_fields(self, temp_data_dir):
        """Should update nested fields."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            personality = PersonalityManager.create(name="Nested Update")
            updated = PersonalityManager.update(
                personality.id,
                {"llm_config": {"temperature": 0.9}}
            )

            assert updated.llm_config.temperature == 0.9

    def test_cannot_update_builtin(self, temp_data_dir):
        """Should not allow updating builtin personalities."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            with pytest.raises(ValueError, match="builtin"):
                PersonalityManager.update("default", {"description": "Hacked"})


class TestPersonalityDeletion:
    """Tests for personality deletion."""

    def test_delete_personality(self, temp_data_dir):
        """Should delete a personality."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            personality = PersonalityManager.create(name="Delete Me")
            personality_id = personality.id

            result = PersonalityManager.delete(personality_id)

            assert result is True
            assert not (personalities_dir / f"{personality_id}.json").exists()
            assert PersonalityManager.load(personality_id) is None

    def test_cannot_delete_builtin(self, temp_data_dir):
        """Should not allow deleting builtin personalities."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            with pytest.raises(ValueError, match="builtin"):
                PersonalityManager.delete("default")

    def test_delete_nonexistent_returns_false(self, temp_data_dir):
        """Should return False for nonexistent personality."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            result = PersonalityManager.delete("nonexistent_12345")

            assert result is False


class TestPersonalityListing:
    """Tests for personality listing."""

    def test_list_all(self, temp_data_dir):
        """Should list all personalities."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            PersonalityManager.create(name="List Test 1")
            PersonalityManager.create(name="List Test 2")

            all_personalities = PersonalityManager.list_all()

            assert len(all_personalities) >= 2
            assert any(p["name"] == "List Test 1" for p in all_personalities)

    def test_list_excludes_builtins(self, temp_data_dir):
        """Should optionally exclude builtins."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            PersonalityManager.create(name="Custom Only")

            custom_only = PersonalityManager.list_all(include_builtins=False)

            assert all(not p.get("is_builtin", False) for p in custom_only)

    def test_list_by_category(self, temp_data_dir):
        """Should filter by category."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            PersonalityManager.create(name="Cat1", category="category_a")
            PersonalityManager.create(name="Cat2", category="category_b")

            cat_a = PersonalityManager.list_all(category="category_a")

            assert all(p["category"] == "category_a" for p in cat_a)


class TestPersonalityDuplication:
    """Tests for personality duplication."""

    def test_duplicate_personality(self, temp_data_dir):
        """Should duplicate a personality."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            original = PersonalityManager.create(
                name="Original",
                system_prompt="Original prompt"
            )

            duplicate = PersonalityManager.duplicate(original.id, "Copy of Original")

            assert duplicate is not None
            assert duplicate.id != original.id
            assert duplicate.name == "Copy of Original"
            assert duplicate.llm_config.system_prompt == original.llm_config.system_prompt
            assert duplicate.metadata.is_builtin is False


class TestPersonalityExportImport:
    """Tests for personality export/import."""

    def test_export_personality(self, temp_data_dir):
        """Should export personality to file."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)
        export_path = temp_data_dir / "exported.json"

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            personality = PersonalityManager.create(name="Export Me")

            result = PersonalityManager.export_to_file(personality.id, str(export_path))

            assert result is True
            assert export_path.exists()

            # Verify exported data
            with open(export_path) as f:
                data = json.load(f)
            assert data["name"] == "Export Me"

    def test_import_personality(self, temp_data_dir):
        """Should import personality from file."""
        personalities_dir = temp_data_dir / "personalities"
        personalities_dir.mkdir(parents=True, exist_ok=True)
        import_path = temp_data_dir / "to_import.json"

        # Create import file
        import_data = {
            "id": "imported",
            "name": "Imported Bot",
            "description": "An imported personality",
            "llm_config": {
                "model": "openai/gpt-4o",
                "temperature": 0.7,
                "max_tokens": 2000,
                "system_prompt": "You are imported."
            },
            "behavior": {"style_tags": [], "response_length": "medium"},
            "memory": {"enabled": True, "domains": []},
            "metadata": {
                "category": "imported",
                "author": "external",
                "is_default": False,
                "is_builtin": False,
                "usage_count": 0
            }
        }
        with open(import_path, "w") as f:
            json.dump(import_data, f)

        with patch('cathedral.PersonalityGate.PERSONALITIES_DIR', personalities_dir):
            PersonalityManager._initialized = False
            PersonalityManager._cache = {}
            PersonalityManager.initialize()

            imported = PersonalityManager.import_from_file(str(import_path))

            assert imported is not None
            assert imported.name == "Imported Bot"
            assert imported.metadata.is_builtin is False


class TestPersonalityModel:
    """Tests for Personality model."""

    def test_personality_to_dict(self, mock_personality_data):
        """Should serialize to dict."""
        personality = Personality.from_dict(mock_personality_data)

        data = personality.to_dict()

        assert data["id"] == "test_personality"
        assert data["name"] == "Test Personality"
        assert "llm_config" in data

    def test_personality_from_dict(self, mock_personality_data):
        """Should deserialize from dict."""
        personality = Personality.from_dict(mock_personality_data)

        assert personality.id == "test_personality"
        assert personality.llm_config.model == "openai/gpt-4o"
        assert personality.llm_config.temperature == 0.7
