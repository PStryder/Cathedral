"""
PersonalityGate - Agent personality management for Cathedral.

Provides:
- Personality configuration storage and retrieval
- Built-in personalities for common use cases
- Thread-personality association
- SubAgent personality support
- Export/import for sharing
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from cathedral.shared.gate import (
    GateLogger,
    ConfigLoader,
    build_health_status,
    deep_update,
)

from .models import Personality, PersonalitySnapshot, LLMConfig
from .defaults import BUILTIN_PERSONALITIES, get_builtin_personalities

# Logger for this gate
_log = GateLogger.get("PersonalityGate")

# Storage paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONALITIES_DIR = PROJECT_ROOT / "data" / "personalities"
PERSONALITIES_DIR.mkdir(parents=True, exist_ok=True)


class PersonalityManager:
    """Manages personality configurations."""

    _cache: Dict[str, Personality] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls):
        """Initialize personality system, creating builtins if needed."""
        if cls._initialized:
            return

        # Ensure builtins exist
        for personality in BUILTIN_PERSONALITIES:
            file_path = PERSONALITIES_DIR / f"{personality.id}.json"
            if not file_path.exists():
                cls._save_to_file(personality)
            cls._cache[personality.id] = personality

        cls._initialized = True

    @classmethod
    def _save_to_file(cls, personality: Personality):
        """Save personality to JSON file."""
        file_path = PERSONALITIES_DIR / f"{personality.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(personality.to_dict(), f, indent=2, default=str)

    @classmethod
    def _load_from_file(cls, personality_id: str) -> Optional[Personality]:
        """Load personality from JSON file."""
        file_path = PERSONALITIES_DIR / f"{personality_id}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Personality.from_dict(data)
        except (json.JSONDecodeError, Exception) as e:
            _log.error(f"Error loading {personality_id}: {e}")
            return None

    @classmethod
    def load(cls, personality_id: str) -> Optional[Personality]:
        """
        Load a personality by ID.

        Checks cache first, then file system.
        """
        cls.initialize()

        # Check cache
        if personality_id in cls._cache:
            return cls._cache[personality_id]

        # Load from file
        personality = cls._load_from_file(personality_id)
        if personality:
            cls._cache[personality_id] = personality

        return personality

    @classmethod
    def get_default(cls) -> Personality:
        """Get the default personality."""
        cls.initialize()
        default = cls.load("default")
        if default:
            return default
        # Fallback to first builtin
        return BUILTIN_PERSONALITIES[0]

    # ==================== Validation ====================

    @classmethod
    def validate_llm_config(
        cls,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None,
    ) -> tuple[bool, List[str]]:
        """
        Validate LLM configuration values.

        Args:
            model: Model identifier to validate
            temperature: Temperature value to validate
            max_tokens: Max tokens value to validate
            system_prompt: System prompt to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if model is not None:
            # Check for common model format issues
            if not model.strip():
                errors.append("Model cannot be empty")
            elif "/" not in model and not model.startswith("gpt-"):
                # OpenRouter format: provider/model or OpenAI format: gpt-*
                _log.warning(f"Model '{model}' may not be in standard format (provider/model)")

        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                errors.append(f"Temperature must be between 0.0 and 2.0, got {temperature}")

        if max_tokens is not None:
            if not 100 <= max_tokens <= 128000:
                errors.append(f"Max tokens must be between 100 and 128000, got {max_tokens}")

        if system_prompt is not None:
            if not system_prompt.strip():
                errors.append("System prompt cannot be empty")
            elif len(system_prompt) > 100000:
                errors.append(f"System prompt too long ({len(system_prompt)} chars, max 100000)")

        return len(errors) == 0, errors

    # ==================== Health Checks ====================

    @classmethod
    def is_healthy(cls) -> bool:
        """Check if the gate is operational."""
        return cls._initialized and PERSONALITIES_DIR.exists()

    @classmethod
    def get_health_status(cls) -> Dict[str, Any]:
        """Get detailed health information."""
        checks = {
            "personalities_dir_exists": PERSONALITIES_DIR.exists(),
            "default_personality_exists": cls.exists("default") if cls._initialized else False,
        }

        details = {
            "personalities_dir": str(PERSONALITIES_DIR),
            "cached_count": len(cls._cache),
            "builtin_count": len(BUILTIN_PERSONALITIES),
        }

        if cls._initialized:
            try:
                all_personalities = list(PERSONALITIES_DIR.glob("*.json"))
                details["total_count"] = len(all_personalities)
            except Exception:
                details["total_count"] = "unknown"

        return build_health_status(
            gate_name="PersonalityGate",
            initialized=cls._initialized,
            dependencies=["filesystem"],
            checks=checks,
            details=details,
        )

    @classmethod
    def get_dependencies(cls) -> List[str]:
        """List external dependencies."""
        return ["filesystem"]

    @classmethod
    def create(
        cls,
        name: str,
        personality_id: str = None,
        description: str = "",
        system_prompt: str = "You are a helpful assistant.",
        model: str = "openai/gpt-4o-2024-11-20",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        style_tags: List[str] = None,
        memory_domains: List[str] = None,
        category: str = "custom"
    ) -> Personality:
        """
        Create a new personality.

        Args:
            name: Display name
            personality_id: Unique ID (auto-generated from name if not provided)
            description: Brief description
            system_prompt: The system prompt
            model: LLM model to use
            temperature: Response temperature
            max_tokens: Max response tokens
            style_tags: Behavioral style tags
            memory_domains: Memory domains to prioritize
            category: Category for organization

        Returns:
            The created Personality

        Raises:
            ValueError: If LLM configuration is invalid
        """
        cls.initialize()

        # Validate LLM config
        is_valid, errors = cls.validate_llm_config(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        if not is_valid:
            raise ValueError(f"Invalid LLM configuration: {'; '.join(errors)}")

        # Generate ID if not provided
        if not personality_id:
            personality_id = name.lower().replace(" ", "_").replace("-", "_")
            # Ensure uniqueness
            base_id = personality_id
            counter = 1
            while cls.exists(personality_id):
                personality_id = f"{base_id}_{counter}"
                counter += 1

        personality = Personality(
            id=personality_id,
            name=name,
            description=description,
            llm_config=LLMConfig(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            ),
            metadata={
                "category": category,
                "author": "user",
                "is_default": False,
                "is_builtin": False
            }
        )

        if style_tags:
            personality.behavior.style_tags = style_tags
        if memory_domains:
            personality.memory.domains = memory_domains

        # Save and cache
        cls._save_to_file(personality)
        cls._cache[personality_id] = personality

        return personality

    @classmethod
    def update(cls, personality_id: str, updates: Dict[str, Any]) -> Optional[Personality]:
        """
        Update an existing personality.

        Args:
            personality_id: ID of personality to update
            updates: Dict of fields to update (can be nested)

        Returns:
            Updated Personality or None if not found
        """
        personality = cls.load(personality_id)
        if not personality:
            return None

        # Don't allow modifying builtins directly (create a copy instead)
        if personality.metadata.is_builtin:
            raise ValueError("Cannot modify builtin personalities. Create a copy instead.")

        # Apply updates
        current = personality.to_dict()
        cls._deep_update(current, updates)
        current["updated_at"] = datetime.utcnow().isoformat()

        # Recreate and save
        personality = Personality.from_dict(current)
        cls._save_to_file(personality)
        cls._cache[personality_id] = personality

        return personality

    @classmethod
    def _deep_update(cls, base: dict, updates: dict):
        """Deep merge updates into base dict."""
        # Use shared utility
        deep_update(base, updates)

    @classmethod
    def delete(cls, personality_id: str) -> bool:
        """
        Delete a personality.

        Returns False if builtin or not found.
        """
        personality = cls.load(personality_id)
        if not personality:
            return False

        if personality.metadata.is_builtin:
            raise ValueError("Cannot delete builtin personalities.")

        # Remove file
        file_path = PERSONALITIES_DIR / f"{personality_id}.json"
        if file_path.exists():
            file_path.unlink()

        # Remove from cache
        cls._cache.pop(personality_id, None)

        return True

    @classmethod
    def exists(cls, personality_id: str) -> bool:
        """Check if a personality exists."""
        cls.initialize()
        if personality_id in cls._cache:
            return True
        file_path = PERSONALITIES_DIR / f"{personality_id}.json"
        return file_path.exists()

    @classmethod
    def list_all(cls, category: str = None, include_builtins: bool = True) -> List[Dict]:
        """
        List all available personalities.

        Args:
            category: Filter by category
            include_builtins: Whether to include builtin personalities

        Returns:
            List of personality summaries
        """
        cls.initialize()

        results = []

        # Scan directory for all personalities
        for file_path in PERSONALITIES_DIR.glob("*.json"):
            personality_id = file_path.stem
            personality = cls.load(personality_id)

            if not personality:
                continue

            if not include_builtins and personality.metadata.is_builtin:
                continue

            if category and personality.metadata.category != category:
                continue

            results.append({
                "id": personality.id,
                "name": personality.name,
                "description": personality.description,
                "category": personality.metadata.category,
                "model": personality.llm_config.model,
                "is_builtin": personality.metadata.is_builtin,
                "is_default": personality.metadata.is_default,
                "usage_count": personality.metadata.usage_count,
                "style_tags": personality.behavior.style_tags
            })

        # Sort: default first, then by usage, then alphabetically
        results.sort(key=lambda x: (
            not x["is_default"],
            -x["usage_count"],
            x["name"].lower()
        ))

        return results

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of unique categories."""
        cls.initialize()
        categories = set()
        for file_path in PERSONALITIES_DIR.glob("*.json"):
            personality = cls.load(file_path.stem)
            if personality:
                categories.add(personality.metadata.category)
        return sorted(categories)

    @classmethod
    def export_to_file(cls, personality_id: str, output_path: str) -> bool:
        """Export a personality to a standalone JSON file."""
        personality = cls.load(personality_id)
        if not personality:
            return False

        # Create export version (without usage stats)
        export_data = personality.to_dict()
        export_data["metadata"]["usage_count"] = 0
        export_data["metadata"]["is_builtin"] = False

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        return True

    @classmethod
    def import_from_file(cls, file_path: str, new_id: str = None) -> Optional[Personality]:
        """
        Import a personality from a JSON file.

        Args:
            file_path: Path to the personality JSON file
            new_id: Optional new ID (to avoid conflicts)

        Returns:
            The imported Personality or None on error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Override ID if provided
            if new_id:
                data["id"] = new_id

            # Ensure unique ID
            base_id = data["id"]
            counter = 1
            while cls.exists(data["id"]):
                data["id"] = f"{base_id}_{counter}"
                counter += 1

            # Mark as user-created
            data["metadata"]["is_builtin"] = False
            data["metadata"]["author"] = "imported"
            data["created_at"] = datetime.utcnow().isoformat()
            data["updated_at"] = datetime.utcnow().isoformat()

            personality = Personality.from_dict(data)
            cls._save_to_file(personality)
            cls._cache[personality.id] = personality

            return personality

        except Exception as e:
            _log.error(f"Import error: {e}")
            return None

    @classmethod
    def duplicate(cls, personality_id: str, new_name: str) -> Optional[Personality]:
        """
        Create a copy of an existing personality.

        Useful for customizing builtin personalities.
        """
        original = cls.load(personality_id)
        if not original:
            return None

        # Create new ID
        new_id = new_name.lower().replace(" ", "_")
        counter = 1
        while cls.exists(new_id):
            new_id = f"{new_name.lower().replace(' ', '_')}_{counter}"
            counter += 1

        # Copy data
        data = original.to_dict()
        data["id"] = new_id
        data["name"] = new_name
        data["metadata"]["is_builtin"] = False
        data["metadata"]["is_default"] = False
        data["metadata"]["author"] = "user"
        data["metadata"]["usage_count"] = 0
        data["created_at"] = datetime.utcnow().isoformat()
        data["updated_at"] = datetime.utcnow().isoformat()

        personality = Personality.from_dict(data)
        cls._save_to_file(personality)
        cls._cache[personality.id] = personality

        return personality

    @classmethod
    def create_snapshot(cls, personality: Personality) -> PersonalitySnapshot:
        """Create a minimal snapshot for thread history."""
        return PersonalitySnapshot(
            id=personality.id,
            name=personality.name,
            system_prompt=personality.llm_config.system_prompt,
            model=personality.llm_config.model,
            temperature=personality.llm_config.temperature
        )

    @classmethod
    def record_usage(cls, personality_id: str):
        """Record that a personality was used."""
        personality = cls.load(personality_id)
        if personality and not personality.metadata.is_builtin:
            personality.increment_usage()
            cls._save_to_file(personality)


# Convenience functions
def load(personality_id: str) -> Optional[Personality]:
    """Load a personality by ID."""
    return PersonalityManager.load(personality_id)


def get_default() -> Personality:
    """Get the default personality."""
    return PersonalityManager.get_default()


def create(**kwargs) -> Personality:
    """Create a new personality."""
    return PersonalityManager.create(**kwargs)


def list_all(**kwargs) -> List[Dict]:
    """List all personalities."""
    return PersonalityManager.list_all(**kwargs)


def exists(personality_id: str) -> bool:
    """Check if personality exists."""
    return PersonalityManager.exists(personality_id)


def export_personality(personality_id: str, output_path: str) -> bool:
    """Export a personality to file."""
    return PersonalityManager.export_to_file(personality_id, output_path)


def import_personality(file_path: str, new_id: str = None) -> Optional[Personality]:
    """Import a personality from file."""
    return PersonalityManager.import_from_file(file_path, new_id)


def initialize():
    """Initialize the personality system."""
    PersonalityManager.initialize()
