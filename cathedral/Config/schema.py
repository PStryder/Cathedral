"""
Configuration schema for Cathedral.

Defines all configurable options with metadata for validation,
documentation, and UI generation.
"""

from enum import Enum
from typing import Optional, List, Any
from dataclasses import dataclass, field


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    SECRET = "secret"      # Masked in UI, never logged
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    PATH = "path"          # File system path
    URL = "url"
    LIST = "list"          # Comma-separated values


class ConfigCategory(Enum):
    """Configuration categories for UI grouping."""
    API_KEYS = "api_keys"
    DATABASE = "database"
    PATHS = "paths"
    SERVER = "server"
    MODELS = "models"
    FEATURES = "features"


@dataclass
class ConfigField:
    """Definition of a configuration field."""
    key: str
    description: str
    config_type: ConfigType
    category: ConfigCategory
    required: bool = False
    default: Any = None
    env_var: str = None          # Override env var name (defaults to key)
    validation: str = None       # Regex pattern or validation hint
    options: List[str] = None    # For enumerated types
    sensitive: bool = False      # Extra protection for secrets
    restart_required: bool = False
    depends_on: str = None       # Key of another config this depends on

    def __post_init__(self):
        if self.env_var is None:
            self.env_var = self.key
        if self.config_type == ConfigType.SECRET:
            self.sensitive = True


# ==================== Schema Definition ====================

CONFIG_SCHEMA: List[ConfigField] = [
    # === API Keys ===
    ConfigField(
        key="OPENROUTER_API_KEY",
        description="API key for OpenRouter LLM access (required for chat)",
        config_type=ConfigType.SECRET,
        category=ConfigCategory.API_KEYS,
        required=True,
        validation=r"^sk-or-.*",
    ),
    ConfigField(
        key="OPENAI_API_KEY",
        description="OpenAI API key for embeddings and audio transcription",
        config_type=ConfigType.SECRET,
        category=ConfigCategory.API_KEYS,
        required=False,
        validation=r"^sk-.*",
    ),

    # === Database ===
    ConfigField(
        key="DATABASE_URL",
        description="PostgreSQL connection URL (with pgvector extension)",
        config_type=ConfigType.URL,
        category=ConfigCategory.DATABASE,
        required=True,
        default="postgresql://user:password@localhost:5432/cathedral",
        validation=r"^postgres(ql)?://.*",
    ),
    ConfigField(
        key="DB_BACKEND",
        description="Database backend type",
        config_type=ConfigType.STRING,
        category=ConfigCategory.DATABASE,
        required=False,
        default="postgres",
        options=["postgres", "sqlite"],
    ),
    ConfigField(
        key="VECTOR_BACKEND",
        description="Vector storage backend for embeddings",
        config_type=ConfigType.STRING,
        category=ConfigCategory.DATABASE,
        required=False,
        default="pgvector",
        options=["pgvector", "faiss"],
    ),
    ConfigField(
        key="AUTO_MIGRATE_ON_STARTUP",
        description="Automatically run database migrations on startup",
        config_type=ConfigType.BOOLEAN,
        category=ConfigCategory.DATABASE,
        required=False,
        default=True,
    ),

    # === Paths ===
    ConfigField(
        key="DATA_DIR",
        description="Root directory for all data storage",
        config_type=ConfigType.PATH,
        category=ConfigCategory.PATHS,
        required=False,
        default="./data",
    ),
    ConfigField(
        key="SCRIPTURE_DIR",
        description="Directory for scripture file storage",
        config_type=ConfigType.PATH,
        category=ConfigCategory.PATHS,
        required=False,
        default="./data/scripture",
    ),
    ConfigField(
        key="AGENTS_DIR",
        description="Directory for sub-agent task data",
        config_type=ConfigType.PATH,
        category=ConfigCategory.PATHS,
        required=False,
        default="./data/agents",
    ),
    ConfigField(
        key="MODELS_DIR",
        description="Directory for local model files (LoomMirror)",
        config_type=ConfigType.PATH,
        category=ConfigCategory.PATHS,
        required=False,
        default="./models",
    ),

    # === Server ===
    ConfigField(
        key="HOST",
        description="Server bind address",
        config_type=ConfigType.STRING,
        category=ConfigCategory.SERVER,
        required=False,
        default="0.0.0.0",
    ),
    ConfigField(
        key="PORT",
        description="Server port",
        config_type=ConfigType.INTEGER,
        category=ConfigCategory.SERVER,
        required=False,
        default=8000,
    ),
    ConfigField(
        key="ALLOWED_ORIGINS",
        description="CORS allowed origins (comma-separated)",
        config_type=ConfigType.LIST,
        category=ConfigCategory.SERVER,
        required=False,
        default="http://localhost:8000,http://localhost:5000",
    ),
    ConfigField(
        key="DEBUG",
        description="Enable debug mode (verbose logging, auto-reload)",
        config_type=ConfigType.BOOLEAN,
        category=ConfigCategory.SERVER,
        required=False,
        default=False,
    ),
    ConfigField(
        key="LOG_LEVEL",
        description="Logging verbosity",
        config_type=ConfigType.STRING,
        category=ConfigCategory.SERVER,
        required=False,
        default="INFO",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
    ),

    # === Models ===
    ConfigField(
        key="DEFAULT_MODEL",
        description="Default LLM model for chat",
        config_type=ConfigType.STRING,
        category=ConfigCategory.MODELS,
        required=False,
        default="openai/gpt-4o-2024-11-20",
    ),
    ConfigField(
        key="VISION_MODEL",
        description="Model for image analysis",
        config_type=ConfigType.STRING,
        category=ConfigCategory.MODELS,
        required=False,
        default="openai/gpt-4o-2024-11-20",
    ),
    ConfigField(
        key="LLM_BACKEND",
        description="LLM backend provider for StarMirror",
        config_type=ConfigType.STRING,
        category=ConfigCategory.MODELS,
        required=False,
        default="openrouter",
        options=["openrouter", "claude_cli", "codex_cli"],
    ),
    ConfigField(
        key="CLAUDE_CLI_CMD",
        description="Claude CLI command (used when LLM_BACKEND=claude_cli)",
        config_type=ConfigType.STRING,
        category=ConfigCategory.MODELS,
        required=False,
        default="claude",
    ),
    ConfigField(
        key="CODEX_CLI_CMD",
        description="Codex CLI command (used when LLM_BACKEND=codex_cli)",
        config_type=ConfigType.STRING,
        category=ConfigCategory.MODELS,
        required=False,
        default="codex",
    ),
    ConfigField(
        key="EMBEDDING_MODEL",
        description="Model for text embeddings",
        config_type=ConfigType.STRING,
        category=ConfigCategory.MODELS,
        required=False,
        default="text-embedding-3-small",
    ),
    ConfigField(
        key="EMBEDDING_DIM",
        description="Embedding vector dimension",
        config_type=ConfigType.INTEGER,
        category=ConfigCategory.MODELS,
        required=False,
        default=1536,
    ),
    ConfigField(
        key="LOOMMIRROR_MODEL_PATH",
        description="Path to local LoomMirror model (GGUF)",
        config_type=ConfigType.PATH,
        category=ConfigCategory.MODELS,
        required=False,
        default="./models/memory/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    ),

    # === Features ===
    ConfigField(
        key="ENABLE_MEMORY_GATE",
        description="Enable MemoryGate semantic memory",
        config_type=ConfigType.BOOLEAN,
        category=ConfigCategory.FEATURES,
        required=False,
        default=True,
    ),
    ConfigField(
        key="ENABLE_SCRIPTURE_RAG",
        description="Enable ScriptureGate RAG context injection",
        config_type=ConfigType.BOOLEAN,
        category=ConfigCategory.FEATURES,
        required=False,
        default=True,
    ),
    ConfigField(
        key="ENABLE_SUBAGENTS",
        description="Enable sub-agent spawning",
        config_type=ConfigType.BOOLEAN,
        category=ConfigCategory.FEATURES,
        required=False,
        default=True,
    ),
    ConfigField(
        key="ENABLE_MULTIMODAL",
        description="Enable image and audio processing",
        config_type=ConfigType.BOOLEAN,
        category=ConfigCategory.FEATURES,
        required=False,
        default=True,
    ),
    ConfigField(
        key="AUTO_EXTRACT_MEMORY",
        description="Automatically extract memories from conversations",
        config_type=ConfigType.BOOLEAN,
        category=ConfigCategory.FEATURES,
        required=False,
        default=True,
    ),
    ConfigField(
        key="MEMORYGATE_TENANCY_MODE",
        description="MemoryGate multi-tenancy mode",
        config_type=ConfigType.STRING,
        category=ConfigCategory.FEATURES,
        required=False,
        default="single",
        options=["single", "multi"],
    ),
]


def get_schema_by_key(key: str) -> Optional[ConfigField]:
    """Get schema field by key."""
    for field in CONFIG_SCHEMA:
        if field.key == key:
            return field
    return None


def get_schema_by_category(category: ConfigCategory) -> List[ConfigField]:
    """Get all fields in a category."""
    return [f for f in CONFIG_SCHEMA if f.category == category]


def get_required_fields() -> List[ConfigField]:
    """Get all required fields."""
    return [f for f in CONFIG_SCHEMA if f.required]


def schema_to_dict() -> dict:
    """Convert schema to dict for API/UI."""
    result = {}
    for cat in ConfigCategory:
        fields = get_schema_by_category(cat)
        result[cat.value] = [
            {
                "key": f.key,
                "description": f.description,
                "type": f.config_type.value,
                "required": f.required,
                "default": f.default,
                "options": f.options,
                "sensitive": f.sensitive,
                "restart_required": f.restart_required,
            }
            for f in fields
        ]
    return result
