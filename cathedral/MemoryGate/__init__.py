"""
MemoryGate integration for Cathedral.
Direct import of MemoryGate core services (no MCP).
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from cathedral.shared.gate import (
    GateLogger,
    GateErrorHandler,
    PathUtils,
    build_health_status,
)

# Logger for this gate
_log = GateLogger.get("MemoryGate")

# Load Cathedral's .env which should have MemoryGate config
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

# Try to find MemoryGate in various locations
MEMORYGATE_PATH = None
_memorygate_available = False

# Candidate paths (in priority order)
_candidate_paths = [
    Path(__file__).resolve().parents[3] / "lv_stack" / "MemoryGate",
    Path(__file__).resolve().parents[2] / "MemoryGate",
    Path(os.environ.get("MEMORYGATE_PATH", "")) if os.environ.get("MEMORYGATE_PATH") else None,
]

# Find module using PathUtils pattern
for candidate in _candidate_paths:
    if candidate and candidate.exists() and (candidate / "core").exists():
        MEMORYGATE_PATH = str(candidate)
        break

if MEMORYGATE_PATH:
    if MEMORYGATE_PATH not in sys.path:
        sys.path.insert(0, MEMORYGATE_PATH)

    # Set defaults for MemoryGate config before importing
    os.environ.setdefault("DB_BACKEND", "postgres")
    os.environ.setdefault("VECTOR_BACKEND", "pgvector")
    os.environ.setdefault("MEMORYGATE_TENANCY_MODE", "single")
    os.environ.setdefault("AUTO_MIGRATE_ON_STARTUP", "true")
    os.environ.setdefault("AUTO_CREATE_EXTENSIONS", "true")
    os.environ.setdefault("LOG_LEVEL", "WARNING")  # Reduce noise

    # Now import MemoryGate modules
    try:
        from core.db import init_db, DB
        from core.context import RequestContext, AuthContext
        from core.services import memory_service
        _memorygate_available = True
    except ImportError as e:
        _log.warning(f"Failed to import core modules: {e}")
        init_db = None
        DB = None
        RequestContext = None
        AuthContext = None
        memory_service = None
else:
    _log.warning("MemoryGate path not found. Knowledge features will be disabled.")
    _log.info("Set MEMORYGATE_PATH environment variable to enable.")
    init_db = None
    DB = None
    RequestContext = None
    AuthContext = None
    memory_service = None

# Module-level state
_initialized = False
_context: Optional[RequestContext] = None


def initialize() -> bool:
    """
    Initialize MemoryGate database connection.
    Returns True if successful, False if configuration is missing.
    """
    global _initialized, _context

    if _initialized:
        return True

    # Check if MemoryGate modules are available
    if not _memorygate_available or init_db is None:
        _log.warning("Core modules not available - memory features disabled")
        return False

    # Check required config
    database_url = os.environ.get("DATABASE_URL")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not database_url:
        _log.warning("DATABASE_URL not set - memory features disabled")
        return False

    if not openai_key:
        _log.warning("OPENAI_API_KEY not set - embeddings disabled")
        # Continue anyway, search will just not work well

    try:
        init_db()
        _context = RequestContext(
            auth=AuthContext(tenant_id="cathedral"),
            agent_uuid="ag_CATHEDRAL"
        )
        _initialized = True
        _log.info("Initialized successfully")
        return True
    except Exception as e:
        _log.error(f"Initialization failed: {e}")
        return False


# ==================== Health Checks ====================


def is_healthy() -> bool:
    """Check if the gate is operational."""
    return _initialized and _memorygate_available


def get_health_status() -> Dict[str, Any]:
    """Get detailed health information."""
    checks = {
        "memorygate_available": _memorygate_available,
        "database_url_set": bool(os.environ.get("DATABASE_URL")),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
    }
    details = {
        "memorygate_path": MEMORYGATE_PATH,
    }

    return build_health_status(
        gate_name="MemoryGate",
        initialized=_initialized,
        dependencies=["MemoryGate external", "PostgreSQL", "OpenAI API"],
        checks=checks,
        details=details,
    )


def get_dependencies() -> List[str]:
    """List external dependencies."""
    return ["MemoryGate external", "PostgreSQL", "OpenAI API"]


def is_initialized() -> bool:
    """Check if MemoryGate is ready."""
    return _initialized


def get_context() -> Optional[RequestContext]:
    """Get the request context for MemoryGate calls."""
    if not _initialized:
        if not initialize():
            return None
    return _context


# === Storage Operations ===

def store_observation(
    text: str,
    confidence: float = 0.8,
    domain: str = None,
    evidence: list = None
) -> Optional[dict]:
    """Store an observation with embedding."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_store(
            observation=text,
            confidence=confidence,
            domain=domain,
            evidence=evidence,
            context=ctx
        )
    except Exception as e:
        _log.error(f"store_observation failed: {e}")
        return None


def store_pattern(
    category: str,
    name: str,
    text: str,
    confidence: float = 0.8,
    evidence_ids: list = None
) -> Optional[dict]:
    """Create or update a pattern (synthesized understanding)."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_update_pattern(
            category=category,
            pattern_name=name,
            pattern_text=text,
            confidence=confidence,
            evidence_observation_ids=evidence_ids,
            context=ctx
        )
    except Exception as e:
        _log.error(f"store_pattern failed: {e}")
        return None


def store_concept(
    name: str,
    concept_type: str,
    description: str,
    domain: str = None,
    status: str = None
) -> Optional[dict]:
    """Store a concept in the knowledge graph."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_store_concept(
            name=name,
            concept_type=concept_type,
            description=description,
            domain=domain,
            status=status,
            context=ctx
        )
    except Exception as e:
        _log.error(f" store_concept failed: {e}")
        return None


# === Search Operations ===

def search(
    query: str,
    limit: int = 5,
    min_confidence: float = 0.0,
    domain: str = None,
    include_cold: bool = False
) -> list:
    """Semantic search across all memory types."""
    ctx = get_context()
    if not ctx:
        return []

    try:
        result = memory_service.memory_search(
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            domain=domain,
            include_cold=include_cold,
            context=ctx
        )
        return result.get("results", [])
    except Exception as e:
        _log.error(f" search failed: {e}")
        return []


def recall(
    domain: str = None,
    limit: int = 10,
    min_confidence: float = 0.0
) -> list:
    """Recall recent observations, optionally filtered by domain."""
    ctx = get_context()
    if not ctx:
        return []

    try:
        result = memory_service.memory_recall(
            domain=domain,
            limit=limit,
            min_confidence=min_confidence,
            context=ctx
        )
        return result.get("observations", [])
    except Exception as e:
        _log.error(f" recall failed: {e}")
        return []


def get_stats() -> Optional[dict]:
    """Get memory system statistics."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_stats()
    except Exception as e:
        _log.error(f" get_stats failed: {e}")
        return None


# === Pattern Operations ===

def get_pattern(category: str, name: str) -> Optional[dict]:
    """Get a specific pattern."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_get_pattern(
            category=category,
            pattern_name=name,
            context=ctx
        )
    except Exception as e:
        _log.error(f" get_pattern failed: {e}")
        return None


def list_patterns(
    category: str = None,
    limit: int = 20,
    min_confidence: float = 0.0
) -> list:
    """List patterns, optionally filtered by category."""
    ctx = get_context()
    if not ctx:
        return []

    try:
        result = memory_service.memory_patterns(
            category=category,
            limit=limit,
            min_confidence=min_confidence,
            context=ctx
        )
        return result.get("patterns", [])
    except Exception as e:
        _log.error(f" list_patterns failed: {e}")
        return []


# === Concept Operations ===

def get_concept(name: str) -> Optional[dict]:
    """Look up a concept by name (case-insensitive)."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_get_concept(
            name=name,
            context=ctx
        )
    except Exception as e:
        _log.error(f" get_concept failed: {e}")
        return None


def add_concept_alias(concept_name: str, alias: str) -> Optional[dict]:
    """Add an alias for a concept."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_add_concept_alias(
            concept_name=concept_name,
            alias=alias,
            context=ctx
        )
    except Exception as e:
        _log.error(f" add_concept_alias failed: {e}")
        return None


def add_concept_relationship(
    from_concept: str,
    to_concept: str,
    rel_type: str,
    weight: float = 0.5,
    description: str = None
) -> Optional[dict]:
    """Create a relationship between concepts."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_add_concept_relationship(
            from_concept=from_concept,
            to_concept=to_concept,
            rel_type=rel_type,
            weight=weight,
            description=description,
            context=ctx
        )
    except Exception as e:
        _log.error(f" add_concept_relationship failed: {e}")
        return None


def get_related_concepts(
    concept_name: str,
    rel_type: str = None,
    min_weight: float = 0.0
) -> list:
    """Get concepts related to a given concept."""
    ctx = get_context()
    if not ctx:
        return []

    try:
        result = memory_service.memory_related_concepts(
            concept_name=concept_name,
            rel_type=rel_type,
            min_weight=min_weight,
            context=ctx
        )
        return result.get("related", [])
    except Exception as e:
        _log.error(f" get_related_concepts failed: {e}")
        return []


# === Relationship Operations ===

def add_relationship(
    from_ref: str,
    to_ref: str,
    rel_type: str,
    weight: float = 0.5,
    description: str = None
) -> Optional[dict]:
    """Create a relationship between two memories (any type)."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_add_relationship(
            from_ref=from_ref,
            to_ref=to_ref,
            rel_type=rel_type,
            weight=weight,
            description=description,
            context=ctx
        )
    except Exception as e:
        _log.error(f" add_relationship failed: {e}")
        return None


def get_related(
    ref: str,
    rel_type: str = None,
    limit: int = 10
) -> list:
    """Get memories related to a given reference."""
    ctx = get_context()
    if not ctx:
        return []

    try:
        result = memory_service.memory_related(
            ref=ref,
            rel_type=rel_type,
            limit=limit,
            context=ctx
        )
        return result.get("related", [])
    except Exception as e:
        _log.error(f" get_related failed: {e}")
        return []


def related(
    ref: str,
    rel_type: str = None,
    limit: int = 50,
    direction: str = "both"
) -> Optional[dict]:
    """List relationships for a reference (both directions)."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_list_relationships(
            ref=ref,
            rel_type=rel_type,
            limit=limit,
            direction=direction,
            context=ctx
        )
    except Exception as e:
        _log.error(f" related failed: {e}")
        return None


# === Chain Operations ===

def create_chain(
    chain_type: str,
    name: str = None,
    title: str = None
) -> Optional[dict]:
    """Create a new memory chain."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_chain_create(
            chain_type=chain_type,
            name=name,
            title=title,
            context=ctx
        )
    except Exception as e:
        _log.error(f" create_chain failed: {e}")
        return None


def append_to_chain(
    chain_id: str,
    item_type: str,
    item_id: str = None,
    text: str = None,
    role: str = None
) -> Optional[dict]:
    """Append an item to a chain."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_chain_append(
            chain_id=chain_id,
            item_type=item_type,
            item_id=item_id,
            text=text,
            role=role,
            context=ctx
        )
    except Exception as e:
        _log.error(f" append_to_chain failed: {e}")
        return None


def get_chain(chain_id: str, limit: int = 50) -> Optional[dict]:
    """Get a chain with its entries."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_chain_get(
            chain_id=chain_id,
            limit=limit,
            context=ctx
        )
    except Exception as e:
        _log.error(f" get_chain failed: {e}")
        return None


def list_chains(
    chain_type: str = None,
    name_contains: str = None,
    limit: int = 50
) -> list:
    """List chains, optionally filtered."""
    ctx = get_context()
    if not ctx:
        return []

    try:
        result = memory_service.memory_chain_list(
            chain_type=chain_type,
            name_contains=name_contains,
            limit=limit,
            context=ctx
        )
        return result.get("chains", [])
    except Exception as e:
        _log.error(f" list_chains failed: {e}")
        return []


# === Reference Operations ===

def get_by_ref(ref: str) -> Optional[dict]:
    """Get a memory by its reference (e.g., 'observation:42')."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_get_by_ref(
            ref=ref,
            context=ctx
        )
    except Exception as e:
        _log.error(f" get_by_ref failed: {e}")
        return None


# === Utility ===

def format_ref(mem_type: str, mem_id: int) -> str:
    """Format a memory reference string."""
    return f"{mem_type}:{mem_id}"
