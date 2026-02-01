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
_anchor_chain_id: Optional[str] = None  # Single-user anchor chain


def initialize() -> bool:
    """
    Initialize MemoryGate database connection.
    Returns True if successful, False if configuration is missing.

    Also looks for an existing anchor chain (chain_type="anchor").
    """
    global _initialized, _context, _anchor_chain_id

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

        # First, bootstrap without agent_uuid to register Cathedral agent
        # This will mint a new UUID or find existing one
        bootstrap_ctx = RequestContext(
            auth=AuthContext(tenant_id="cathedral"),
            agent_uuid=None  # Let bootstrap mint/find
        )

        bootstrap_result = memory_service.memory_bootstrap(
            ai_name="Cathedral",
            ai_platform="Cathedral",
            context=bootstrap_ctx
        )

        # Extract the agent_uuid from bootstrap result (at root level)
        agent_uuid = None
        if bootstrap_result:
            agent_uuid = bootstrap_result.get("agent_uuid")
            if agent_uuid:
                _log.info(f"Using agent_uuid: {agent_uuid}")
            else:
                _log.warning("Bootstrap did not return agent_uuid, will use context without it")

        _context = RequestContext(
            auth=AuthContext(tenant_id="cathedral"),
            agent_uuid=agent_uuid
        )
        _initialized = True

        # Look for existing anchor chain (single-user: only one)
        try:
            anchor_chains = memory_service.memory_chain_list(
                chain_type="anchor",
                limit=1,
                context=_context
            )
            chains = anchor_chains.get("chains", [])
            if chains:
                _anchor_chain_id = chains[0].get("chain_id")
                _log.info(f"Found anchor chain: {_anchor_chain_id}")
            else:
                _log.debug("No anchor chain found")
        except Exception as e:
            _log.debug(f"Anchor chain lookup failed: {e}")

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
    """
    Store an observation with embedding.

    Returns:
        Dict with observation data including 'ref' (e.g., 'observation:123')
    """
    ctx = get_context()
    if not ctx:
        return None

    try:
        result = memory_service.memory_store(
            observation=text,
            confidence=confidence,
            domain=domain,
            evidence=evidence,
            context=ctx
        )
        # Ensure ref is in response
        if result and "ref" not in result:
            obs_id = result.get("id") or result.get("observation_id")
            if obs_id:
                result["ref"] = f"observation:{obs_id}"
        return result
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
    """
    Create or update a pattern (synthesized understanding).

    Returns:
        Dict with pattern data including 'ref' (e.g., 'pattern:123')
    """
    ctx = get_context()
    if not ctx:
        return None

    try:
        result = memory_service.memory_update_pattern(
            category=category,
            pattern_name=name,
            pattern_text=text,
            confidence=confidence,
            evidence_observation_ids=evidence_ids,
            context=ctx
        )
        # Ensure ref is in response
        if result and "ref" not in result:
            pattern_id = result.get("id") or result.get("pattern_id")
            if pattern_id:
                result["ref"] = f"pattern:{pattern_id}"
        return result
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
    """
    Store a concept in the knowledge graph.

    Returns:
        Dict with concept data including 'ref' (e.g., 'concept:Cathedral')
        Note: concept refs use name, not numeric ID
    """
    ctx = get_context()
    if not ctx:
        return None

    try:
        result = memory_service.memory_store_concept(
            name=name,
            concept_type=concept_type,
            description=description,
            domain=domain,
            status=status,
            context=ctx
        )
        # Ensure ref is in response (concepts use name as ref)
        if result and "ref" not in result:
            concept_name = result.get("name") or name
            if concept_name:
                result["ref"] = f"concept:{concept_name}"
        return result
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
    min_confidence: float = 0.0,
    include_cold: bool = False
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
            include_cold=include_cold,
            context=ctx
        )
        # MemoryGate returns 'results' not 'observations'
        return result.get("results", [])
    except Exception as e:
        _log.error(f" recall failed: {e}")
        return []


def get_stats() -> Optional[dict]:
    """Get memory system statistics."""
    ctx = get_context()
    if not ctx:
        return None

    try:
        return memory_service.memory_stats(context=ctx)
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
    """
    Create a relationship between concepts.

    Returns:
        Dict with relationship data including concept refs
    """
    ctx = get_context()
    if not ctx:
        return None

    try:
        result = memory_service.memory_add_concept_relationship(
            from_concept=from_concept,
            to_concept=to_concept,
            rel_type=rel_type,
            weight=weight,
            description=description,
            context=ctx
        )
        # Ensure refs are in response
        if result:
            if "from_ref" not in result:
                result["from_ref"] = f"concept:{from_concept}"
            if "to_ref" not in result:
                result["to_ref"] = f"concept:{to_concept}"
            if "rel_type" not in result:
                result["rel_type"] = rel_type
        return result
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
    """
    Create a relationship between two memories (any type).

    Returns:
        Dict with relationship data including 'edge_id' and refs
    """
    ctx = get_context()
    if not ctx:
        return None

    try:
        result = memory_service.memory_add_relationship(
            from_ref=from_ref,
            to_ref=to_ref,
            rel_type=rel_type,
            weight=weight,
            description=description,
            context=ctx
        )
        # Ensure refs are in response
        if result:
            if "from_ref" not in result:
                result["from_ref"] = from_ref
            if "to_ref" not in result:
                result["to_ref"] = to_ref
            if "rel_type" not in result:
                result["rel_type"] = rel_type
        return result
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


# === Anchor Chain Operations (Single-User) ===

def get_anchor_chain_id() -> Optional[str]:
    """
    Get the anchor chain ID.

    Returns the ID of the anchor chain if one exists, None otherwise.
    For single-user Cathedral, there's only one anchor chain.
    """
    return _anchor_chain_id


def get_anchor_chain(limit: int = 50) -> Optional[dict]:
    """
    Get the anchor chain with its entries.

    The anchor chain contains agent identity, domain taxonomy,
    relationship vocabulary, and operational protocols.

    Returns:
        Chain dict with entries, or None if no anchor chain exists.
    """
    if not _anchor_chain_id:
        return None

    return get_chain(_anchor_chain_id, limit=limit)


def set_anchor_chain(chain_id: str) -> bool:
    """
    Set the anchor chain ID.

    For single-user Cathedral, this sets THE anchor chain.
    The chain should already exist (created via create_chain).

    Args:
        chain_id: The chain ID to use as anchor

    Returns:
        True if successful, False otherwise.
    """
    global _anchor_chain_id

    ctx = get_context()
    if not ctx:
        return False

    # Verify chain exists
    chain = get_chain(chain_id, limit=1)
    if not chain:
        _log.error(f"Cannot set anchor chain: chain {chain_id} not found")
        return False

    _anchor_chain_id = chain_id
    _log.info(f"Anchor chain set to: {chain_id}")
    return True


def create_anchor_chain(
    name: str = "cathedral_anchor",
    title: str = "Cathedral Agent Anchor"
) -> Optional[str]:
    """
    Create a new anchor chain and set it as the active anchor.

    Args:
        name: Chain name (default: cathedral_anchor)
        title: Chain title

    Returns:
        Chain ID if successful, None otherwise.
    """
    global _anchor_chain_id

    ctx = get_context()
    if not ctx:
        return None

    try:
        result = memory_service.memory_chain_create(
            chain_type="anchor",
            name=name,
            title=title,
            context=ctx
        )
        if result:
            chain_id = result.get("chain_id")
            if chain_id:
                _anchor_chain_id = chain_id
                _log.info(f"Created anchor chain: {chain_id}")
                return chain_id
    except Exception as e:
        _log.error(f"Failed to create anchor chain: {e}")

    return None


def append_to_anchor_chain(
    item_type: str,
    text: str = None,
    role: str = None
) -> Optional[dict]:
    """
    Append an entry to the anchor chain.

    Common roles for anchor chains:
    - constitution: Identity and core principles
    - verb_list: Relationship vocabulary
    - playbook: Operational protocols
    - domain_taxonomy: Domain categorization

    Args:
        item_type: Type of item (e.g., "text", "ref")
        text: Text content for the entry
        role: Entry role (constitution, verb_list, playbook, etc.)

    Returns:
        Result dict if successful, None otherwise.
    """
    if not _anchor_chain_id:
        _log.error("No anchor chain set")
        return None

    return append_to_chain(
        chain_id=_anchor_chain_id,
        item_type=item_type,
        text=text,
        role=role
    )


# === Reference Operations ===

def get_by_ref(ref: str) -> Optional[dict]:
    """
    Get a memory by its reference.

    Args:
        ref: Reference string (e.g., 'observation:42', 'concept:Cathedral')

    Returns:
        Memory dict with full data, or None if not found
    """
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


def get_many_by_refs(refs: List[str]) -> List[dict]:
    """
    Get multiple memories by their references in a single call.

    Args:
        refs: List of reference strings (e.g., ['observation:42', 'concept:Cathedral'])

    Returns:
        List of memory dicts (may be fewer than requested if some not found)
    """
    ctx = get_context()
    if not ctx:
        return []

    if not refs:
        return []

    try:
        result = memory_service.memory_get_many_by_refs(
            refs=refs,
            context=ctx
        )
        # Handle various response formats
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return result.get("items", result.get("memories", []))
        return []
    except Exception as e:
        _log.error(f" get_many_by_refs failed: {e}")
        return []


# === Utility ===

def format_ref(mem_type: str, mem_id: int) -> str:
    """Format a memory reference string."""
    return f"{mem_type}:{mem_id}"


# === Self-Documentation ===

def get_info() -> dict:
    """
    Get comprehensive documentation for MemoryGate.

    Returns complete tool documentation including purpose, call formats,
    expected responses, and domain-specific guidance.
    """
    return {
        "gate": "MemoryGate",
        "version": "1.0",
        "purpose": "Persistent semantic memory system for storing and retrieving observations, patterns, concepts, and relationships. Enables agents to remember facts, build knowledge graphs, and search across accumulated knowledge.",

        "concepts": {
            "observations": "Atomic facts with confidence levels (0-1). The basic unit of memory.",
            "patterns": "Synthesized understanding across multiple observations. Upserts by category+name.",
            "concepts": "Named entities in a knowledge graph. Case-insensitive, alias-aware.",
            "relationships": "Semantic connections between any memory items using ref format.",
            "chains": "Ordered sequences of items for temporal/causal tracking.",
            "anchor_chain": "Special chain (type='anchor') containing agent identity, vocabulary, and protocols. Single-user: one per instance.",
            "refs": "Universal reference format: 'type:id' (e.g., 'observation:42', 'concept:Cathedral')",
        },

        "anchor_chain_roles": {
            "constitution": "Agent identity and core principles",
            "verb_list": "Relationship vocabulary (rel_types to use)",
            "playbook": "Operational protocols and behaviors",
            "domain_taxonomy": "Domain categorization guidance",
        },

        "confidence_guide": {
            "1.0": "Direct observation, absolute certainty",
            "0.9-0.99": "Very high confidence, strong evidence",
            "0.8-0.89": "High confidence, solid evidence",
            "0.7-0.79": "Good confidence, some uncertainty",
            "0.5-0.69": "Moderate confidence, competing interpretations",
            "below_0.5": "Speculative, weak evidence - use sparingly",
        },

        "recommended_domains": [
            "technical", "project", "architecture", "decision",
            "user_preference", "system", "interaction", "error",
        ],

        "tools": {
            "search": {
                "purpose": "Semantic search across all memory types using vector similarity",
                "call_format": {
                    "query": {"type": "string", "required": True, "description": "Search query text"},
                    "limit": {"type": "integer", "required": False, "default": 5, "description": "Max results to return"},
                    "min_confidence": {"type": "number", "required": False, "default": 0.0, "description": "Minimum confidence threshold (0-1)"},
                    "domain": {"type": "string", "required": False, "description": "Filter to specific domain"},
                    "include_cold": {"type": "boolean", "required": False, "default": False, "description": "Include archived/cold tier memories"},
                },
                "response": {
                    "type": "array",
                    "items": "Memory objects with id, content, confidence, domain, similarity score",
                },
                "example": 'MemoryGate.search(query="database migration", limit=5, min_confidence=0.7)',
            },

            "recall": {
                "purpose": "List recent observations, optionally filtered by domain",
                "call_format": {
                    "domain": {"type": "string", "required": False, "description": "Filter to specific domain"},
                    "limit": {"type": "integer", "required": False, "default": 10, "description": "Max results"},
                    "min_confidence": {"type": "number", "required": False, "default": 0.0, "description": "Minimum confidence"},
                },
                "response": {
                    "type": "array",
                    "items": "Observation objects with id, observation text, confidence, domain, timestamp",
                },
                "example": 'MemoryGate.recall(domain="technical", limit=10)',
            },

            "store_observation": {
                "purpose": "Store a new observation/fact in memory with embedding for semantic search",
                "call_format": {
                    "text": {"type": "string", "required": True, "description": "The observation text to store"},
                    "confidence": {"type": "number", "required": False, "default": 0.8, "description": "Confidence level 0-1"},
                    "domain": {"type": "string", "required": False, "description": "Category/domain tag"},
                    "evidence": {"type": "array", "required": False, "description": "List of supporting evidence strings"},
                },
                "response": {
                    "id": "integer - ID of stored observation",
                    "ref": "string - Reference in format 'observation:id'",
                    "status": "string - 'stored' on success",
                },
                "example": 'MemoryGate.store_observation(text="User prefers dark mode", confidence=0.95, domain="user_preference")',
            },

            "get_concept": {
                "purpose": "Look up a concept by name (case-insensitive, alias-aware)",
                "call_format": {
                    "name": {"type": "string", "required": True, "description": "Concept name or alias"},
                },
                "response": {
                    "type": "object",
                    "fields": "id, name, type, description, domain, status, metadata, relationships",
                },
                "example": 'MemoryGate.get_concept(name="Cathedral")',
            },

            "get_related": {
                "purpose": "Get items related to a memory reference via relationships",
                "call_format": {
                    "ref": {"type": "string", "required": True, "description": "Reference (e.g., 'observation:42', 'concept:Name')"},
                    "rel_type": {"type": "string", "required": False, "description": "Filter by relationship type"},
                    "limit": {"type": "integer", "required": False, "default": 50, "description": "Max results"},
                },
                "response": {
                    "type": "array",
                    "items": "Related items with relationship details (type, weight, direction)",
                },
                "example": 'MemoryGate.get_related(ref="observation:42", rel_type="supports")',
            },

            "add_relationship": {
                "purpose": "Create a semantic relationship between two memory items",
                "call_format": {
                    "from_ref": {"type": "string", "required": True, "description": "Source reference"},
                    "to_ref": {"type": "string", "required": True, "description": "Target reference"},
                    "rel_type": {"type": "string", "required": True, "description": "Relationship type (free-form)"},
                    "weight": {"type": "number", "required": False, "default": 0.5, "description": "Relationship strength 0-1"},
                    "description": {"type": "string", "required": False, "description": "Description of relationship"},
                },
                "response": {
                    "status": "string - 'created' or 'updated'",
                    "edge_id": "string - ID of the relationship edge",
                },
                "common_rel_types": ["supports", "contradicts", "causes", "enables", "part_of", "related_to", "supersedes"],
                "example": 'MemoryGate.add_relationship(from_ref="observation:42", to_ref="observation:43", rel_type="supports", weight=0.8)',
            },

            "get_stats": {
                "purpose": "Get memory system statistics and counts",
                "call_format": {},
                "response": {
                    "counts": "Object with observation/pattern/concept/document counts",
                    "domains": "Object mapping domain names to counts",
                    "ai_instances": "Array of registered AI agents",
                    "tiers": "Object with hot/cold tier breakdowns",
                },
                "example": "MemoryGate.get_stats()",
            },
        },

        "best_practices": [
            "Always search before storing to avoid duplicates",
            "Use appropriate confidence levels - don't default everything to 1.0",
            "Tag observations with domains for better organization",
            "Create relationships to build a connected knowledge graph",
            "Use patterns to synthesize understanding across observations",
        ],
    }


__all__ = [
    # Lifecycle
    "initialize",
    "is_healthy",
    "get_health_status",
    "get_dependencies",
    "get_context",
    # Storage
    "store_observation",
    "store_pattern",
    "store_concept",
    # Retrieval
    "search",
    "recall",
    "get_stats",
    "get_pattern",
    "list_patterns",
    "get_concept",
    # Relationships
    "add_concept_alias",
    "add_concept_relationship",
    "get_related_concepts",
    "add_relationship",
    "get_related",
    "related",
    # Chains
    "create_chain",
    "append_to_chain",
    "get_chain",
    "list_chains",
    # Anchor Chain (single-user)
    "get_anchor_chain_id",
    "get_anchor_chain",
    "set_anchor_chain",
    "create_anchor_chain",
    "append_to_anchor_chain",
    # References
    "get_by_ref",
    "get_many_by_refs",
    "format_ref",
    # Documentation
    "get_info",
]
