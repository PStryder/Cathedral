"""
Memory API endpoints for knowledge and conversation memory.

Provides REST endpoints for MemoryGate operations including:
- Observations, patterns, and concepts
- Semantic search
- Memory chains
- Concept relationships
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ==================== Request Models ====================


class StoreObservationRequest(BaseModel):
    """Request model for storing an observation."""
    observation: str
    domain: Optional[str] = None
    confidence: float = 0.8
    evidence: Optional[List[str]] = None


class StorePatternRequest(BaseModel):
    """Request model for storing/updating a pattern."""
    category: str
    pattern_name: str
    pattern_text: str
    confidence: float = 0.8
    evidence_observation_ids: Optional[List[int]] = None


class StoreConceptRequest(BaseModel):
    """Request model for storing a concept."""
    name: str
    concept_type: str
    description: str
    domain: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Request model for memory search."""
    query: str
    limit: int = 10
    domain: Optional[str] = None
    min_confidence: float = 0.0
    include_cold: bool = False


class RecallRequest(BaseModel):
    """Request model for memory recall."""
    domain: Optional[str] = None
    min_confidence: float = 0.0
    limit: int = 10
    include_cold: bool = False


class AddConceptAliasRequest(BaseModel):
    """Request model for adding concept alias."""
    alias: str


class AddRelationshipRequest(BaseModel):
    """Request model for adding concept relationship."""
    to_concept: str
    rel_type: str
    weight: float = 0.5
    description: Optional[str] = None


class AddGenericRelationshipRequest(BaseModel):
    """Request model for adding generic relationship."""
    from_ref: str
    to_ref: str
    rel_type: str
    weight: Optional[float] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CreateChainRequest(BaseModel):
    """Request model for creating a chain."""
    chain_type: str
    name: Optional[str] = None
    title: Optional[str] = None
    scope: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AppendChainRequest(BaseModel):
    """Request model for appending to a chain."""
    item_type: str
    item_id: Optional[str] = None
    text: Optional[str] = None
    role: Optional[str] = None


def create_router(templates, MemoryGate, emit_event) -> APIRouter:
    """Create Memory API router."""
    router = APIRouter()

    # ==================== UI Page ====================

    @router.get("/memory", response_class=HTMLResponse)
    async def memory_page(request: Request):
        """Serve the memory management UI."""
        return templates.TemplateResponse("memory.html", {"request": request})

    # ==================== Initialization & Status ====================

    @router.post("/api/memory/initialize")
    async def api_initialize():
        """Initialize the memory system."""
        success = MemoryGate.initialize()
        return {"success": success, "initialized": MemoryGate.is_initialized()}

    @router.get("/api/memory/stats")
    async def api_stats():
        """Get memory statistics."""
        stats = MemoryGate.get_stats()
        if stats is None:
            raise HTTPException(status_code=503, detail="MemoryGate not initialized")
        return stats

    # ==================== Observations ====================

    @router.post("/api/memory/observations")
    async def api_store_observation(data: StoreObservationRequest):
        """Store a new observation."""
        result = MemoryGate.store_observation(
            text=data.observation,
            domain=data.domain,
            confidence=data.confidence,
            evidence=data.evidence,
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to store observation")

        await emit_event(
            "memory",
            f"Stored observation in domain: {data.domain or 'general'}",
            operation="store_observation",
            observation_id=result.get("id"),
        )

        return result

    @router.get("/api/memory/observations")
    async def api_recall_observations(
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        include_cold: bool = False,
    ):
        """Recall observations with optional filters."""
        results = MemoryGate.recall(
            domain=domain,
            min_confidence=min_confidence,
            limit=limit,
            include_cold=include_cold,
        )

        return {"observations": results or [], "count": len(results or [])}

    # ==================== Patterns ====================

    @router.post("/api/memory/patterns")
    async def api_store_pattern(data: StorePatternRequest):
        """Store or update a pattern."""
        result = MemoryGate.store_pattern(
            category=data.category,
            name=data.pattern_name,
            text=data.pattern_text,
            confidence=data.confidence,
            evidence_ids=data.evidence_observation_ids,
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to store pattern")

        await emit_event(
            "memory",
            f"Stored pattern: {data.category}/{data.pattern_name}",
            operation="store_pattern",
            pattern_id=result.get("id"),
        )

        return result

    @router.get("/api/memory/patterns")
    async def api_list_patterns(
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 20,
        include_cold: bool = False,
    ):
        """List patterns with optional filters."""
        results = MemoryGate.list_patterns(
            category=category,
            min_confidence=min_confidence,
            limit=limit,
            include_cold=include_cold,
        )

        return {"patterns": results or [], "count": len(results or [])}

    @router.get("/api/memory/patterns/{category}/{pattern_name}")
    async def api_get_pattern(category: str, pattern_name: str):
        """Get a specific pattern by category and name."""
        result = MemoryGate.get_pattern(category, pattern_name)

        if result is None:
            raise HTTPException(status_code=404, detail="Pattern not found")

        return result

    # ==================== Concepts ====================

    @router.post("/api/memory/concepts")
    async def api_store_concept(data: StoreConceptRequest):
        """Store a new concept."""
        result = MemoryGate.store_concept(
            name=data.name,
            concept_type=data.concept_type,
            description=data.description,
            domain=data.domain,
            status=data.status,
            metadata=data.metadata,
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to store concept")

        await emit_event(
            "memory",
            f"Stored concept: {data.name}",
            operation="store_concept",
            concept_id=result.get("id"),
        )

        return result

    @router.get("/api/memory/concepts/{name}")
    async def api_get_concept(name: str, include_cold: bool = False):
        """Get a concept by name."""
        result = MemoryGate.get_concept(name, include_cold=include_cold)

        if result is None:
            raise HTTPException(status_code=404, detail="Concept not found")

        return result

    @router.post("/api/memory/concepts/{name}/aliases")
    async def api_add_concept_alias(name: str, data: AddConceptAliasRequest):
        """Add an alias to a concept."""
        result = MemoryGate.add_concept_alias(name, data.alias)

        if result is None:
            raise HTTPException(status_code=404, detail="Concept not found")

        return result

    @router.post("/api/memory/concepts/{name}/relationships")
    async def api_add_concept_relationship(name: str, data: AddRelationshipRequest):
        """Add a relationship from this concept to another."""
        result = MemoryGate.add_concept_relationship(
            from_concept=name,
            to_concept=data.to_concept,
            rel_type=data.rel_type,
            weight=data.weight,
            description=data.description,
        )

        if result is None:
            raise HTTPException(status_code=404, detail="One or both concepts not found")

        await emit_event(
            "memory",
            f"Added relationship: {name} -> {data.to_concept}",
            operation="add_relationship",
        )

        return result

    @router.get("/api/memory/concepts/{name}/related")
    async def api_get_related_concepts(
        name: str,
        rel_type: Optional[str] = None,
        min_weight: float = 0.0,
        include_cold: bool = False,
    ):
        """Get concepts related to a given concept."""
        results = MemoryGate.get_related_concepts(
            concept_name=name,
            rel_type=rel_type,
            min_weight=min_weight,
            include_cold=include_cold,
        )

        return {"related": results or [], "count": len(results or [])}

    # ==================== Search ====================

    @router.post("/api/memory/search")
    async def api_search(data: SearchRequest):
        """Semantic search across memory."""
        results = MemoryGate.search(
            query=data.query,
            limit=data.limit,
            domain=data.domain,
            min_confidence=data.min_confidence,
            include_cold=data.include_cold,
        )

        return {"results": results or [], "count": len(results or [])}

    @router.get("/api/memory/search")
    async def api_search_get(
        query: str,
        limit: int = 10,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        include_cold: bool = False,
    ):
        """Semantic search (GET version)."""
        results = MemoryGate.search(
            query=query,
            limit=limit,
            domain=domain,
            min_confidence=min_confidence,
            include_cold=include_cold,
        )

        return {"results": results or [], "count": len(results or [])}

    # ==================== Relationships ====================

    @router.post("/api/memory/relationships")
    async def api_add_relationship(data: AddGenericRelationshipRequest):
        """Add a relationship between any two memory items."""
        result = MemoryGate.add_relationship(
            from_ref=data.from_ref,
            to_ref=data.to_ref,
            rel_type=data.rel_type,
            weight=data.weight,
            description=data.description,
            metadata=data.metadata,
        )

        if result is None:
            raise HTTPException(status_code=400, detail="Failed to add relationship")

        await emit_event(
            "memory",
            f"Added relationship: {data.from_ref} -> {data.to_ref}",
            operation="add_relationship",
        )

        return result

    @router.get("/api/memory/related/{ref:path}")
    async def api_get_related(
        ref: str,
        rel_type: Optional[str] = None,
        min_weight: Optional[float] = None,
        limit: int = 50,
    ):
        """Get items related to a given reference."""
        results = MemoryGate.get_related(
            ref=ref,
            rel_type=rel_type,
            min_weight=min_weight,
            limit=limit,
        )

        return {"related": results or [], "count": len(results or [])}

    # ==================== Chains ====================

    @router.post("/api/memory/chains")
    async def api_create_chain(data: CreateChainRequest):
        """Create a new chain."""
        result = MemoryGate.create_chain(
            chain_type=data.chain_type,
            name=data.name,
            title=data.title,
            scope=data.scope,
            metadata=data.metadata,
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to create chain")

        await emit_event(
            "memory",
            f"Created chain: {data.name or data.chain_type}",
            operation="create_chain",
            chain_id=result.get("chain_id"),
        )

        return result

    @router.get("/api/memory/chains")
    async def api_list_chains(
        chain_type: Optional[str] = None,
        name_contains: Optional[str] = None,
        limit: int = 100,
    ):
        """List chains with optional filters."""
        results = MemoryGate.list_chains(
            chain_type=chain_type,
            name_contains=name_contains,
            limit=limit,
        )

        return {"chains": results or [], "count": len(results or [])}

    @router.get("/api/memory/chains/{chain_id}")
    async def api_get_chain(chain_id: str, limit: int = 50):
        """Get a chain by ID with its entries."""
        result = MemoryGate.get_chain(chain_id, limit=limit)

        if result is None:
            raise HTTPException(status_code=404, detail="Chain not found")

        return result

    @router.post("/api/memory/chains/{chain_id}/append")
    async def api_append_to_chain(chain_id: str, data: AppendChainRequest):
        """Append an entry to a chain."""
        result = MemoryGate.append_to_chain(
            chain_id=chain_id,
            item_type=data.item_type,
            item_id=data.item_id,
            text=data.text,
            role=data.role,
        )

        if result is None:
            raise HTTPException(status_code=400, detail="Failed to append to chain")

        return result

    # ==================== Reference Lookup ====================

    @router.get("/api/memory/ref/{ref:path}")
    async def api_get_by_ref(ref: str, include_cold: bool = False):
        """Get a memory item by its reference."""
        result = MemoryGate.get_by_ref(ref, include_cold=include_cold)

        if result is None:
            raise HTTPException(status_code=404, detail="Reference not found")

        return result

    return router


__all__ = ["create_router"]
