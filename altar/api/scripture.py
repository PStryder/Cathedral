"""
Scripture API endpoints for document storage and retrieval.

Provides REST endpoints for ScriptureGate operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import base64

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel


class StoreTextRequest(BaseModel):
    """Request model for storing text content."""
    content: str
    title: str
    filename: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    source_type: str = "generated"
    source_ref: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StoreArtifactRequest(BaseModel):
    """Request model for storing artifacts."""
    content: Any  # Can be string, dict, or base64-encoded bytes
    title: str
    filename: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    source_ref: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Request model for scripture search."""
    query: str
    limit: int = 10
    file_type: Optional[str] = None
    tags: Optional[List[str]] = None
    min_similarity: float = 0.3


class BuildContextRequest(BaseModel):
    """Request model for building RAG context."""
    query: str
    limit: int = 3
    file_types: Optional[List[str]] = None
    min_similarity: float = 0.4


def create_router(templates, ScriptureGate, emit_event) -> APIRouter:
    """Create Scripture API router."""
    router = APIRouter()

    # ==================== UI Page ====================

    @router.get("/scripture", response_class=HTMLResponse)
    async def scripture_page(request: Request):
        """Serve the scripture library UI."""
        return templates.TemplateResponse("scripture.html", {"request": request})

    # ==================== Store Operations ====================

    @router.post("/api/scripture/store")
    async def api_store_file(
        file: UploadFile = File(...),
        title: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        tags: Optional[str] = Form(None),  # Comma-separated
        file_type: Optional[str] = Form(None),
        source_type: str = Form("upload"),
        source_ref: Optional[str] = Form(None),
    ):
        """
        Store a file in ScriptureGate.

        Accepts multipart/form-data file upload.
        """
        content = await file.read()

        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        result = await ScriptureGate.store(
            source=content,
            title=title or file.filename,
            description=description,
            tags=tag_list,
            file_type=file_type,
            original_name=file.filename,
            source_type=source_type,
            source_ref=source_ref,
        )

        await emit_event(
            "scripture",
            f"Stored scripture: {result.get('title', 'Untitled')}",
            operation="store",
            scripture_uid=result.get("scripture_uid"),
        )

        return result

    @router.post("/api/scripture/store/text")
    async def api_store_text(data: StoreTextRequest):
        """Store text content as a scripture."""
        result = await ScriptureGate.store_text(
            content=data.content,
            title=data.title,
            filename=data.filename,
            description=data.description,
            tags=data.tags,
            source_type=data.source_type,
            source_ref=data.source_ref,
            metadata=data.metadata,
        )

        await emit_event(
            "scripture",
            f"Stored text scripture: {data.title}",
            operation="store_text",
            scripture_uid=result.get("scripture_uid"),
        )

        return result

    @router.post("/api/scripture/store/artifact")
    async def api_store_artifact(data: StoreArtifactRequest):
        """Store an artifact (JSON, code, etc.) as a scripture."""
        result = await ScriptureGate.store_artifact(
            content=data.content,
            title=data.title,
            filename=data.filename,
            description=data.description,
            tags=data.tags,
            source_ref=data.source_ref,
            metadata=data.metadata,
        )

        await emit_event(
            "scripture",
            f"Stored artifact: {data.title}",
            operation="store_artifact",
            scripture_uid=result.get("scripture_uid"),
        )

        return result

    # ==================== Search & Retrieval ====================

    @router.post("/api/scripture/search")
    async def api_search(data: SearchRequest):
        """Semantic search across scriptures."""
        results = await ScriptureGate.search(
            query=data.query,
            limit=data.limit,
            file_type=data.file_type,
            tags=data.tags,
            min_similarity=data.min_similarity,
        )

        return {"results": results, "count": len(results)}

    @router.get("/api/scripture/search")
    async def api_search_get(
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_similarity: float = 0.3,
    ):
        """Semantic search (GET version)."""
        results = await ScriptureGate.search(
            query=query,
            limit=limit,
            file_type=file_type,
            min_similarity=min_similarity,
        )

        return {"results": results, "count": len(results)}

    @router.get("/api/scripture/list")
    async def api_list_scriptures(
        file_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        """List scriptures with optional filters."""
        results = ScriptureGate.list_scriptures(
            file_type=file_type,
            source=source,
            limit=limit,
            offset=offset,
        )

        return {"scriptures": results, "count": len(results)}

    @router.get("/api/scripture/{scripture_uid}")
    async def api_get_scripture(scripture_uid: str):
        """Get scripture by UID."""
        result = ScriptureGate.get(scripture_uid)

        if not result:
            raise HTTPException(status_code=404, detail="Scripture not found")

        return result

    @router.get("/api/scripture/ref/{ref:path}")
    async def api_get_by_ref(ref: str):
        """Get scripture by reference (scripture:type/uid_prefix)."""
        # Reconstruct ref with prefix if needed
        if not ref.startswith("scripture:"):
            ref = f"scripture:{ref}"

        result = ScriptureGate.search_by_ref(ref)

        if not result:
            raise HTTPException(status_code=404, detail="Scripture not found")

        return result

    # ==================== Read Content ====================

    @router.get("/api/scripture/{scripture_uid}/content")
    async def api_read_content(scripture_uid: str, as_text: bool = True):
        """Read scripture file content."""
        content = ScriptureGate.read(scripture_uid, as_text=as_text)

        if content is None:
            raise HTTPException(status_code=404, detail="Scripture not found")

        if as_text:
            return {"content": content}
        else:
            # Return base64-encoded bytes for binary content
            return {"content": base64.b64encode(content).decode("ascii"), "encoding": "base64"}

    @router.get("/api/scripture/{scripture_uid}/download")
    async def api_download(scripture_uid: str):
        """Download scripture file."""
        scripture = ScriptureGate.get(scripture_uid)
        if not scripture:
            raise HTTPException(status_code=404, detail="Scripture not found")

        content = ScriptureGate.read(scripture_uid, as_text=False)
        if content is None:
            raise HTTPException(status_code=404, detail="Scripture content not found")

        # Get filename from scripture
        filename = scripture.get("file_path", "").split("/")[-1]
        if not filename:
            filename = f"{scripture_uid}.bin"

        return Response(
            content=content,
            media_type=scripture.get("mime_type", "application/octet-stream"),
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # ==================== Delete Operations ====================

    @router.delete("/api/scripture/{scripture_uid}")
    async def api_delete_scripture(scripture_uid: str, hard_delete: bool = False):
        """Delete a scripture."""
        # Get title before deletion for event
        scripture = ScriptureGate.get(scripture_uid)
        title = scripture.get("title", "Unknown") if scripture else "Unknown"

        success = ScriptureGate.remove(scripture_uid, hard_delete=hard_delete)

        if not success:
            raise HTTPException(status_code=404, detail="Scripture not found")

        await emit_event(
            "scripture",
            f"Deleted scripture: {title}",
            operation="delete",
            scripture_uid=scripture_uid,
            hard_delete=hard_delete,
        )

        return {"status": "deleted", "scripture_uid": scripture_uid}

    # ==================== RAG Context ====================

    @router.post("/api/scripture/context")
    async def api_build_context(data: BuildContextRequest):
        """Build RAG context from relevant scriptures."""
        context = await ScriptureGate.build_context(
            query=data.query,
            limit=data.limit,
            file_types=data.file_types,
            min_similarity=data.min_similarity,
        )

        return {"context": context}

    # ==================== Indexing ====================

    @router.post("/api/scripture/{scripture_uid}/reindex")
    async def api_reindex(scripture_uid: str):
        """Re-index a scripture (regenerate text and embedding)."""
        success = await ScriptureGate.reindex(scripture_uid)

        if not success:
            raise HTTPException(status_code=404, detail="Scripture not found or indexing failed")

        return {"status": "reindexed", "scripture_uid": scripture_uid}

    @router.post("/api/scripture/backfill")
    async def api_backfill_index(batch_size: int = 20):
        """Index scriptures that haven't been indexed yet."""
        count = await ScriptureGate.backfill_index(batch_size=batch_size)

        await emit_event(
            "scripture",
            f"Backfill indexed {count} scriptures",
            operation="backfill_index",
            count=count,
        )

        return {"status": "completed", "indexed_count": count}

    # ==================== Stats ====================

    @router.get("/api/scripture/stats")
    async def api_stats():
        """Get scripture statistics."""
        return ScriptureGate.stats()

    return router


__all__ = ["create_router"]
