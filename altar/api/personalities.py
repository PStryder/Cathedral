from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path


class PersonalityCreate(BaseModel):
    """Model for creating a personality."""
    name: str
    description: str = ""
    system_prompt: str = "You are a helpful assistant."
    model: str = "openai/gpt-4o-2024-11-20"
    temperature: float = 0.7
    max_tokens: int = 4000
    style_tags: list = []
    memory_domains: list = []
    category: str = "custom"


class PersonalityUpdate(BaseModel):
    """Model for updating a personality."""
    name: str | None = None
    description: str | None = None
    llm_config: Dict[str, Any] | None = None
    behavior: Dict[str, Any] | None = None
    memory: Dict[str, Any] | None = None
    tools: Dict[str, Any] | None = None
    examples: list | None = None


class PersonalityImport(BaseModel):
    """Model for importing a personality."""
    personality_data: Dict[str, Any]
    new_id: str | None = None


def create_router(templates, PersonalityGate) -> APIRouter:
    router = APIRouter()

    @router.get("/personalities", response_class=HTMLResponse)
    async def personalities_page(request: Request):
        """Serve the personality editor UI."""
        return templates.TemplateResponse("personalities.html", {"request": request})

    @router.get("/api/personalities")
    async def api_list_personalities(category: str = None, include_builtins: bool = True):
        """List all available personalities."""
        return {"personalities": PersonalityGate.list_all(category=category, include_builtins=include_builtins)}

    @router.get("/api/personalities/categories")
    async def api_personality_categories():
        """Get list of personality categories."""
        return {"categories": PersonalityGate.PersonalityManager.get_categories()}

    @router.get("/api/personality/{personality_id}")
    async def api_get_personality(personality_id: str):
        """Get a specific personality by ID."""
        personality = PersonalityGate.load(personality_id)
        if not personality:
            raise HTTPException(status_code=404, detail="Personality not found")
        return personality.to_dict()

    @router.post("/api/personality")
    async def api_create_personality(data: PersonalityCreate):
        """Create a new personality."""
        try:
            personality = PersonalityGate.create(
                name=data.name,
                description=data.description,
                system_prompt=data.system_prompt,
                model=data.model,
                temperature=data.temperature,
                max_tokens=data.max_tokens,
                style_tags=data.style_tags,
                memory_domains=data.memory_domains,
                category=data.category
            )
            return {"status": "created", "personality": personality.to_dict()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.put("/api/personality/{personality_id}")
    async def api_update_personality(personality_id: str, updates: PersonalityUpdate):
        """Update an existing personality."""
        try:
            # Build update dict, excluding None values
            update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}
            personality = PersonalityGate.PersonalityManager.update(personality_id, update_dict)
            if not personality:
                raise HTTPException(status_code=404, detail="Personality not found")
            return {"status": "updated", "personality": personality.to_dict()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/api/personality/{personality_id}")
    async def api_delete_personality(personality_id: str):
        """Delete a personality."""
        try:
            success = PersonalityGate.PersonalityManager.delete(personality_id)
            if not success:
                raise HTTPException(status_code=404, detail="Personality not found")
            return {"status": "deleted"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/api/personality/{personality_id}/duplicate")
    async def api_duplicate_personality(personality_id: str, new_name: str):
        """Duplicate a personality with a new name."""
        personality = PersonalityGate.PersonalityManager.duplicate(personality_id, new_name)
        if not personality:
            raise HTTPException(status_code=404, detail="Original personality not found")
        return {"status": "duplicated", "personality": personality.to_dict()}

    @router.get("/api/personality/{personality_id}/export")
    async def api_export_personality(personality_id: str):
        """Export a personality as JSON for download."""
        personality = PersonalityGate.load(personality_id)
        if not personality:
            raise HTTPException(status_code=404, detail="Personality not found")

        # Return as downloadable JSON
        export_data = personality.to_dict()
        export_data["metadata"]["usage_count"] = 0
        export_data["metadata"]["is_builtin"] = False

        return export_data

    @router.post("/api/personality/import")
    async def api_import_personality(data: PersonalityImport):
        """Import a personality from JSON data."""
        import tempfile

        # Write to temp file and import
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data.personality_data, f)
            temp_path = f.name

        try:
            personality = PersonalityGate.import_personality(temp_path, data.new_id)
            if not personality:
                raise HTTPException(status_code=400, detail="Failed to import personality")
            return {"status": "imported", "personality": personality.to_dict()}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    return router


__all__ = ["create_router"]
