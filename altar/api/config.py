from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, PlainTextResponse


def create_router(templates, Config) -> APIRouter:
    router = APIRouter()

    @router.get("/config", response_class=HTMLResponse)
    async def config_page(request: Request):
        """Serve the configuration editor UI."""
        return templates.TemplateResponse("config.html", {"request": request})

    @router.get("/api/config/schema")
    async def api_config_schema():
        """Get configuration schema grouped by category."""
        return Config.get_schema()

    @router.get("/api/config")
    async def api_config_get():
        """Get current configuration values (secrets masked)."""
        return Config.get_all(include_secrets=False)

    @router.get("/api/config/status")
    async def api_config_status():
        """Get configuration status (missing, invalid, configured counts)."""
        return Config.get_status()

    @router.post("/api/config")
    async def api_config_save(updates: Dict[str, Any]):
        """Save configuration updates."""
        manager = Config.get_manager()

        errors = []
        for key, value in updates.items():
            if not manager.set(key, value):
                errors.append(f"Unknown config key: {key}")

        if errors:
            raise HTTPException(status_code=400, detail="; ".join(errors))

        return {"status": "saved", "updated": len(updates)}

    @router.post("/api/config/reload")
    async def api_config_reload():
        """Reload configuration from disk."""
        Config.reload()
        return {"status": "reloaded"}

    @router.get("/api/config/template", response_class=PlainTextResponse)
    async def api_config_template():
        """Download .env.example template."""
        manager = Config.get_manager()
        return manager.create_env_template()

    return router


__all__ = ["create_router"]
