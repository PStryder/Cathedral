from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class WebSearchRequest(BaseModel):
    """Model for web search."""
    query: str
    max_results: int = 10


class FetchRequest(BaseModel):
    """Model for page fetch."""
    url: str
    mode: str = "simple"  # simple or headless
    format: str = "markdown"  # markdown, text, html


def create_router(BrowserGate, emit_event) -> APIRouter:
    router = APIRouter()

    @router.get("/api/browser/status")
    async def api_browser_status():
        """Get BrowserGate and extension server status."""
        try:
            from cathedral.BrowserGate import get_server
            server = get_server()
            return {
                "websocket_running": server.is_running,
                "websocket_port": server.port,
                "connected_clients": server.client_count,
                "providers": BrowserGate.get_browser().list_providers(),
            }
        except Exception as e:
            return {
                "websocket_running": False,
                "error": str(e),
            }

    @router.post("/api/browser/search")
    async def api_browser_search(data: WebSearchRequest):
        """Search the web."""
        try:
            response = await BrowserGate.search(data.query, max_results=data.max_results)
            await emit_event("browser", f"Search: {data.query[:50]}...", results=len(response.results))
            return {
                "query": response.query,
                "provider": response.provider.value,
                "results": [r.model_dump() for r in response.results],
                "search_time_ms": response.search_time_ms,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/browser/fetch")
    async def api_browser_fetch(data: FetchRequest):
        """Fetch page content."""
        try:
            mode = BrowserGate.FetchMode.HEADLESS if data.mode == "headless" else BrowserGate.FetchMode.SIMPLE
            fmt = {
                "text": BrowserGate.ContentFormat.TEXT,
                "html": BrowserGate.ContentFormat.HTML,
            }.get(data.format, BrowserGate.ContentFormat.MARKDOWN)

            page = await BrowserGate.fetch(data.url, mode=mode, output_format=fmt)
            await emit_event("browser", f"Fetched: {page.title[:50]}...", url=data.url)
            return page.model_dump()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router


__all__ = ["create_router"]
