from __future__ import annotations

from cathedral import (
    MemoryGate,
    PersonalityGate,
    SecurityManager,
    FileSystemGate,
    ShellGate,
    BrowserGate,
    ScriptureGate,
)
from cathedral.MemoryGate.discovery import (
    start_discovery,
    stop_discovery,
    queue_message_discovery,
)
from cathedral.runtime import loom


async def startup(emit_event):
    """Initialize subsystems on server startup."""
    MemoryGate.initialize()
    PersonalityGate.initialize()
    SecurityManager.initialize()
    FileSystemGate.initialize()
    ShellGate.initialize()

    # Start Knowledge Discovery background worker
    try:
        await start_discovery()
        # Wire discovery to Loom message flow
        loom.enable_discovery(queue_message_discovery)
        print("[Cathedral] Knowledge Discovery service started")
    except Exception as e:
        print(f"[Cathedral] Knowledge Discovery failed to start: {e}")

    # Start BrowserGate WebSocket server for browser extension
    try:
        from cathedral.BrowserGate import start_extension_server, get_server

        server = await start_extension_server()

        # Wire up extension handlers to Cathedral systems
        async def handle_chat_message(content: str, url: str = None):
            """Queue content from extension for next chat."""
            # Store as pending context (could be enhanced to inject into next message)
            await emit_event("browser", f"Received from extension: {content[:50]}...", url=url)
            return {"queued": True}

        async def handle_store_scripture(content: str, title: str = None, url: str = None):
            """Store content as scripture."""
            result = await ScriptureGate.store_text(
                content=content,
                title=title or "From Browser Extension",
                source_type="browser_extension",
                metadata={"url": url} if url else {},
            )
            await emit_event("browser", f"Stored as scripture: {result.get('ref', '?')}")
            return result

        async def handle_search_memory(query: str):
            """Search memory for related content."""
            results = MemoryGate.search(query, limit=5)
            return results or []

        server.handler.on_chat_message = handle_chat_message
        server.handler.on_store_scripture = handle_store_scripture
        server.handler.on_search_memory = handle_search_memory

        print(f"[Cathedral] BrowserGate WebSocket server started on ws://localhost:{server.port}")
    except Exception as e:
        print(f"[Cathedral] BrowserGate WebSocket server failed to start: {e}")


async def shutdown():
    """Cleanup on server shutdown."""
    # Stop Knowledge Discovery worker
    try:
        loom.disable_discovery()
        await stop_discovery()
        print("[Cathedral] Knowledge Discovery service stopped")
    except Exception:
        pass

    # Stop BrowserGate WebSocket server
    try:
        from cathedral.BrowserGate import stop_extension_server
        await stop_extension_server()
        print("[Cathedral] BrowserGate WebSocket server stopped")
    except Exception:
        pass


__all__ = ["startup", "shutdown"]
