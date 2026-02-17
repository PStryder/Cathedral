from __future__ import annotations

from cathedral import (
    MemoryGate,
    PersonalityGate,
    SecurityManager,
    FileSystemGate,
    ShellGate,
    BrowserGate,
    ScriptureGate,
    SubAgentGate,
    ToolGate,
    MCPClient,
    Config,
    AgencyGate,
    VolitionGate,
    PerceptionGate,
)
from cathedral.shared import db_service
from cathedral.shared.gate import GateLogger
from cathedral.MemoryGate.conversation import db as conversation_db
from cathedral.MemoryGate.discovery import (
    start_discovery,
    stop_discovery,
    queue_message_discovery,
)
from cathedral.runtime import loom

# Lifecycle logger
_log = GateLogger.get("Lifecycle")


async def startup(emit_event):
    """Initialize subsystems on server startup."""
    database_url = Config.get("DATABASE_URL")
    if database_url:
        db_service.init_db(database_url)
        conversation_db.init_conversation_db()
        ScriptureGate.init_scripture_db()
    else:
        _log.warning("DATABASE_URL not set - conversation/scripture DB disabled")

    # Initialize all gates explicitly
    MemoryGate.initialize()
    PersonalityGate.initialize()
    SecurityManager.initialize()
    FileSystemGate.initialize()
    ShellGate.initialize()
    BrowserGate.initialize()
    ScriptureGate.initialize()
    ToolGate.initialize()
    MCPClient.initialize()
    # SubAgentGate initializes lazily via get_manager()

    # Faculta Gates (AgencyGate -> VolitionGate -> PerceptionGate)
    AgencyGate.initialize()
    VolitionGate.initialize()
    PerceptionGate.initialize(publish_fn=emit_event)
    _log.info("Faculta gates initialized (AgencyGate, VolitionGate, PerceptionGate)")

    # Connect MCP servers with auto_connect enabled
    try:
        results = await MCPClient.connect_enabled_servers()
        connected = sum(1 for v in results.values() if v)
        total = len(results)
        if total > 0:
            _log.info(f"MCPClient connected to {connected}/{total} servers")
    except Exception as e:
        _log.error(f"MCPClient auto-connect failed: {e}")

    # Start Knowledge Discovery background worker
    try:
        await start_discovery()
        # Wire discovery to Loom message flow
        loom.enable_discovery(queue_message_discovery)
        _log.info("Knowledge Discovery service started")
    except Exception as e:
        _log.error(f"Knowledge Discovery failed to start: {e}")

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

        _log.info(f"BrowserGate WebSocket server started on ws://localhost:{server.port}")
    except Exception as e:
        _log.error(f"BrowserGate WebSocket server failed to start: {e}")


async def shutdown():
    """Cleanup on server shutdown."""
    # Shutdown Faculta gates
    try:
        await PerceptionGate.shutdown()
        await AgencyGate.close_all()
        _log.info("Faculta gates shut down")
    except Exception as e:
        _log.error(f"Faculta shutdown error: {e}")

    # Disconnect MCP servers
    try:
        await MCPClient.disconnect_all()
        _log.info("MCPClient disconnected all servers")
    except Exception as e:
        _log.error(f"MCPClient shutdown error: {e}")

    # Stop Knowledge Discovery worker
    try:
        loom.disable_discovery()
        await stop_discovery()
        _log.info("Knowledge Discovery service stopped")
    except Exception as e:
        _log.error(f"Knowledge Discovery shutdown error: {e}")

    # Stop BrowserGate WebSocket server
    try:
        from cathedral.BrowserGate import stop_extension_server
        await stop_extension_server()
        _log.info("BrowserGate WebSocket server stopped")
    except Exception as e:
        _log.error(f"BrowserGate shutdown error: {e}")


__all__ = ["startup", "shutdown"]
