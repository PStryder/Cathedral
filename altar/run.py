import sys
import os
import json

# Add the root project directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

from cathedral import process_input_stream, loom, MemoryGate
from cathedral import Config
from cathedral import PersonalityGate
from cathedral import SecurityManager
from cathedral import FileSystemGate
from cathedral import ShellGate
from cathedral import BrowserGate
from cathedral.MemoryGate.discovery import (
    start_discovery, stop_discovery, queue_message_discovery
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any

app = FastAPI()


@app.on_event("startup")
async def startup_event():
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
        from cathedral import ScriptureGate

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
                metadata={"url": url} if url else {}
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


@app.on_event("shutdown")
async def shutdown_event():
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


# Setup templates and static files
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

origins = [
    "http://localhost:5000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security middleware - check lock status for protected routes
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import RedirectResponse, JSONResponse


class SecurityMiddleware(BaseHTTPMiddleware):
    """Check if session is locked and redirect/block as needed."""

    # Routes that don't require unlock
    PUBLIC_PATHS = {
        "/lock",
        "/security/setup",
        "/api/security/status",
        "/api/security/unlock",
        "/api/security/setup",
        "/api/security/reset",
        "/static",
        "/favicon.ico",
    }

    async def dispatch(self, request, call_next):
        path = request.url.path

        # Allow public paths
        for public_path in self.PUBLIC_PATHS:
            if path.startswith(public_path):
                return await call_next(request)

        # Check if encryption is enabled and session is locked
        if SecurityManager.is_encryption_enabled() and SecurityManager.is_locked():
            # API requests get 401
            if path.startswith("/api/"):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Session is locked", "locked": True}
                )
            # HTML requests get redirected to lock screen
            return RedirectResponse(url=f"/lock?redirect={path}")

        # Extend session on activity
        if not SecurityManager.is_locked():
            SecurityManager.extend_session()

        return await call_next(request)


app.add_middleware(SecurityMiddleware)


# Request models
class UserInput(BaseModel):
    user_input: str
    thread_uid: str

class ThreadRequest(BaseModel):
    thread_uid: str | None = None
    thread_name: str | None = None


# ========== Web UI Routes ==========

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ========== API Routes for Web UI ==========

@app.get("/api/threads")
async def api_list_threads():
    """List all threads for the sidebar."""
    return {"threads": loom.list_all_threads()}


@app.post("/api/thread")
async def api_create_or_switch_thread(request: ThreadRequest):
    """Create a new thread or switch to an existing one."""
    if request.thread_uid:
        loom.switch_to_thread(request.thread_uid)
        return {"status": "switched", "thread_uid": request.thread_uid}
    else:
        thread_uid = loom.create_new_thread(request.thread_name)
        return {"status": "created", "thread_uid": thread_uid}


@app.get("/api/thread/{thread_uid}/history")
async def api_get_thread_history(thread_uid: str):
    """Get chat history for a specific thread."""
    history = await loom.recall_async(thread_uid)
    return {"history": history if history else []}


@app.post("/api/chat/stream")
async def api_chat_stream(user_input: UserInput):
    """Stream chat response using Server-Sent Events."""
    async def generate():
        try:
            async for token in process_input_stream(user_input.user_input, user_input.thread_uid):
                yield {"data": json.dumps({"token": token})}
            yield {"data": json.dumps({"done": True})}
        except Exception as e:
            yield {"data": json.dumps({"error": str(e)})}

    return EventSourceResponse(generate())


# ========== Configuration Routes ==========

@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    """Serve the configuration editor UI."""
    return templates.TemplateResponse("config.html", {"request": request})


@app.get("/api/config/schema")
async def api_config_schema():
    """Get configuration schema grouped by category."""
    return Config.get_schema()


@app.get("/api/config")
async def api_config_get():
    """Get current configuration values (secrets masked)."""
    return Config.get_all(include_secrets=False)


@app.get("/api/config/status")
async def api_config_status():
    """Get configuration status (missing, invalid, configured counts)."""
    return Config.get_status()


class ConfigUpdate(BaseModel):
    """Model for config updates - accepts any key-value pairs."""
    class Config:
        extra = "allow"


@app.post("/api/config")
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


@app.post("/api/config/reload")
async def api_config_reload():
    """Reload configuration from disk."""
    Config.reload()
    return {"status": "reloaded"}


@app.get("/api/config/template", response_class=PlainTextResponse)
async def api_config_template():
    """Download .env.example template."""
    manager = Config.get_manager()
    return manager.create_env_template()


# ========== Personality Routes ==========

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


@app.get("/personalities", response_class=HTMLResponse)
async def personalities_page(request: Request):
    """Serve the personality editor UI."""
    return templates.TemplateResponse("personalities.html", {"request": request})


@app.get("/api/personalities")
async def api_list_personalities(category: str = None, include_builtins: bool = True):
    """List all available personalities."""
    return {"personalities": PersonalityGate.list_all(category=category, include_builtins=include_builtins)}


@app.get("/api/personalities/categories")
async def api_personality_categories():
    """Get list of personality categories."""
    return {"categories": PersonalityGate.PersonalityManager.get_categories()}


@app.get("/api/personality/{personality_id}")
async def api_get_personality(personality_id: str):
    """Get a specific personality by ID."""
    personality = PersonalityGate.load(personality_id)
    if not personality:
        raise HTTPException(status_code=404, detail="Personality not found")
    return personality.to_dict()


@app.post("/api/personality")
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


@app.put("/api/personality/{personality_id}")
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


@app.delete("/api/personality/{personality_id}")
async def api_delete_personality(personality_id: str):
    """Delete a personality."""
    try:
        success = PersonalityGate.PersonalityManager.delete(personality_id)
        if not success:
            raise HTTPException(status_code=404, detail="Personality not found")
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/personality/{personality_id}/duplicate")
async def api_duplicate_personality(personality_id: str, new_name: str):
    """Duplicate a personality with a new name."""
    personality = PersonalityGate.PersonalityManager.duplicate(personality_id, new_name)
    if not personality:
        raise HTTPException(status_code=404, detail="Original personality not found")
    return {"status": "duplicated", "personality": personality.to_dict()}


@app.get("/api/personality/{personality_id}/export")
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


class PersonalityImport(BaseModel):
    """Model for importing a personality."""
    personality_data: Dict[str, Any]
    new_id: str | None = None


@app.post("/api/personality/import")
async def api_import_personality(data: PersonalityImport):
    """Import a personality from JSON data."""
    import tempfile

    # Write to temp file and import
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data.personality_data, f)
        temp_path = f.name

    try:
        personality = PersonalityGate.import_personality(temp_path, data.new_id)
        if not personality:
            raise HTTPException(status_code=400, detail="Failed to import personality")
        return {"status": "imported", "personality": personality.to_dict()}
    finally:
        Path(temp_path).unlink(missing_ok=True)


# ========== Security Routes ==========

class PasswordRequest(BaseModel):
    """Model for password-based requests."""
    password: str


class SecuritySetupRequest(BaseModel):
    """Model for security setup."""
    password: str
    tier: str = "basic"


class ChangePasswordRequest(BaseModel):
    """Model for password change."""
    current_password: str
    new_password: str


class SecuritySettingsUpdate(BaseModel):
    """Model for security settings update."""
    timeout_minutes: int | None = None
    lock_on_idle: bool | None = None
    tier: str | None = None
    components: Dict[str, bool] | None = None


@app.get("/security", response_class=HTMLResponse)
async def security_page(request: Request):
    """Serve the security settings page."""
    return templates.TemplateResponse("security.html", {"request": request})


@app.get("/security/setup", response_class=HTMLResponse)
async def security_setup_page(request: Request):
    """Serve the security setup page."""
    return templates.TemplateResponse("security_setup.html", {"request": request})


@app.get("/lock", response_class=HTMLResponse)
async def lock_screen_page(request: Request):
    """Serve the lock screen."""
    # If not locked or encryption disabled, redirect to main
    if not SecurityManager.is_encryption_enabled() or not SecurityManager.is_locked():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/")
    return templates.TemplateResponse("lock_screen.html", {"request": request})


@app.get("/api/security/status")
async def api_security_status():
    """Get security status."""
    return SecurityManager.get_status()


@app.post("/api/security/setup")
async def api_security_setup(data: SecuritySetupRequest):
    """Enable encryption with a new password."""
    if SecurityManager.is_encryption_enabled():
        raise HTTPException(status_code=400, detail="Encryption already enabled")

    if not SecurityManager.is_available():
        raise HTTPException(status_code=400, detail="Cryptographic libraries not available")

    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if SecurityManager.setup_encryption(data.password, data.tier):
        return {"status": "enabled"}
    else:
        raise HTTPException(status_code=500, detail="Setup failed")


@app.post("/api/security/unlock")
async def api_security_unlock(data: PasswordRequest):
    """Unlock the session with password."""
    if not SecurityManager.is_encryption_enabled():
        return {"status": "not_enabled"}

    if not SecurityManager.is_locked():
        return {"status": "already_unlocked"}

    if SecurityManager.unlock(data.password):
        return {"status": "unlocked"}
    else:
        raise HTTPException(status_code=401, detail="Incorrect password")


@app.post("/api/security/lock")
async def api_security_lock():
    """Lock the session immediately."""
    SecurityManager.lock()
    return {"status": "locked"}


@app.post("/api/security/change-password")
async def api_change_password(data: ChangePasswordRequest):
    """Change the master password."""
    if not SecurityManager.is_encryption_enabled():
        raise HTTPException(status_code=400, detail="Encryption not enabled")

    if SecurityManager.is_locked():
        raise HTTPException(status_code=401, detail="Session is locked")

    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    if SecurityManager.change_password(data.current_password, data.new_password):
        return {"status": "changed"}
    else:
        raise HTTPException(status_code=401, detail="Current password is incorrect")


@app.post("/api/security/settings")
async def api_security_settings(data: SecuritySettingsUpdate):
    """Update security settings."""
    if not SecurityManager.is_encryption_enabled():
        raise HTTPException(status_code=400, detail="Encryption not enabled")

    if SecurityManager.is_locked():
        raise HTTPException(status_code=401, detail="Session is locked")

    SecurityManager.update_settings(
        timeout_minutes=data.timeout_minutes,
        lock_on_idle=data.lock_on_idle,
        tier=data.tier,
        components=data.components
    )

    return {"status": "updated"}


@app.post("/api/security/disable")
async def api_security_disable(data: PasswordRequest):
    """Disable encryption."""
    if not SecurityManager.is_encryption_enabled():
        return {"status": "already_disabled"}

    from cathedral.SecurityManager.config import get_config
    config = get_config()

    if config.disable_encryption(data.password):
        SecurityManager.lock()  # Clear session state
        return {"status": "disabled"}
    else:
        raise HTTPException(status_code=401, detail="Incorrect password")


@app.post("/api/security/reset")
async def api_security_reset():
    """Reset all security and encrypted data (DANGEROUS)."""
    from cathedral.SecurityManager.config import get_config
    config = get_config()
    config.reset()

    # Note: This should also clear encrypted data files
    # For now, just reset the config
    return {"status": "reset"}


# ========== System Events SSE ==========

import asyncio
from collections import deque
from typing import Set
import time

# Event bus for system-wide notifications
class EventBus:
    """Simple pub/sub event bus for system events."""

    def __init__(self, max_history: int = 100):
        self._subscribers: Set[asyncio.Queue] = set()
        self._history: deque = deque(maxlen=max_history)
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to events, returns a queue for receiving."""
        queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from events."""
        async with self._lock:
            self._subscribers.discard(queue)

    async def publish(self, event_type: str, data: dict):
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        self._history.append(event)

        async with self._lock:
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # Skip if queue is full

    def get_recent(self, count: int = 20) -> list:
        """Get recent events from history."""
        return list(self._history)[-count:]


# Global event bus instance
event_bus = EventBus()


async def emit_event(event_type: str, message: str, **kwargs):
    """Helper to emit events to the bus."""
    data = {"message": message, **kwargs}
    await event_bus.publish(event_type, data)


# Make emit_event available to cathedral modules
import cathedral
cathedral.emit_event = emit_event


@app.get("/api/events")
async def api_events():
    """SSE endpoint for real-time system events."""
    async def generate():
        queue = await event_bus.subscribe()
        try:
            # Send initial connection event
            yield {
                "event": "system",
                "data": json.dumps({"message": "Connected to event stream"})
            }

            while True:
                try:
                    # Wait for events with timeout for keepalive
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        "event": event["type"],
                        "data": json.dumps(event["data"])
                    }
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield {"event": "ping", "data": "{}"}

        except asyncio.CancelledError:
            pass
        finally:
            await event_bus.unsubscribe(queue)

    return EventSourceResponse(generate())


# ========== Agent Status Routes ==========

# Agent status tracking (simple in-memory for now)
_agent_updates: deque = deque(maxlen=50)
_agent_update_lock = asyncio.Lock()


async def record_agent_update(agent_id: str, message: str, status: str = "running"):
    """Record an agent status update."""
    async with _agent_update_lock:
        _agent_updates.append({
            "id": agent_id,
            "message": message,
            "status": status,
            "timestamp": time.time()
        })
    # Also emit to event bus
    await emit_event("agent", message, id=agent_id, status=status)


# Make available to cathedral
cathedral.record_agent_update = record_agent_update


@app.get("/api/agents/status")
async def api_agents_status(since: float = 0):
    """Get agent status updates since timestamp (for polling fallback)."""
    async with _agent_update_lock:
        updates = [u for u in _agent_updates if u["timestamp"] > since]
    return {"updates": updates, "timestamp": time.time()}


# ========== FileSystemGate Routes ==========

class FolderCreate(BaseModel):
    """Model for creating a folder configuration."""
    folder_id: str
    path: str
    name: str
    permission: str = "read_only"
    backup_policy: str = "on_modify"
    max_file_size_mb: int = 10
    allowed_extensions: list | None = None
    blocked_extensions: list | None = None
    recursive: bool = True


class FolderUpdate(BaseModel):
    """Model for updating a folder configuration."""
    name: str | None = None
    permission: str | None = None
    backup_policy: str | None = None
    max_file_size_mb: int | None = None
    allowed_extensions: list | None = None
    blocked_extensions: list | None = None
    recursive: bool | None = None


class FileWriteRequest(BaseModel):
    """Model for writing a file."""
    content: str
    encoding: str = "utf-8"
    create_dirs: bool = False


@app.get("/files", response_class=HTMLResponse)
async def files_page(request: Request):
    """Serve the file management UI."""
    return templates.TemplateResponse("files.html", {"request": request})


@app.get("/api/files/folders")
async def api_list_folders():
    """List all configured folders."""
    return {"folders": FileSystemGate.list_folders()}


@app.post("/api/files/folders")
async def api_add_folder(data: FolderCreate):
    """Add a new folder configuration."""
    kwargs = {}
    if data.allowed_extensions is not None:
        kwargs["allowed_extensions"] = data.allowed_extensions
    if data.blocked_extensions is not None:
        kwargs["blocked_extensions"] = data.blocked_extensions

    success, message = FileSystemGate.add_folder(
        folder_id=data.folder_id,
        path=data.path,
        name=data.name,
        permission=data.permission,
        backup_policy=data.backup_policy,
        max_file_size_mb=data.max_file_size_mb,
        recursive=data.recursive,
        **kwargs
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    await emit_event("filesystem", message, operation="add_folder", folder_id=data.folder_id)
    return {"status": "created", "message": message}


@app.get("/api/files/folders/{folder_id}")
async def api_get_folder(folder_id: str):
    """Get a folder configuration."""
    folder = FileSystemGate.get_folder(folder_id)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder.to_dict()


@app.put("/api/files/folders/{folder_id}")
async def api_update_folder(folder_id: str, data: FolderUpdate):
    """Update a folder configuration."""
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    success, message = FileSystemGate.update_folder(folder_id, **updates)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    await emit_event("filesystem", message, operation="update_folder", folder_id=folder_id)
    return {"status": "updated", "message": message}


@app.delete("/api/files/folders/{folder_id}")
async def api_remove_folder(folder_id: str):
    """Remove a folder configuration."""
    success, message = FileSystemGate.remove_folder(folder_id)

    if not success:
        raise HTTPException(status_code=404, detail=message)

    await emit_event("filesystem", message, operation="remove_folder", folder_id=folder_id)
    return {"status": "deleted", "message": message}


@app.get("/api/files/list")
async def api_list_files(folder_id: str, path: str = "", show_hidden: bool = False):
    """List directory contents."""
    result = FileSystemGate.list_dir(folder_id, path, show_hidden)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return {"files": result.data, "path": result.path}


@app.get("/api/files/read")
async def api_read_file(folder_id: str, path: str, encoding: str = "utf-8"):
    """Read a file's contents."""
    result = FileSystemGate.read_file(folder_id, path, encoding)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return {"content": result.data, "path": result.path, "message": result.message}


@app.post("/api/files/write")
async def api_write_file(folder_id: str, path: str, data: FileWriteRequest):
    """Write content to a file."""
    result = FileSystemGate.write_file(
        folder_id, path, data.content, data.encoding, data.create_dirs
    )
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    await emit_event("filesystem", result.message, operation="write", path=result.path)
    return result.to_dict()


@app.post("/api/files/mkdir")
async def api_mkdir(folder_id: str, path: str, parents: bool = False):
    """Create a directory."""
    result = FileSystemGate.mkdir(folder_id, path, parents)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    await emit_event("filesystem", result.message, operation="mkdir", path=result.path)
    return result.to_dict()


@app.delete("/api/files/delete")
async def api_delete_file(folder_id: str, path: str, recursive: bool = False):
    """Delete a file or directory."""
    result = FileSystemGate.delete(folder_id, path, recursive)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    await emit_event("filesystem", result.message, operation="delete", path=result.path)
    return result.to_dict()


@app.get("/api/files/info")
async def api_file_info(folder_id: str, path: str):
    """Get file or directory information."""
    result = FileSystemGate.info(folder_id, path)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return result.data


@app.get("/api/files/backups")
async def api_list_backups(folder_id: str | None = None, limit: int = 50):
    """List backups."""
    return {"backups": FileSystemGate.list_backups(folder_id, limit)}


@app.post("/api/files/backups/{backup_id}/restore")
async def api_restore_backup(backup_id: str, overwrite: bool = True):
    """Restore a file from backup."""
    success, message = FileSystemGate.restore_backup(backup_id, overwrite)
    if not success:
        raise HTTPException(status_code=400, detail=message)

    await emit_event("filesystem", message, operation="restore", backup_id=backup_id)
    return {"status": "restored", "message": message}


@app.delete("/api/files/backups/{backup_id}")
async def api_delete_backup(backup_id: str):
    """Delete a backup."""
    success, message = FileSystemGate.delete_backup(backup_id)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    return {"status": "deleted", "message": message}


@app.get("/api/files/backups/stats")
async def api_backup_stats():
    """Get backup statistics."""
    return FileSystemGate.get_backup_stats()


@app.get("/api/files/config")
async def api_files_config():
    """Get FileSystemGate configuration."""
    return FileSystemGate.get_config()


# ========== ShellGate Routes ==========

class ShellExecuteRequest(BaseModel):
    """Model for executing a command."""
    command: str
    working_dir: str | None = None
    timeout: int | None = None


class ShellConfigUpdate(BaseModel):
    """Model for updating shell configuration."""
    default_timeout_seconds: int | None = None
    max_timeout_seconds: int | None = None
    default_working_dir: str | None = None
    allowed_commands: list | None = None
    blocked_commands: list | None = None
    require_unlock: bool | None = None
    log_commands: bool | None = None
    max_concurrent_background: int | None = None


@app.get("/shell", response_class=HTMLResponse)
async def shell_page(request: Request):
    """Serve the shell execution UI."""
    return templates.TemplateResponse("shell.html", {"request": request})


@app.post("/api/shell/execute")
async def api_shell_execute(data: ShellExecuteRequest):
    """Execute a command synchronously."""
    result = ShellGate.execute(data.command, data.working_dir, data.timeout)

    if result.success:
        await emit_event("shell", f"Command completed: {data.command[:50]}",
                        operation="execute", exit_code=result.exit_code)

    return result.to_dict()


@app.post("/api/shell/stream")
async def api_shell_stream(data: ShellExecuteRequest):
    """Execute a command with streaming output via SSE."""
    async def generate():
        try:
            async for line in ShellGate.execute_stream(
                data.command, data.working_dir, data.timeout
            ):
                yield {"data": json.dumps({"output": line})}
            yield {"data": json.dumps({"done": True})}
        except Exception as e:
            yield {"data": json.dumps({"error": str(e)})}

    return EventSourceResponse(generate())


@app.post("/api/shell/background")
async def api_shell_background(data: ShellExecuteRequest):
    """Execute a command in the background."""
    execution = ShellGate.execute_background(data.command, data.working_dir)

    if execution.status.value == "failed":
        raise HTTPException(status_code=400, detail=execution.stderr)

    await emit_event("shell", f"Background command started: {data.command[:50]}",
                    operation="background", execution_id=execution.id)

    return execution.to_dict()


@app.get("/api/shell/status/{execution_id}")
async def api_shell_status(execution_id: str):
    """Get status of a background command."""
    execution = ShellGate.get_status(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution.to_dict()


@app.post("/api/shell/cancel/{execution_id}")
async def api_shell_cancel(execution_id: str):
    """Cancel a background command."""
    success = ShellGate.cancel(execution_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel (not found or already complete)")

    await emit_event("shell", f"Command cancelled: {execution_id}",
                    operation="cancel", execution_id=execution_id)

    return {"status": "cancelled"}


@app.get("/api/shell/running")
async def api_shell_running():
    """List running background commands."""
    return {"running": ShellGate.list_running()}


@app.get("/api/shell/background")
async def api_shell_list_background():
    """List all background commands."""
    return {"commands": ShellGate.list_background()}


@app.get("/api/shell/history")
async def api_shell_history(limit: int = 50, success_only: bool = False):
    """Get command history."""
    return {"history": ShellGate.get_history(limit, success_only)}


@app.delete("/api/shell/history")
async def api_shell_clear_history():
    """Clear command history."""
    count = ShellGate.clear_history()
    return {"status": "cleared", "count": count}


@app.get("/api/shell/validate")
async def api_shell_validate(command: str):
    """Validate a command without executing."""
    is_valid, error = ShellGate.validate_command(command)
    risk = ShellGate.estimate_risk(command)
    return {
        "valid": is_valid,
        "error": error,
        "risk": risk
    }


@app.get("/api/shell/config")
async def api_shell_config():
    """Get shell configuration."""
    return ShellGate.get_config()


@app.put("/api/shell/config")
async def api_shell_config_update(data: ShellConfigUpdate):
    """Update shell configuration."""
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    ShellGate.update_config(**updates)
    return {"status": "updated"}


@app.post("/api/shell/blocked")
async def api_shell_add_blocked(command: str):
    """Add a command to the blocklist."""
    ShellGate.add_blocked_command(command)
    return {"status": "added"}


@app.delete("/api/shell/blocked")
async def api_shell_remove_blocked(command: str):
    """Remove a command from the blocklist."""
    success = ShellGate.remove_blocked_command(command)
    if not success:
        raise HTTPException(status_code=404, detail="Command not in blocklist")
    return {"status": "removed"}


# ========== BrowserGate Routes ==========

class WebSearchRequest(BaseModel):
    """Model for web search."""
    query: str
    max_results: int = 10


class FetchRequest(BaseModel):
    """Model for page fetch."""
    url: str
    mode: str = "simple"  # simple or headless
    format: str = "markdown"  # markdown, text, html


@app.get("/api/browser/status")
async def api_browser_status():
    """Get BrowserGate and extension server status."""
    try:
        from cathedral.BrowserGate import get_server
        server = get_server()
        return {
            "websocket_running": server.is_running,
            "websocket_port": server.port,
            "connected_clients": server.client_count,
            "providers": BrowserGate.get_browser().list_providers()
        }
    except Exception as e:
        return {
            "websocket_running": False,
            "error": str(e)
        }


@app.post("/api/browser/search")
async def api_browser_search(data: WebSearchRequest):
    """Search the web."""
    try:
        response = await BrowserGate.search(data.query, max_results=data.max_results)
        await emit_event("browser", f"Search: {data.query[:50]}...", results=len(response.results))
        return {
            "query": response.query,
            "provider": response.provider.value,
            "results": [r.model_dump() for r in response.results],
            "search_time_ms": response.search_time_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/browser/fetch")
async def api_browser_fetch(data: FetchRequest):
    """Fetch page content."""
    try:
        mode = BrowserGate.FetchMode.HEADLESS if data.mode == "headless" else BrowserGate.FetchMode.SIMPLE
        fmt = {
            "text": BrowserGate.ContentFormat.TEXT,
            "html": BrowserGate.ContentFormat.HTML
        }.get(data.format, BrowserGate.ContentFormat.MARKDOWN)

        page = await BrowserGate.fetch(data.url, mode=mode, output_format=fmt)
        await emit_event("browser", f"Fetched: {page.title[:50]}...", url=data.url)
        return page.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
