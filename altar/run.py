import os
import sys
from pathlib import Path

# Add the root project directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from altar.api import (
    browser,
    chat,
    config,
    events,
    files,
    health,
    memory,
    personalities,
    scripture,
    security,
    shell,
    subagent,
    toolgate,
)
from altar.lifecycle import startup, shutdown
from altar.middleware import SecurityMiddleware
from altar.services import AgentTracker, EventBus, build_emitter

from cathedral.pipeline import process_input_stream
from cathedral.runtime import loom
from cathedral.services import ServiceRegistry
from cathedral import (
    BrowserGate,
    Config,
    FileSystemGate,
    MemoryGate,
    PersonalityGate,
    ScriptureGate,
    SecurityManager,
    ShellGate,
    SubAgentGate,
)


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _load_origins() -> list[str]:
    default_origins = ["http://localhost:8000", "http://localhost:5000"]
    origins = Config.get("ALLOWED_ORIGINS", default_origins)

    if isinstance(origins, str):
        return [o.strip() for o in origins.split(",") if o.strip()]
    elif isinstance(origins, list):
        # Ensure all items are strings
        return [str(o) for o in origins if o]
    else:
        # Unexpected type, use defaults
        return default_origins


app = FastAPI()

# Setup templates and static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Eventing + agent status services
event_bus = EventBus()
emit_event = build_emitter(event_bus)
agent_tracker = AgentTracker(event_bus)
services = ServiceRegistry(
    emit_event=emit_event,
    record_agent_update=agent_tracker.record_update,
)


@app.on_event("startup")
async def startup_event():
    await startup(emit_event)


@app.on_event("shutdown")
async def shutdown_event():
    await shutdown()


app.add_middleware(SecurityMiddleware, security_manager=SecurityManager)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.create_router(templates, process_input_stream, loom, services))
app.include_router(config.create_router(templates, Config))
app.include_router(personalities.create_router(templates, PersonalityGate))
app.include_router(security.create_router(templates, SecurityManager))
app.include_router(files.create_router(templates, FileSystemGate, emit_event))
app.include_router(shell.create_router(templates, ShellGate, emit_event))
app.include_router(browser.create_router(BrowserGate, emit_event))
app.include_router(events.create_router(event_bus, agent_tracker))
app.include_router(health.create_router())
app.include_router(scripture.create_router(templates, ScriptureGate, emit_event))
app.include_router(memory.create_router(templates, MemoryGate, emit_event))
app.include_router(subagent.create_router(templates, SubAgentGate, emit_event))
app.include_router(toolgate.create_router())
