# Cathedral Repository Map

Quick orientation guide to the codebase structure.

---

## Top-Level Structure

```
Cathedral-v2/
├── altar/                  # HTTP API server (FastAPI)
├── cathedral/              # Core library - all Gates and services
├── data/                   # Runtime data storage
├── tests/                  # Test suite
├── legacy/                 # Historical reference docs
├── scripts/                # Database init scripts
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Full stack with PostgreSQL+pgvector
├── requirements.txt        # Python dependencies
├── start-cathedral.bat     # Windows Docker startup
├── start-cathedral.sh      # Unix Docker startup
└── .env.example            # Configuration template
```

---

## cathedral/ - Core Library

### Entry Points

| File | Purpose |
|------|---------|
| `__init__.py` | Central export hub for all Gates |
| `runtime.py` | Lazy service proxies (`loom`, `memory`) |
| `conversation.py` | Loom-compatible facade over MemoryGate |

### Gates (Modular Subsystems)

Each Gate follows a consistent pattern: `initialize()`, `is_healthy()`, `get_dependencies()`.

| Gate | Directory | Purpose |
|------|-----------|---------|
| **MemoryGate** | `cathedral/MemoryGate/` | Semantic knowledge storage (observations, patterns, concepts, chains) |
| **ContextGate** | `cathedral/MemoryGate/context_gate.py` | Heuristic gating for context injection decisions |
| **ScriptureGate** | `cathedral/ScriptureGate/` | Document library with embeddings and RAG |
| **ToolGate** | `cathedral/ToolGate/` | Provider-agnostic tool calling protocol (31 tools) |
| **FileSystemGate** | `cathedral/FileSystemGate/` | Secure file access with folder permissions |
| **ShellGate** | `cathedral/ShellGate/` | Command execution with blocklists |
| **BrowserGate** | `cathedral/BrowserGate/` | Web search and page fetching |
| **PersonalityGate** | `cathedral/PersonalityGate/` | Agent personality configuration |
| **SubAgentGate** | `cathedral/SubAgentGate/` | Background worker task management |
| **SecurityManager** | `cathedral/SecurityManager/` | AES-256-GCM encryption for sensitive data |
| **MCPClient** | `cathedral/MCPClient/` | External MCP server connections |
| **StarMirror** | `cathedral/StarMirror/` | Multi-provider LLM communication |

### Key Subdirectories

```
cathedral/
├── MemoryGate/
│   ├── __init__.py           # 40+ memory operations (store, search, recall)
│   ├── auto_memory.py        # Extract memories from conversations
│   ├── context_gate.py       # Smart context injection filtering
│   ├── discovery.py          # Background knowledge discovery
│   └── conversation/
│       ├── db.py             # Database init, FTS support
│       ├── models.py         # Thread, Message, Embedding, Summary models
│       └── embeddings.py     # OpenAI embedding integration
│
├── ToolGate/
│   ├── registry.py           # ToolRegistry - maps 31 tools across Gates
│   ├── orchestrator.py       # Tool execution loop with streaming
│   ├── policy.py             # 5 security levels (READ_ONLY → PRIVILEGED)
│   ├── protocol.py           # JSON-in-text parsing
│   └── prompt.py             # Tool-calling system prompt
│
├── StarMirror/
│   ├── router.py             # Provider routing (OpenRouter, Claude CLI, Codex)
│   ├── providers/            # LLM provider implementations
│   └── MediaGate/            # Multi-modal support (vision, audio)
│
├── pipeline/
│   └── chat.py               # Core chat processing, slash commands, tool loop
│
├── shared/
│   ├── gate.py               # GateLogger, GateErrorHandler, base patterns
│   ├── db.py                 # Database session wrappers
│   └── embeddings.py         # Embedding service
│
├── Config/
│   ├── __init__.py           # ConfigManager with schema validation
│   └── schema.py             # Configuration schema definition
│
└── Memory/
    └── __init__.py           # UnifiedMemory - combines conversation + knowledge
```

---

## altar/ - HTTP Server

### Entry Points

| File | Purpose |
|------|---------|
| `run.py` | FastAPI app init, route registration, lifecycle hooks |
| `lifecycle.py` | Startup (init all Gates) and shutdown handlers |

### API Routes (`altar/api/`)

| Router | Endpoints | Purpose |
|--------|-----------|---------|
| `chat.py` | `/`, `/api/chat/stream`, `/api/threads` | Chat UI and streaming |
| `memory.py` | `/api/memory/*` | MemoryGate operations |
| `scripture.py` | `/api/scripture/*` | Document storage and RAG |
| `toolgate.py` | `/api/tools/*` | Tool registry and policy |
| `config.py` | `/config`, `/api/config` | Configuration management |
| `health.py` | `/api/health` | Gate health status |
| `personalities.py` | `/api/personalities/*` | Personality CRUD |
| `security.py` | `/api/security/*` | Encryption unlock/lock |
| `files.py` | `/api/files/*` | FileSystemGate operations |
| `shell.py` | `/api/shell/*` | Command execution |
| `browser.py` | `/api/browser/*` | Web search and fetch |
| `subagent.py` | `/api/agents/*` | Background workers |
| `mcp.py` | `/api/mcp/*` | MCP server management |
| `events.py` | `/api/events` | Server-Sent Events stream |

### Supporting

```
altar/
├── services/
│   ├── events.py             # EventBus - pub/sub system
│   └── agents.py             # AgentTracker - active execution tracking
├── middleware/
│   └── security.py           # CORS and security headers
├── templates/                # Jinja2 HTML templates
└── static/                   # CSS, JS, assets
```

---

## data/ - Runtime Storage

```
data/
├── config/                   # JSON configuration persistence
├── personalities/            # Agent personality definitions (JSON)
├── scripture/                # Stored documents
├── agents/                   # SubAgent task results
├── backups/                  # FileSystemGate backup records
├── shell_history/            # Command execution history
└── keys/                     # Encryption key storage
```

---

## tests/ - Test Suite

```
tests/
├── conftest.py               # Fixtures (database, mocking, cleanup)
├── test_context_gate.py      # Context injection heuristics
├── test_filesystemgate.py    # File operations and security
├── test_mcpclient.py         # MCP server integration
├── test_personalitygate.py   # Personality management
├── test_scripturegate.py     # Document storage and search
├── test_security_manager.py  # Encryption tests
├── test_shared_gate.py       # Shared utilities
├── test_shellgate.py         # Command execution security
├── test_subagent_worker.py   # Background workers
├── test_toolgate.py          # Tool protocol and orchestration
└── test_unified_memory.py    # Memory interface
```

Run with: `pytest tests/ -v`

---

## Core Execution Flow

```
User Message
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  altar/api/chat.py  POST /api/chat/stream                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  cathedral/pipeline/chat.py  process_input_stream()         │
│    1. Check for slash commands (/history, /forget, etc.)    │
│    2. Load personality system prompt                        │
│    3. Fetch conversation history                            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  ContextGate.decide()  (cathedral/MemoryGate/context_gate)  │
│    Heuristic scoring: should we inject memory context?      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ (if score >= threshold)
┌─────────────────────────────────────────────────────────────┐
│  MemoryGate.hybrid_search()  +  ScriptureGate.build_rag()   │
│    Retrieve relevant memories and documents                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  StarMirror.reflect_stream()                                │
│    Route to LLM provider (OpenRouter / Claude CLI / Codex)  │
│    Stream response tokens via SSE                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ (if response contains tool calls)
┌─────────────────────────────────────────────────────────────┐
│  ToolGate.orchestrator.execute_tools()                      │
│    Parse JSON tool calls → dispatch to Gate → return result │
│    Loop until no more tool calls or budget exhausted        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  MemoryGate.auto_memory.extract_from_exchange()             │
│    Store observations from the conversation                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Response streamed to client
```

---

## Where to Look For X

| Topic | Location |
|-------|----------|
| **Memory storage** | `cathedral/MemoryGate/__init__.py` |
| **Context injection** | `cathedral/MemoryGate/context_gate.py`, `auto_memory.py` |
| **Hybrid search (FTS + semantic)** | `cathedral/MemoryGate/conversation/__init__.py` → `hybrid_search()` |
| **Conversation history** | `cathedral/MemoryGate/conversation/models.py` |
| **Document/RAG storage** | `cathedral/ScriptureGate/` |
| **Tool definitions** | `cathedral/ToolGate/registry.py` |
| **Tool security policies** | `cathedral/ToolGate/policy.py`, `models.py` |
| **Tool execution loop** | `cathedral/ToolGate/orchestrator.py` |
| **LLM provider routing** | `cathedral/StarMirror/router.py` |
| **Chat pipeline** | `cathedral/pipeline/chat.py` |
| **Slash commands** | `cathedral/pipeline/chat.py`, `cathedral/commands/` |
| **File access security** | `cathedral/FileSystemGate/security.py` |
| **Shell command security** | `cathedral/ShellGate/security.py` |
| **Encryption** | `cathedral/SecurityManager/crypto.py` |
| **MCP server connections** | `cathedral/MCPClient/` |
| **Agent personalities** | `cathedral/PersonalityGate/`, `data/personalities/` |
| **Background workers** | `cathedral/SubAgentGate/` |
| **Configuration schema** | `cathedral/Config/schema.py` |
| **Database models** | `cathedral/MemoryGate/conversation/models.py`, `cathedral/ScriptureGate/models.py` |
| **API endpoints** | `altar/api/*.py` |
| **Server startup** | `altar/lifecycle.py` |
| **Health checks** | `altar/api/health.py` |
| **Tests** | `tests/test_*.py` |
| **Docker setup** | `Dockerfile`, `docker-compose.yml` |
| **Environment config** | `.env.example`, `.env.docker` |

---

## Key Files Quick Reference

| What | File |
|------|------|
| Main app entry | `altar/run.py` |
| Gate initialization | `altar/lifecycle.py` |
| Chat processing | `cathedral/pipeline/chat.py` |
| Tool registry | `cathedral/ToolGate/registry.py` |
| Memory operations | `cathedral/MemoryGate/__init__.py` |
| Context gating | `cathedral/MemoryGate/context_gate.py` |
| LLM routing | `cathedral/StarMirror/router.py` |
| Configuration | `cathedral/Config/__init__.py` |
| Shared patterns | `cathedral/shared/gate.py` |
