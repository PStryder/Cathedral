# Cathedral Feature Set

> Auto-generated documentation of Cathedral's complete feature set, including API endpoints, slash commands, and Gate modules.

## Overview

Cathedral is a modular AI assistant platform with persistent memory, document retrieval, system integration, and multi-modal capabilities.

---

## 1. REST API Endpoints

### Chat & Threads
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Main chat interface (HTML) |
| `GET` | `/api/threads` | List all conversation threads |
| `POST` | `/api/thread` | Create or switch to a thread |
| `GET` | `/api/thread/{uid}/history` | Get thread message history |
| `POST` | `/api/chat/stream` | Stream chat response (SSE) |

### Configuration
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/config` | Config editor UI |
| `GET` | `/api/config/schema` | Get config schema by category |
| `GET` | `/api/config` | Get current config (secrets masked) |
| `GET` | `/api/config/status` | Missing/invalid config status |
| `POST` | `/api/config` | Save config updates |
| `POST` | `/api/config/reload` | Reload config from disk |
| `GET` | `/api/config/template` | Download `.env.example` |

### Health & Monitoring
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/health` | Full health status of all Gates |
| `GET` | `/api/health/gate/{name}` | Specific Gate health |
| `GET` | `/api/health/summary` | Quick health overview |
| `GET` | `/api/events` | Server-Sent Events stream |
| `GET` | `/api/agents/status` | Active agent tracker status |

### Memory (MemoryGate)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/memory` | Memory browser UI |
| `POST` | `/api/memory/initialize` | Initialize MemoryGate |
| `GET` | `/api/memory/stats` | Memory statistics |
| `POST` | `/api/memory/observations` | Store an observation |
| `GET` | `/api/memory/observations` | Recall observations by domain |
| `POST` | `/api/memory/patterns` | Create/update a pattern |
| `GET` | `/api/memory/patterns` | List patterns |
| `GET` | `/api/memory/patterns/{cat}/{name}` | Get specific pattern |
| `POST` | `/api/memory/concepts` | Store a concept |
| `GET` | `/api/memory/concepts/{name}` | Get concept by name |
| `POST` | `/api/memory/concepts/{name}/aliases` | Add concept alias |
| `POST` | `/api/memory/concepts/{name}/relationships` | Add concept relationship |
| `GET` | `/api/memory/concepts/{name}/related` | Get related concepts |
| `POST` | `/api/memory/search` | Semantic search (POST) |
| `GET` | `/api/memory/search` | Semantic search (GET) |
| `POST` | `/api/memory/relationships` | Add generic relationship |
| `GET` | `/api/memory/related/{ref}` | Get related items by ref |
| `POST` | `/api/memory/chains` | Create a chain |
| `GET` | `/api/memory/chains` | List chains |
| `GET` | `/api/memory/chains/{id}` | Get chain contents |
| `POST` | `/api/memory/chains/{id}/append` | Append to chain |
| `GET` | `/api/memory/ref/{ref}` | Get item by reference |

### Scripture (Document Library)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/scripture` | Scripture browser UI |
| `POST` | `/api/scripture/store` | Store file/content |
| `POST` | `/api/scripture/store/text` | Store text content |
| `POST` | `/api/scripture/store/artifact` | Store JSON artifact |
| `POST` | `/api/scripture/search` | Semantic search (POST) |
| `GET` | `/api/scripture/search` | Semantic search (GET) |
| `GET` | `/api/scripture/list` | List scriptures with filters |
| `GET` | `/api/scripture/{uid}` | Get scripture metadata |
| `GET` | `/api/scripture/ref/{ref}` | Get by reference |
| `GET` | `/api/scripture/{uid}/content` | Get file content |
| `GET` | `/api/scripture/{uid}/download` | Download file |
| `DELETE` | `/api/scripture/{uid}` | Delete scripture |
| `POST` | `/api/scripture/context` | Build RAG context |
| `POST` | `/api/scripture/{uid}/reindex` | Re-index scripture |
| `POST` | `/api/scripture/backfill` | Backfill missing embeddings |
| `GET` | `/api/scripture/stats` | Scripture statistics |

### Personalities
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/personalities` | Personality manager UI |
| `GET` | `/api/personalities` | List all personalities |
| `GET` | `/api/personalities/categories` | Get unique categories |
| `GET` | `/api/personality/{id}` | Get personality by ID |
| `POST` | `/api/personality` | Create personality |
| `PUT` | `/api/personality/{id}` | Update personality |
| `DELETE` | `/api/personality/{id}` | Delete personality |
| `POST` | `/api/personality/{id}/duplicate` | Duplicate personality |
| `GET` | `/api/personality/{id}/export` | Export to JSON |
| `POST` | `/api/personality/import` | Import from JSON |

### Security
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/security` | Security settings UI |
| `GET` | `/security/setup` | Initial setup UI |
| `GET` | `/lock` | Lock screen UI |
| `GET` | `/api/security/status` | Get lock/encryption status |
| `POST` | `/api/security/setup` | Initial master password setup |
| `POST` | `/api/security/unlock` | Unlock with password |
| `POST` | `/api/security/lock` | Lock session |
| `POST` | `/api/security/change-password` | Change master password |
| `POST` | `/api/security/settings` | Update security settings |
| `POST` | `/api/security/disable` | Disable encryption |
| `POST` | `/api/security/reset` | Factory reset security |

### Files (FileSystemGate)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/files` | File browser UI |
| `GET` | `/api/files/folders` | List managed folders |
| `POST` | `/api/files/folders` | Add managed folder |
| `GET` | `/api/files/folders/{id}` | Get folder config |
| `PUT` | `/api/files/folders/{id}` | Update folder |
| `DELETE` | `/api/files/folders/{id}` | Remove folder |
| `GET` | `/api/files/list` | List directory contents |
| `GET` | `/api/files/read` | Read file content |
| `POST` | `/api/files/write` | Write file content |
| `POST` | `/api/files/mkdir` | Create directory |
| `DELETE` | `/api/files/delete` | Delete file/directory |
| `GET` | `/api/files/info` | Get file metadata |
| `GET` | `/api/files/backups` | List backups |
| `POST` | `/api/files/backups/{id}/restore` | Restore backup |
| `DELETE` | `/api/files/backups/{id}` | Delete backup |
| `GET` | `/api/files/backups/stats` | Backup statistics |
| `GET` | `/api/files/config` | Get FileSystemGate config |

### Shell (ShellGate)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/shell` | Shell interface UI |
| `POST` | `/api/shell/execute` | Execute command (sync) |
| `POST` | `/api/shell/stream` | Execute with streaming |
| `POST` | `/api/shell/background` | Execute in background |
| `GET` | `/api/shell/status/{id}` | Get background status |
| `POST` | `/api/shell/cancel/{id}` | Cancel background command |
| `GET` | `/api/shell/running` | List running commands |
| `GET` | `/api/shell/background` | List all background commands |
| `GET` | `/api/shell/history` | Command history |
| `DELETE` | `/api/shell/history` | Clear history |
| `GET` | `/api/shell/validate` | Validate command safety |
| `GET` | `/api/shell/config` | Get ShellGate config |
| `PUT` | `/api/shell/config` | Update config |
| `POST` | `/api/shell/blocked` | Add to blocklist |
| `DELETE` | `/api/shell/blocked` | Remove from blocklist |

### Browser (BrowserGate)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/browser/status` | Get browser gate status |
| `POST` | `/api/browser/search` | Web search |
| `POST` | `/api/browser/fetch` | Fetch page content |

### SubAgents
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/agents` | SubAgent manager UI |
| `POST` | `/api/agents/spawn` | Spawn new sub-agent |
| `GET` | `/api/agents` | List all agents |
| `GET` | `/api/agents/running` | List running agents |
| `GET` | `/api/agents/{id}` | Get agent status |
| `GET` | `/api/agents/{id}/result` | Get agent result |
| `POST` | `/api/agents/{id}/cancel` | Cancel agent |
| `POST` | `/api/agents/check` | Check for completed agents |
| `POST` | `/api/agents/cleanup` | Clean up old agents |

### ToolGate (Tool Calling)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/toolgate` | Tool Protocol editor UI |
| `GET` | `/api/toolgate/prompt` | Get current tool prompt config |
| `GET` | `/api/toolgate/prompt/default` | Get default tool prompt |
| `GET` | `/api/toolgate/prompt/warning` | Get edit warning message |
| `POST` | `/api/toolgate/prompt` | Update tool prompt (requires acknowledgment) |
| `POST` | `/api/toolgate/prompt/restore` | Restore default prompt |
| `GET` | `/api/toolgate/prompt/validate` | Validate prompt syntax |
| `GET` | `/api/toolgate/tools` | List available tools |
| `GET` | `/api/toolgate/status` | ToolGate health status |

---

## 2. Slash Commands (Chat Interface)

### Thread Management
| Command | Purpose |
|---------|---------|
| `/history` | Show current thread history |
| `/forget` | Clear current thread |
| `/export thread` | Export thread to scripture |
| `/import bios <path>` | Import bios file |
| `/import glyph <path>` | Import glyph file |

### Memory Operations
| Command | Purpose |
|---------|---------|
| `/search <query>` | Semantic search across memory |
| `/usearch <query>` | Unified search (all sources) |
| `/memory` or `/memstat` | Show memory status |
| `/remember <fact>` | Store an observation |
| `/memories [domain]` | List memories by domain |
| `/concept <name>` | Get concept details |
| `/pattern <cat> <name>` | Get pattern details |
| `/memstats` | Detailed memory statistics |
| `/discover <text>` | Run knowledge discovery |
| `/related <ref>` | Get related items |
| `/discovery` | Discovery service status |
| `/loomsearch <query>` | Search conversation memory |
| `/backfill` | Backfill missing embeddings |

### Metadata Channel
| Command | Purpose |
|---------|---------|
| `/meta <target>` | Query metadata |
| `/metafields` | List available metadata fields |

### Multi-Modal
| Command | Purpose |
|---------|---------|
| `/image <path>` | Analyze image |
| `/describe <path>` | Describe image content |
| `/compare <p1> <p2>` | Compare two images |
| `/transcribe <path>` | Transcribe audio |
| `/audio <path>` | Audio analysis |

### SubAgents
| Command | Purpose |
|---------|---------|
| `/spawn <task>` | Spawn background agent |
| `/agents` | List all agents |
| `/agent <id>` | Get agent status |
| `/result <id>` | Get agent result |
| `/cancel <id>` | Cancel agent |

### Scripture (Documents)
| Command | Purpose |
|---------|---------|
| `/store <path>` | Store file as scripture |
| `/scripture <ref>` | Get scripture by reference |
| `/scriptsearch <query>` | Search scriptures |
| `/scriptures [type]` | List scriptures |
| `/scriptstats` | Scripture statistics |
| `/scriptindex` | Re-index all scriptures |

### Personality Management
| Command | Purpose |
|---------|---------|
| `/personalities` | List all personalities |
| `/personality <id>` | Switch to personality |
| `/personality` | Show current personality |
| `/personality-info <id>` | Get personality details |
| `/personality-create <name>` | Create new personality |
| `/personality-delete <id>` | Delete personality |
| `/personality-export <id>` | Export personality |
| `/personality-copy <id> <name>` | Duplicate personality |

### File Operations
| Command | Purpose |
|---------|---------|
| `/sources` | List managed folders |
| `/sources-add <id> <path>` | Add managed folder |
| `/ls <folder:path>` | List directory |
| `/cat <folder:path>` | Read file |
| `/writefile <folder:path>` | Write file (interactive) |
| `/mkdir <folder:path>` | Create directory |
| `/rm <folder:path>` | Delete file/directory |
| `/backups [folder]` | List backups |
| `/restore <backup_id>` | Restore backup |

### Shell Execution
| Command | Purpose |
|---------|---------|
| `/shell <cmd>` | Execute command |
| `/shellbg <cmd>` | Execute in background |
| `/shellstatus <id>` | Get command status |
| `/shellkill <id>` | Cancel command |
| `/shellhistory` | Command history |

### Web Browsing
| Command | Purpose |
|---------|---------|
| `/websearch <query>` | Web search |
| `/fetch <url>` | Fetch page content |
| `/browse <query>` | Search + fetch top results |

### Security
| Command | Purpose |
|---------|---------|
| `/lock` | Lock session |
| `/security` | Security status |
| `/security-status` | Detailed security status |

---

## 3. Gate Modules

### StarMirror (LLM Interface)
**Purpose**: Multi-model LLM abstraction layer

**Features**:
- OpenRouter API integration (40+ models)
- Claude CLI backend
- Codex CLI backend
- Vision model support
- Audio transcription
- Streaming responses

**Key Functions**:
```python
reflect(messages)           # Sync LLM call
reflect_stream(messages)    # Async streaming
transmit_vision(image, prompt)  # Image analysis
transcribe_audio(path)      # Audio-to-text
```

### MemoryGate (Knowledge System)
**Purpose**: Persistent knowledge with semantic search

**Features**:
- Observations (facts with confidence)
- Patterns (synthesized insights)
- Concepts (entities with relationships)
- Graph relationships
- Chains (ordered sequences)
- Auto-extraction from conversations

**Storage**: PostgreSQL + pgvector (or SQLite + FAISS)

### ScriptureGate (Document Library)
**Purpose**: Indexed document storage with RAG

**Features**:
- File storage with organization
- Text extraction from files
- Embedding generation
- Semantic search
- RAG context injection
- Multiple file types (document, image, audio, artifact, thread)

**Storage**: File system + PostgreSQL index

### PersonalityGate (Agent Personas)
**Purpose**: Agent personality management

**Features**:
- Built-in personalities
- Custom personality CRUD
- Per-thread personality
- LLM config (model, temperature, system prompt)
- Import/export for sharing

### SecurityManager (Encryption & Auth)
**Purpose**: Encryption and access control

**Features**:
- AES-256-GCM encryption
- Argon2id key derivation
- Master password protection
- Session locking with timeout
- Component-level encryption

### FileSystemGate (Secure Files)
**Purpose**: Secure file system access

**Features**:
- Folder-based permissions (read/write/none)
- Path traversal prevention
- Extension blocking
- Auto-backup on modify/delete
- Audit logging

### ShellGate (Command Execution)
**Purpose**: Secure command execution

**Features**:
- Blocklist/allowlist validation
- Dangerous construct detection
- Environment sanitization
- Background execution
- Command history
- Risk estimation

### BrowserGate (Web Access)
**Purpose**: Web search and page fetching

**Features**:
- Multi-provider search (DuckDuckGo, SearXNG, Brave)
- Page fetching (simple or headless)
- Content conversion (markdown/text/html)
- Browser extension WebSocket server

### SubAgentGate (Background Workers)
**Purpose**: Spawn and manage async tasks

**Features**:
- Background task execution
- Status tracking
- Result storage
- Task cancellation
- Cleanup of old tasks

### MetadataChannel (Internal Routing)
**Purpose**: Metadata routing between systems

**Features**:
- Provider registration
- Field-based queries
- Policy enforcement (rate limiting, restrictions)

### ToolGate (Tool Calling)
**Purpose**: Provider-agnostic tool calling for AI agents

**Features**:
- JSON-in-text protocol (works with any LLM)
- 31 tools across 6 Gates
- Policy classes: READ_ONLY, WRITE, DESTRUCTIVE, PRIVILEGED, NETWORK
- Bounded execution loop (max iterations, calls per step)
- Configurable, versioned system prompt
- Safe fallback on broken custom prompts
- UI for prompt editing and tool browsing

**Key Functions**:
```python
build_tool_prompt(policies)    # Generate tool prompt for LLM
parse_tool_calls(response)     # Extract tool calls from response
execute_single(call)           # Execute one tool call
create_orchestrator(policies)  # Create execution orchestrator
```

**Available Tools**:
| Gate | Tools |
|------|-------|
| MemoryGate | search, recall, store_observation, get_concept, get_related, add_relationship, get_stats |
| FileSystemGate | list_dir, read_file, write_file, mkdir, delete, info, list_folders |
| ShellGate | execute, validate_command, estimate_risk, get_history |
| ScriptureGate | search, build_context, store_text, get, list_scriptures, stats |
| BrowserGate | search, fetch |
| SubAgentGate | spawn, status, result, list_agents, cancel |

---

## 4. Configuration

### Configuration Categories

| Category | Key Settings |
|----------|--------------|
| **API_KEYS** | `OPENROUTER_API_KEY`, `OPENAI_API_KEY` |
| **DATABASE** | `DATABASE_URL`, `DB_BACKEND` (postgres/sqlite), `VECTOR_BACKEND` (pgvector/faiss) |
| **PATHS** | `DATA_DIR`, `SCRIPTURE_DIR`, `AGENTS_DIR`, `MODELS_DIR` |
| **SERVER** | `HOST`, `PORT`, `ALLOWED_ORIGINS`, `DEBUG`, `LOG_LEVEL` |
| **MODELS** | `DEFAULT_MODEL`, `VISION_MODEL`, `EMBEDDING_MODEL`, `LOOMMIRROR_MODEL_PATH` |
| **FEATURES** | `ENABLE_MEMORY_GATE`, `ENABLE_SCRIPTURE_RAG`, `ENABLE_SUBAGENTS`, `ENABLE_MULTIMODAL`, `AUTO_EXTRACT_MEMORY` |
| **TOOL_PROTOCOL** | `TOOL_PROTOCOL_PROMPT`, `TOOL_MAX_ITERATIONS`, `TOOL_MAX_CALLS_PER_STEP`, `TOOL_MAX_TOTAL_CALLS` |

### Configuration Priority
1. Environment variables (highest)
2. `data/config.json`
3. Schema defaults (lowest)

---

## 5. Data Flow

```
User Input → Chat Pipeline
    │
    ├─ Slash Command? → Command Router → Gate Operations
    │
    └─ Regular Message:
        │
        ├─ 1. Append to Loom (conversation memory)
        │
        ├─ 2. Build Context (Cathedral Context Assembly Order):
        │     ├─ [0] Tool Protocol prompt (if tools enabled)
        │     ├─ [1] Personality system prompt
        │     ├─ [2] Current user message
        │     ├─ [3] Memory context (MemoryGate)
        │     ├─ [4] RAG context (ScriptureGate)
        │     └─ [5+] Prior conversation history
        │
        ├─ 3. Call StarMirror (LLM)
        │     └─ Stream tokens → SSE → Client
        │
        ├─ 4. Tool Execution Loop (if tools enabled):
        │     ├─ Parse tool calls from response
        │     ├─ Execute via ToolGate orchestrator
        │     ├─ Inject results into conversation
        │     └─ Get next response (repeat until done)
        │
        ├─ 5. Store Final Response in Loom
        │
        └─ 6. Post-Processing:
              ├─ Auto-extract memory (if enabled)
              └─ Emit completion events
```

---

## 6. Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI + Uvicorn |
| **Database** | PostgreSQL + pgvector (or SQLite + FAISS) |
| **ORM** | SQLAlchemy (async support) |
| **LLM API** | OpenRouter (40+ models) |
| **Local LLM** | llama-cpp-python (TinyLlama for summarization) |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Encryption** | AES-256-GCM + Argon2id |
| **Events** | Server-Sent Events (SSE) |
| **Frontend** | Jinja2 templates + Tailwind CSS |
| **Tool Calling** | Provider-agnostic JSON protocol (31 tools) |

---

## 7. Quick Start

### Running the Server
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run server
python -m altar.run
# Or with hot reload:
uvicorn altar.run:app --reload
```

### Required Configuration
- `OPENROUTER_API_KEY` - For LLM access
- `OPENAI_API_KEY` - For embeddings (semantic search)
- `DATABASE_URL` - PostgreSQL connection string (optional, defaults to SQLite)

### Default URLs
- Chat UI: `http://localhost:8000/`
- Config: `http://localhost:8000/config`
- Health: `http://localhost:8000/api/health`
