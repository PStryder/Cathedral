# Cathedral

<div align="center">

**A modular AI assistant platform with persistent memory, semantic search, and system integration.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#features) | [Quick Start](#quick-start) | [Configuration](#configuration) | [Usage](#usage) | [API](#api-reference) | [Development](#development)

</div>

---

## Overview

Cathedral is a self-hosted AI conversation platform that combines:

- **Persistent Memory** - Every conversation is stored and semantically searchable
- **Knowledge Extraction** - Automatically extracts facts, concepts, and patterns from conversations
- **RAG Integration** - Store documents and inject relevant context into prompts
- **System Integration** - Secure file access, shell commands, web browsing
- **Multi-Modal Support** - Image analysis, audio transcription, vision understanding
- **Modular Architecture** - Enable/disable features via configuration

Built with FastAPI, PostgreSQL + pgvector, and designed for local deployment.

---

## Features

### Conversation & Memory

| Feature | Description |
|---------|-------------|
| **Multi-Thread Chats** | Independent conversation threads with separate histories |
| **Semantic Search** | All messages embedded and searchable via vector similarity |
| **Thread Summaries** | Automatic summarization of long conversations |
| **Personality Per Thread** | Different agent behaviors for different threads |
| **Context Injection** | Automatic retrieval of relevant past conversations |

### Knowledge System (MemoryGate)

| Feature | Description |
|---------|-------------|
| **Observations** | Store facts with confidence scores and domains |
| **Concepts** | Knowledge graph nodes with relationships |
| **Patterns** | Synthesized insights across conversations |
| **Auto-Extraction** | Background extraction of knowledge from chats |
| **Knowledge Discovery** | Async discovery of relationships via embeddings |

### Document Library (ScriptureGate)

| Feature | Description |
|---------|-------------|
| **Document Storage** | Store files with automatic organization |
| **Content Extraction** | Extract text from PDFs, images, audio |
| **Semantic Search** | Find documents by meaning, not just keywords |
| **RAG Context** | Automatically inject relevant documents into prompts |
| **Multiple Types** | Documents, images, audio, artifacts, threads |

### Tool Calling (ToolGate)

| Feature | Description |
|---------|-------------|
| **Provider-Agnostic** | JSON-in-text protocol works with any LLM |
| **31 Tools** | Across 6 Gates: Memory, Files, Shell, Scripture, Browser, SubAgents |
| **Policy Control** | 5 policy classes: READ_ONLY, WRITE, DESTRUCTIVE, PRIVILEGED, NETWORK |
| **Bounded Execution** | Configurable limits (iterations, calls per step, total calls) |
| **Configurable Prompt** | Versioned, editable system prompt with safe fallback |
| **UI Integration** | Tools toggle, inline execution display, protocol editor |

### System Integration

| Feature | Description |
|---------|-------------|
| **Secure Shell** | Command execution with blocklist/allowlist validation |
| **File Management** | Folder-based permissions with auto-backup |
| **Web Browsing** | Multi-provider search (DuckDuckGo, Brave, SearXNG) |
| **Page Fetching** | Simple or headless browser with content conversion |
| **Sub-Agents** | Spawn async worker tasks for background processing |
| **Browser Extension** | WebSocket server for browser integration |

### Multi-Modal

| Feature | Description |
|---------|-------------|
| **Vision** | Analyze images with vision-capable models |
| **Image Description** | Detailed image content descriptions |
| **Image Comparison** | Compare two images |
| **Audio Transcription** | Speech-to-text via Whisper API |

### Security

| Feature | Description |
|---------|-------------|
| **AES-256-GCM** | Military-grade encryption for sensitive data |
| **Argon2id** | Secure password hashing |
| **Session Locking** | Timeout-based auto-lock |
| **Path Validation** | Prevent directory traversal attacks |
| **Command Filtering** | Block dangerous shell commands |

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **PostgreSQL 15+** with pgvector extension (or SQLite for testing)
- **OpenRouter API key** (for LLM access)
- **OpenAI API key** (for embeddings)

### Installation

```bash
# Clone the repository
git clone https://github.com/PStryder/Cathedral.git
cd Cathedral

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Local LLM for summarization
pip install -r requirements-llama.txt
```

### Database Setup

#### Option A: PostgreSQL (Recommended for Production)

```sql
-- Create database
CREATE DATABASE cathedral;
\c cathedral

-- Enable pgvector extension
CREATE EXTENSION vector;
```

```bash
# Set environment variable
export DATABASE_URL="postgresql://user:password@localhost:5432/cathedral"
```

#### Option B: SQLite (Quick Start / Development)

```bash
# SQLite requires no setup - just set the URL
export DATABASE_URL="sqlite+aiosqlite:///./data/cathedral.db"
export DB_BACKEND="sqlite"
export VECTOR_BACKEND="faiss"
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env  # or your preferred editor
```

**Minimum required settings:**

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/cathedral

# API Keys
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
```

### Run the Server

```bash
# Production
python -m altar.run

# Development (with auto-reload)
uvicorn altar.run:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Configuration

Cathedral uses a layered configuration system:

1. **Environment variables** (highest priority)
2. **`data/config.json`** (persistent settings)
3. **Schema defaults** (lowest priority)

Access the configuration UI at `/config` or use the REST API.

### All Configuration Options

#### API Keys

| Setting | Description | Required |
|---------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM access | Yes |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes |

#### Database

| Setting | Description | Default |
|---------|-------------|---------|
| `DATABASE_URL` | Database connection string | Required |
| `DB_BACKEND` | Database type: `postgres` or `sqlite` | `postgres` |
| `VECTOR_BACKEND` | Vector store: `pgvector` or `faiss` | `pgvector` |
| `AUTO_MIGRATE_ON_STARTUP` | Auto-create tables | `true` |
| `AUTO_CREATE_EXTENSIONS` | Auto-create pgvector extension | `true` |

#### Models

| Setting | Description | Default |
|---------|-------------|---------|
| `DEFAULT_MODEL` | Default LLM model | `openai/gpt-4o` |
| `VISION_MODEL` | Vision-capable model | `openai/gpt-4o` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `EMBEDDING_DIM` | Embedding dimensions | `1536` |
| `LOOMMIRROR_MODEL_PATH` | Path to local GGUF model | Optional |

#### Server

| Setting | Description | Default |
|---------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated) | `*` |
| `DEBUG` | Debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |

#### Features

| Setting | Description | Default |
|---------|-------------|---------|
| `ENABLE_MEMORY_GATE` | Enable knowledge system | `true` |
| `ENABLE_SCRIPTURE_RAG` | Enable document RAG | `true` |
| `ENABLE_SUBAGENTS` | Enable sub-agent spawning | `true` |
| `ENABLE_MULTIMODAL` | Enable vision/audio | `true` |
| `AUTO_EXTRACT_MEMORY` | Auto-extract from conversations | `true` |

#### Tool Calling (ToolGate)

| Setting | Description | Default |
|---------|-------------|---------|
| `TOOL_PROTOCOL_PROMPT` | Custom tool protocol prompt | (built-in) |
| `TOOL_MAX_ITERATIONS` | Max tool execution loop iterations | `6` |
| `TOOL_MAX_CALLS_PER_STEP` | Max tool calls per iteration | `5` |
| `TOOL_MAX_TOTAL_CALLS` | Max total calls per request | `20` |

#### Paths

| Setting | Description | Default |
|---------|-------------|---------|
| `DATA_DIR` | Runtime data directory | `data` |
| `SCRIPTURE_DIR` | Document storage | `data/scripture` |
| `AGENTS_DIR` | Sub-agent data | `data/agents` |
| `MODELS_DIR` | Local model storage | `models` |

---

## Usage

### Web Interface

The primary interface is the web UI at `http://localhost:8000/`:

- **Chat** - Main conversation interface (with tools toggle)
- **Config** (`/config`) - Configuration editor
- **Memory** (`/memory`) - Memory browser
- **Scripture** (`/scripture`) - Document library
- **Personalities** (`/personalities`) - Personality manager
- **Security** (`/security`) - Encryption settings
- **Files** (`/files`) - File browser
- **Shell** (`/shell`) - Command interface
- **Agents** (`/agents`) - Sub-agent manager
- **Tool Protocol** (`/toolgate`) - Tool calling configuration

### Slash Commands

Cathedral supports 60+ slash commands in the chat interface.

#### Thread Management
```
/history              Show current thread history
/forget               Clear thread memory
/export thread        Export thread to scripture
/import bios <path>   Import bios file
/import glyph <path>  Import glyph file
```

#### Memory Operations
```
/search <query>       Semantic search across all memory
/usearch <query>      Unified search (all sources)
/memory               Show memory status
/remember <fact>      Store an observation
/memories [domain]    List memories by domain
/concept <name>       Get concept details
/pattern <cat> <name> Get pattern details
/memstats             Detailed memory statistics
/discover <text>      Run knowledge discovery
/related <ref>        Get related items
/discovery            Discovery service status
/loomsearch <query>   Search conversation memory
/backfill             Backfill missing embeddings
```

#### Documents (Scripture)
```
/store <path>         Store file as scripture
/scripture <ref>      Get scripture by reference
/scriptsearch <query> Search documents
/scriptures [type]    List documents
/scriptstats          Document statistics
/scriptindex          Re-index all documents
```

#### Multi-Modal
```
/image <path>         Analyze image
/describe <path>      Describe image content
/compare <p1> <p2>    Compare two images
/transcribe <path>    Transcribe audio
/audio <path>         Audio analysis
```

#### Sub-Agents
```
/spawn <task>         Spawn background agent
/agents               List all agents
/agent <id>           Get agent status
/result <id>          Get agent result
/cancel <id>          Cancel agent
```

#### Personalities
```
/personalities           List all personalities
/personality <id>        Switch to personality
/personality             Show current personality
/personality-info <id>   Get personality details
/personality-create <n>  Create new personality
/personality-delete <id> Delete personality
/personality-export <id> Export personality
/personality-copy <id>   Duplicate personality
```

#### File Operations
```
/sources              List managed folders
/sources-add <id> <p> Add managed folder
/ls <folder:path>     List directory
/cat <folder:path>    Read file
/writefile <f:path>   Write file
/mkdir <folder:path>  Create directory
/rm <folder:path>     Delete file/directory
/backups [folder]     List backups
/restore <backup_id>  Restore backup
```

#### Shell Execution
```
/shell <cmd>          Execute command
/shellbg <cmd>        Execute in background
/shellstatus <id>     Get command status
/shellkill <id>       Cancel command
/shellhistory         Command history
```

#### Web Browsing
```
/websearch <query>    Web search
/fetch <url>          Fetch page content
/browse <query>       Search + fetch top results
```

#### Security
```
/lock                 Lock session
/security             Security status
/security-status      Detailed security status
```

#### Metadata
```
/meta <target>        Query metadata
/metafields           List available fields
```

---

## API Reference

Cathedral exposes a comprehensive REST API. Full documentation available at `/docs` (Swagger UI) when the server is running.

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main chat interface |
| `POST` | `/api/chat/stream` | Stream chat response (SSE) |
| `GET` | `/api/threads` | List conversation threads |
| `POST` | `/api/thread` | Create/switch thread |
| `GET` | `/api/thread/{uid}/history` | Get thread history |
| `GET` | `/api/events` | Subscribe to events (SSE) |
| `GET` | `/api/health` | System health status |

### ToolGate Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/toolgate` | Tool Protocol editor UI |
| `GET` | `/api/toolgate/prompt` | Get current tool prompt config |
| `POST` | `/api/toolgate/prompt` | Update tool prompt (requires acknowledgment) |
| `POST` | `/api/toolgate/prompt/restore` | Restore default prompt |
| `GET` | `/api/toolgate/prompt/validate` | Validate prompt syntax |
| `GET` | `/api/toolgate/tools` | List available tools |
| `GET` | `/api/toolgate/status` | ToolGate health status |

### Full API Documentation

See [`docs/FEATURES.md`](docs/FEATURES.md) for complete endpoint documentation including:

- 90+ REST endpoints across 13 routers
- Request/response formats
- Authentication requirements
- Example usage

---

## Architecture

### Gate Pattern

Cathedral uses a "Gate" pattern where each subsystem is independently initialized and can be enabled/disabled:

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│                          (altar/run.py)                         │
├─────────────────────────────────────────────────────────────────┤
│                         API Routers                             │
│  chat | config | health | memory | scripture | personalities   │
│  security | files | shell | browser | subagent | toolgate      │
├─────────────────────────────────────────────────────────────────┤
│                       Chat Pipeline                             │
│              (cathedral/pipeline/chat.py)                       │
├───────────┬───────────┬───────────┬───────────┬────────────────┤
│ StarMirror│  ToolGate │MemoryGate │Scripture  │  Personality   │
│   (LLM)   │  (Tools)  │(Knowledge)│  (RAG)    │    (Agent)     │
├───────────┼───────────┼───────────┼───────────┼────────────────┤
│FileSystem │  Shell    │  Browser  │ SubAgent  │   Security     │
│  (Files)  │(Commands) │   (Web)   │ (Workers) │   (Crypto)     │
├───────────┴───────────┴───────────┴───────────┴────────────────┤
│                    Shared Utilities                             │
│         (cathedral/shared: logging, config, db, paths)          │
├─────────────────────────────────────────────────────────────────┤
│                       Database Layer                            │
│           PostgreSQL + pgvector  |  SQLite + FAISS              │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input
    │
    ├─ Slash Command? ──► Command Router ──► Gate Operations
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
        │     └─ Stream tokens ──► SSE ──► Client
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
              ├─ Auto-extract memory
              └─ Emit completion events
```

### Project Structure

```
Cathedral/
├── altar/                    # FastAPI server
│   ├── run.py                # Application entry point
│   ├── lifecycle.py          # Startup/shutdown handlers
│   ├── api/                  # REST API routers
│   │   ├── chat.py           # Chat endpoints
│   │   ├── config.py         # Configuration endpoints
│   │   ├── health.py         # Health check endpoints
│   │   ├── memory.py         # MemoryGate endpoints
│   │   ├── scripture.py      # ScriptureGate endpoints
│   │   ├── personalities.py  # Personality endpoints
│   │   ├── security.py       # Security endpoints
│   │   ├── files.py          # FileSystemGate endpoints
│   │   ├── shell.py          # ShellGate endpoints
│   │   ├── browser.py        # BrowserGate endpoints
│   │   ├── subagent.py       # SubAgentGate endpoints
│   │   ├── toolgate.py       # ToolGate endpoints
│   │   └── events.py         # SSE events endpoint
│   ├── services/             # Event bus, agent tracker
│   ├── middleware/           # Security middleware
│   ├── templates/            # Jinja2 HTML templates
│   └── static/               # CSS, JS, images
│
├── cathedral/                # Core subsystems
│   ├── StarMirror/           # LLM interface
│   │   ├── router.py         # Multi-backend routing
│   │   └── providers/        # OpenRouter, Claude CLI, Codex
│   ├── MemoryGate/           # Knowledge system
│   │   ├── conversation/     # Loom conversation memory
│   │   └── discovery.py      # Knowledge discovery
│   ├── ScriptureGate/        # Document library
│   │   ├── storage.py        # File storage
│   │   ├── indexer.py        # Embedding generation
│   │   └── models.py         # Database models
│   ├── PersonalityGate/      # Personality management
│   │   ├── models.py         # Personality schema
│   │   └── defaults.py       # Built-in personalities
│   ├── SecurityManager/      # Encryption & auth
│   │   ├── crypto.py         # AES-256-GCM + Argon2
│   │   └── session.py        # Session management
│   ├── FileSystemGate/       # File access control
│   │   ├── security.py       # Path validation
│   │   ├── operations.py     # File operations
│   │   └── backup.py         # Backup management
│   ├── ShellGate/            # Command execution
│   │   ├── security.py       # Command validation
│   │   └── executor.py       # Process execution
│   ├── BrowserGate/          # Web access
│   │   ├── providers/        # Search providers
│   │   ├── fetcher.py        # Page fetching
│   │   └── websocket_server.py  # Browser extension
│   ├── SubAgentGate/         # Worker agents
│   ├── ToolGate/             # Tool calling system
│   │   ├── models.py         # Protocol types (ToolCall, ToolResult)
│   │   ├── registry.py       # Tool registry (31 tools)
│   │   ├── protocol.py       # JSON parsing/validation
│   │   ├── orchestrator.py   # Execution loop
│   │   ├── policy.py         # Policy management
│   │   ├── prompt.py         # Tool prompt generation
│   │   └── prompt_config.py  # Configurable prompt storage
│   ├── MetadataChannel/      # Metadata routing
│   ├── Memory/               # Unified memory interface
│   ├── Config/               # Configuration management
│   │   └── schema.py         # Config schema definition
│   ├── commands/             # Slash command router
│   ├── pipeline/             # Chat processing pipeline
│   ├── runtime.py            # Lazy-loaded proxies
│   └── shared/               # Shared utilities
│       ├── gate.py           # Gate base utilities
│       ├── db.py             # Database abstraction
│       └── db_service.py     # DB initialization
│
├── data/                     # Runtime data
│   ├── config.json           # Persistent configuration
│   ├── personalities/        # Custom personalities
│   ├── scripture/            # Document storage
│   ├── agents/               # Sub-agent data
│   └── backups/              # File backups
│
├── models/                   # Local models
│   └── memory/               # LoomMirror GGUF models
│
├── tests/                    # Test suite
│   ├── conftest.py           # Pytest fixtures
│   ├── test_*.py             # Unit tests
│   └── ...
│
├── docs/                     # Documentation
│   └── FEATURES.md           # Complete feature reference
│
├── .env.example              # Environment template
├── requirements.txt          # Python dependencies
├── requirements-llama.txt    # Local LLM dependencies
└── pytest.ini                # Pytest configuration
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI 0.115+ / Uvicorn |
| **Database** | PostgreSQL 15+ with pgvector (or SQLite + FAISS) |
| **ORM** | SQLAlchemy 2.0 (async) |
| **LLM API** | OpenRouter (40+ models) |
| **Local LLM** | llama-cpp-python (TinyLlama 1.1B for summarization) |
| **Embeddings** | OpenAI text-embedding-3-small (1536 dim) |
| **Encryption** | AES-256-GCM + Argon2id (via cryptography lib) |
| **Events** | Server-Sent Events (SSE) |
| **Frontend** | Jinja2 templates + vanilla JS + Tailwind CSS |
| **Search** | DuckDuckGo / SearXNG / Brave |
| **Tool Calling** | Provider-agnostic JSON-in-text protocol (31 tools) |

---

## Development

### Running in Development

```bash
# With auto-reload
uvicorn altar.run:app --reload --port 8000

# With debug logging
DEBUG=true LOG_LEVEL=DEBUG python -m altar.run
```

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=cathedral --cov-report=html

# Specific test file
pytest tests/test_filesystemgate.py -v

# Skip slow/network tests
pytest tests/ -m "not slow and not network"
```

### Code Quality

```bash
# Type checking
mypy cathedral/

# Linting
ruff check cathedral/

# Formatting
ruff format cathedral/
```

### Database Migrations

Cathedral uses auto-migration by default. Tables are created on startup.

```bash
# Disable auto-migration
AUTO_MIGRATE_ON_STARTUP=false

# Manual migration (if needed)
python -c "from cathedral.shared.db_service import init_db; init_db('your-url')"
```

### Adding a New Gate

1. Create module in `cathedral/NewGate/`
2. Implement `__init__.py` with:
   - `initialize()` function
   - `is_healthy()` health check
   - `get_health_status()` detailed status
   - `__all__` exports
3. Add to `cathedral/__init__.py` exports
4. Initialize in `altar/lifecycle.py`
5. Create API router in `altar/api/newgate.py`
6. Add router to `altar/run.py`
7. Add tests in `tests/test_newgate.py`
8. (Optional) Register tools in `cathedral/ToolGate/registry.py` for agentic access

---

## Deployment

### Docker (Coming Soon)

```dockerfile
# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "altar.run"]
```

### Production Checklist

- [ ] Use PostgreSQL (not SQLite) for production
- [ ] Set `DEBUG=false`
- [ ] Configure `ALLOWED_ORIGINS` for CORS
- [ ] Use environment variables for secrets (not `.env`)
- [ ] Enable HTTPS via reverse proxy (nginx/caddy)
- [ ] Set up database backups
- [ ] Monitor with `/api/health` endpoint
- [ ] Configure log aggregation

### Reverse Proxy (nginx example)

```nginx
server {
    listen 443 ssl;
    server_name cathedral.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
    }
}
```

---

## Troubleshooting

### Common Issues

#### "pgvector extension not found"

```sql
-- Connect to your database and run:
CREATE EXTENSION vector;
```

Or set `AUTO_CREATE_EXTENSIONS=true` in your environment.

#### "Module not found: cathedral"

Ensure you're running from the project root:

```bash
cd /path/to/Cathedral
python -m altar.run
```

#### "OpenRouter API error"

1. Check your `OPENROUTER_API_KEY` is valid
2. Verify the model name is correct (e.g., `openai/gpt-4o`)
3. Check your OpenRouter account has credits

#### "Embedding generation failed"

1. Check your `OPENAI_API_KEY` is valid
2. Ensure the embedding model exists (`text-embedding-3-small`)
3. Check OpenAI API status

#### "Database connection failed"

1. Verify PostgreSQL is running
2. Check `DATABASE_URL` format: `postgresql://user:pass@host:port/dbname`
3. Ensure the database exists and user has permissions

#### "SQLite async error"

Install aiosqlite:

```bash
pip install aiosqlite
```

### Getting Help

1. Check the [docs/FEATURES.md](docs/FEATURES.md) for complete API reference
2. Review server logs (`LOG_LEVEL=DEBUG` for verbose output)
3. Check `/api/health` for system status
4. Open an issue on GitHub

---

## Security Considerations

- **Local deployment only** - Cathedral is designed for local/private use
- **Use strong passwords** - Argon2id is secure, but weak passwords aren't
- **Review shell commands** - Check the ShellGate blocklist configuration
- **Limit folder access** - Only grant `read_write` to necessary folders
- **Protect API keys** - Use environment variables, not config files
- **Network isolation** - Use firewall rules if exposed on network
- **Regular backups** - Enable FileSystemGate auto-backup feature
- **Tool policy control** - ToolGate defaults to READ_ONLY; explicitly enable WRITE/PRIVILEGED policies
- **Tool prompt protection** - Custom tool prompts require explicit risk acknowledgment

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure tests pass (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow existing code patterns
- Add docstrings to public functions
- Include `__all__` exports in modules
- Write tests for new features
- Update documentation as needed

---

## License

MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [OpenRouter](https://openrouter.ai/) - LLM API aggregator
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity for PostgreSQL
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Local LLM inference
