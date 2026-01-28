# Cathedral

A local AI chat application with persistent memory, knowledge discovery, and system integration capabilities.

Cathedral is a modular, extensible platform for AI-powered multi-modal conversation with knowledge preservation, security, and deep system integration. Built with Python/FastAPI backend and PostgreSQL + pgvector for semantic search.

## Features

### Core Capabilities

- **Multi-Thread Conversations** - Independent chat threads with separate histories and personalities
- **Semantic Memory** - All messages embedded and searchable via vector similarity
- **Knowledge Extraction** - Automatic extraction of observations, concepts, and patterns from conversations
- **Knowledge Discovery** - Background async discovery of relationships via embedding similarity
- **RAG Integration** - Document library with semantic search and context injection
- **Multi-Modal** - Image analysis, audio transcription, vision understanding
- **Customizable Personalities** - Per-thread agent behavior with model/temperature control

### System Integration

- **Secure Shell Execution** - Command validation, blocking, and sandboxing
- **File Management** - Controlled folder access with automatic backups
- **Web Browsing** - Search and fetch with multiple providers
- **Sub-Agent Delegation** - Spawn async worker tasks
- **Browser Extension** - WebSocket integration for context menus

### Security

- **AES-256-GCM Encryption** - Master password protection
- **Session Locking** - Timeout-based auto-lock
- **Component-Level Controls** - Granular encryption settings

## Architecture

Cathedral is organized into "Gates" - modular subsystems that can be enabled/disabled:

| Gate | Purpose |
|------|---------|
| **StarMirror** | LLM interface (OpenRouter API, 40+ models) |
| **Loom** | Conversation memory with semantic search |
| **MemoryGate** | Knowledge system (observations, concepts, patterns) |
| **ScriptureGate** | Document library and RAG |
| **PersonalityGate** | Agent personality management |
| **SecurityManager** | Encryption and access control |
| **FileSystemGate** | Managed file access with backups |
| **ShellGate** | Secure command execution |
| **BrowserGate** | Web search and page fetching |
| **SubAgentGate** | Async worker delegation |

## Requirements

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- OpenRouter API key (for LLM access)
- OpenAI API key (for embeddings)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PStryder/Cathedral.git
cd Cathedral
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up PostgreSQL with pgvector:
```sql
CREATE DATABASE cathedral;
\c cathedral
CREATE EXTENSION vector;
```

5. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

Required environment variables:
```
DATABASE_URL=postgresql://user:pass@localhost:5432/cathedral
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
```

6. Run the server:
```bash
python -m altar.run
```

The web UI will be available at `http://localhost:8000`

## Configuration

Cathedral uses a layered configuration system:

1. Environment variables (highest priority)
2. `data/config.json` (persistent settings)
3. Schema defaults (lowest priority)

Access the configuration UI at `/config` or use the API at `/api/config`.

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `OPENAI_API_KEY` | OpenAI API key (embeddings) | Required |
| `LLM_MODEL` | Default LLM model | gpt-4o |
| `EMBEDDING_MODEL` | Embedding model | text-embedding-3-small |
| `SERVER_PORT` | Server port | 8000 |

## Usage

### Slash Commands

Cathedral supports 60+ slash commands. Here are the most common:

#### Conversation
- `/history` - Show thread history
- `/forget` - Clear thread memory
- `/personality <id>` - Switch personality

#### Memory
- `/search <query>` - Semantic search all memory
- `/remember <text>` - Store an observation
- `/memories` - Recall recent observations
- `/discover <ref>` - Find related knowledge
- `/related <ref>` - Show relationships

#### Documents
- `/store <path>` - Store file as scripture
- `/scriptsearch <query>` - Search documents
- `/scriptures` - List stored documents

#### Multi-Modal
- `/image <path> <prompt>` - Analyze image
- `/describe <path>` - Describe image
- `/transcribe <audio>` - Transcribe audio

#### System
- `/shell <cmd>` - Execute command
- `/websearch <query>` - Search the web
- `/fetch <url>` - Fetch page content
- `/spawn <task>` - Spawn sub-agent

#### Files
- `/sources` - List managed folders
- `/ls <path>` - List directory
- `/cat <path>` - Read file

### API

Cathedral exposes a REST API for all functionality:

- `POST /api/chat/stream` - Stream chat response (SSE)
- `GET /api/threads` - List conversation threads
- `POST /api/thread` - Create/switch thread
- `GET /api/config` - Get configuration
- `GET /api/events` - Subscribe to events (SSE)

See the full API documentation at `/docs` when the server is running.

## Database Schema

### Loom (Conversation Memory)
- `threads` - Conversation threads
- `messages` - Chat messages with embeddings
- `summaries` - Thread summaries
- `facts` - Extracted facts

### MemoryGate (Knowledge)
- `observations` - Stored facts
- `concepts` - Knowledge graph nodes
- `patterns` - Synthesized patterns
- `relationships` - Entity connections
- `embeddings` - Unified vector storage

### ScriptureGate (Documents)
- `scriptures` - Stored documents with embeddings

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| Database | PostgreSQL + pgvector |
| ORM | SQLAlchemy (async) |
| LLM | OpenRouter API |
| Local LLM | TinyLlama 1.1B (GGUF) |
| Embeddings | OpenAI text-embedding-3-small |
| Encryption | AES-256-GCM + Argon2id |
| Events | Server-Sent Events (SSE) |

## Project Structure

```
Cathedral/
├── altar/              # FastAPI server
│   ├── run.py          # Main entry point
│   ├── templates/      # Jinja2 templates
│   └── static/         # Static assets
├── cathedral/          # Core subsystems
│   ├── StarMirror/     # LLM interface
│   ├── MemoryGate/     # Knowledge system
│   ├── ScriptureGate/  # Document library
│   ├── PersonalityGate/# Personality management
│   ├── SecurityManager/# Encryption
│   ├── FileSystemGate/ # File access
│   ├── ShellGate/      # Command execution
│   ├── BrowserGate/    # Web access
│   ├── SubAgentGate/   # Worker agents
│   ├── MetadataChannel/# Metadata routing
│   └── Config/         # Configuration
├── loom/               # Conversation memory
│   ├── models.py       # Database models
│   ├── db.py           # Database connection
│   └── embeddings.py   # Embedding generation
├── data/               # Runtime data
│   ├── config.json     # Persistent config
│   ├── scripture/      # Stored documents
│   └── agents/         # Agent data
└── models/             # Local models
    └── memory/         # TinyLlama for summarization
```

## Development

### Running in Development

```bash
# With auto-reload
uvicorn altar.run:app --reload --port 8000

# Or directly
python -m altar.run
```

### Running Tests

```bash
pytest tests/
```

### Database Migrations

Cathedral uses auto-migration on startup by default. To disable:

```
AUTO_MIGRATE_ON_STARTUP=false
```

## Security Considerations

- **Never expose Cathedral directly to the internet** - It's designed for local use
- **Use strong master passwords** - Argon2id provides good protection but weak passwords are still weak
- **Review shell commands** - ShellGate validates commands but review the blocklist
- **Manage folder permissions** - Only grant read_write to folders that need it
- **Keep API keys secure** - Use environment variables, not config files

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see the [LICENSE](LICENSE) file for details.
