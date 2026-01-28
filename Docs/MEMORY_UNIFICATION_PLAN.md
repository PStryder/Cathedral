# Cathedral Memory Unification Plan

**Version**: 1.0.0
**Date**: 2026-01-28
**Status**: Proposed
**Scope**: Refactor Cathedral to unify Loom and MemoryGate into a single memory stack

---

## 1. Overview

### 1.1 Current Architecture

```
Cathedral
├── Loom (conversation memory)
│   ├── PostgreSQL + pgvector
│   ├── Thread/Message storage
│   ├── OpenAI embeddings
│   ├── Semantic search
│   └── LoomMirror (summarization)
│
├── MemoryGate (via MCP)
│   ├── Observations, Patterns, Concepts
│   ├── Chains, Relationships
│   └── Hot/cold tiering
│
└── Problem: Two separate memory systems with redundant:
    - Database connections
    - Embedding pipelines
    - Search implementations
    - Session tracking
```

### 1.2 Target Architecture

```
Cathedral
└── UnifiedMemory (single memory service)
    ├── Conversation Layer (replaces Loom)
    │   └── Backed by MemoryGate MCP tools
    │
    ├── Knowledge Layer (MemoryGate)
    │   └── Observations, Patterns, Concepts, etc.
    │
    ├── Shared Infrastructure
    │   ├── Single embedding pipeline
    │   ├── Single PostgreSQL connection
    │   └── Unified semantic search
    │
    └── Cross-pollination
        ├── Conversations → Observations
        └── Observations → Context injection
```

---

## 2. Phased Implementation

### Phase 1: Abstraction Layer (Non-Breaking)
**Goal**: Create unified memory interface without removing Loom

### Phase 2: Migration
**Goal**: Migrate Loom storage to MemoryGate-backed implementation

### Phase 3: Cleanup
**Goal**: Remove Loom, simplify codebase

---

## 3. Phase 1: Abstraction Layer

### 3.1 Create `cathedral/Memory/__init__.py`

New unified memory module that wraps both systems:

```python
"""
Cathedral Unified Memory

Provides a single interface for:
- Conversation memory (threads, messages, context)
- Knowledge memory (observations, patterns, concepts)
- Cross-system search and context composition
"""

from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

# Import underlying systems
from loom import Loom
from cathedral import MemoryGate


class MemorySource(Enum):
    CONVERSATION = "conversation"
    OBSERVATION = "observation"
    PATTERN = "pattern"
    CONCEPT = "concept"
    DOCUMENT = "document"


@dataclass
class SearchResult:
    """Unified search result from any memory source."""
    source: MemorySource
    content: str
    similarity: float
    ref: str  # observation:123, message:uuid, etc.
    metadata: Dict[str, Any]


class UnifiedMemory:
    """
    Unified memory interface for Cathedral.

    Combines conversation context (Loom) with knowledge memory (MemoryGate)
    into a single coherent interface.
    """

    def __init__(self):
        self._loom = Loom()
        self._mg_initialized = False

    # ==========================================
    # Conversation Layer (Loom wrapper)
    # ==========================================

    async def create_thread(
        self,
        thread_name: str = None,
        metadata: Dict = None
    ) -> str:
        """Create a new conversation thread."""
        return self._loom.create_new_thread(thread_name)

    async def append_message(
        self,
        role: str,
        content: str,
        thread_uid: str = None,
        extract_memory: bool = False
    ) -> str:
        """Append message to conversation, optionally extracting to memory."""
        message_uid = await self._loom.append_async(role, content, thread_uid)

        if extract_memory and role == "assistant":
            # Extract observations from assistant responses
            await self._extract_observations(content, thread_uid)

        return message_uid

    async def recall_conversation(
        self,
        thread_uid: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Recall conversation messages."""
        return await self._loom.recall_async(thread_uid)

    async def search_conversations(
        self,
        query: str,
        thread_uid: str = None,
        limit: int = 5,
        include_all_threads: bool = False
    ) -> List[SearchResult]:
        """Semantic search across conversations."""
        results = await self._loom.semantic_search(
            query,
            thread_uid=thread_uid,
            limit=limit,
            include_all_threads=include_all_threads
        )
        return [
            SearchResult(
                source=MemorySource.CONVERSATION,
                content=r.get("content", ""),
                similarity=r.get("similarity", 0),
                ref=f"message:{r.get('message_uid')}",
                metadata={"role": r.get("role"), "thread_uid": thread_uid}
            )
            for r in results
        ]

    # ==========================================
    # Knowledge Layer (MemoryGate wrapper)
    # ==========================================

    async def store_observation(
        self,
        observation: str,
        confidence: float = 0.8,
        domain: str = None,
        evidence: List[str] = None
    ) -> Dict:
        """Store an observation in MemoryGate."""
        return MemoryGate.store_observation(
            observation,
            confidence=confidence,
            domain=domain,
            evidence=evidence
        )

    async def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.0,
        domain: str = None
    ) -> List[SearchResult]:
        """Search MemoryGate for relevant observations/patterns/concepts."""
        results = MemoryGate.search(query, limit=limit, domain=domain)
        return [
            SearchResult(
                source=MemorySource(r.get("source_type", "observation")),
                content=r.get("content", r.get("snippet", "")),
                similarity=r.get("similarity", 0),
                ref=r.get("ref", f"{r.get('source_type')}:{r.get('id')}"),
                metadata={
                    "confidence": r.get("confidence"),
                    "domain": r.get("domain")
                }
            )
            for r in (results or [])
        ]

    # ==========================================
    # Unified Operations
    # ==========================================

    async def unified_search(
        self,
        query: str,
        sources: List[MemorySource] = None,
        limit_per_source: int = 3
    ) -> List[SearchResult]:
        """
        Search across all memory sources.

        Args:
            query: Search query
            sources: Which sources to search (default: all)
            limit_per_source: Max results per source

        Returns:
            Combined results sorted by similarity
        """
        if sources is None:
            sources = list(MemorySource)

        results = []

        if MemorySource.CONVERSATION in sources:
            conv_results = await self.search_conversations(
                query, limit=limit_per_source, include_all_threads=True
            )
            results.extend(conv_results)

        knowledge_sources = [s for s in sources if s != MemorySource.CONVERSATION]
        if knowledge_sources:
            kg_results = await self.search_knowledge(query, limit=limit_per_source * 2)
            # Filter by requested source types
            kg_results = [r for r in kg_results if r.source in knowledge_sources]
            results.extend(kg_results[:limit_per_source * len(knowledge_sources)])

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    async def compose_context(
        self,
        user_input: str,
        thread_uid: str,
        max_tokens: int = 4096,
        include_knowledge: bool = True,
        knowledge_limit: int = 3
    ) -> List[Dict]:
        """
        Compose full prompt context combining conversation and knowledge.

        Returns message list ready for LLM.
        """
        # Get conversation context
        context = await self._loom.compose_prompt_context_async(
            user_input, thread_uid
        )

        # Inject relevant knowledge
        if include_knowledge:
            knowledge = await self.search_knowledge(user_input, limit=knowledge_limit)
            if knowledge:
                knowledge_text = self._format_knowledge_context(knowledge)
                # Insert after system prompt (index 1) or at start
                insert_pos = 1 if context and context[0].get("role") == "system" else 0
                context.insert(insert_pos, {
                    "role": "system",
                    "content": knowledge_text
                })

        return context

    # ==========================================
    # Thread Management
    # ==========================================

    def list_threads(self) -> List[Dict]:
        """List all conversation threads."""
        return self._loom.list_all_threads()

    def switch_thread(self, thread_uid: str) -> None:
        """Switch active thread."""
        self._loom.switch_to_thread(thread_uid)

    def get_active_thread(self) -> str:
        """Get active thread UID."""
        return self._loom.get_active_thread_uid()

    def clear_thread(self, thread_uid: str = None) -> None:
        """Clear thread messages."""
        self._loom.clear(thread_uid)

    # ==========================================
    # Memory Extraction
    # ==========================================

    async def _extract_observations(
        self,
        content: str,
        thread_uid: str = None
    ) -> List[str]:
        """Extract observations from content and store in MemoryGate."""
        from cathedral.MemoryGate.auto_memory import extract_observations

        observations = extract_observations(content)
        refs = []

        for obs in observations:
            result = await self.store_observation(
                observation=obs["text"],
                confidence=obs.get("confidence", 0.7),
                domain=obs.get("domain", "extracted"),
                evidence=[f"thread:{thread_uid}"] if thread_uid else None
            )
            if result:
                refs.append(f"observation:{result.get('id')}")

        return refs

    def _format_knowledge_context(self, results: List[SearchResult]) -> str:
        """Format knowledge results as context injection."""
        if not results:
            return ""

        lines = ["[Relevant Knowledge]"]
        for r in results:
            source_label = r.source.value.title()
            lines.append(f"- [{source_label}] {r.content[:200]}")

        return "\n".join(lines)


# Global instance
_memory: Optional[UnifiedMemory] = None


def get_memory() -> UnifiedMemory:
    """Get or create global UnifiedMemory instance."""
    global _memory
    if _memory is None:
        _memory = UnifiedMemory()
    return _memory


# Convenience exports
async def compose_context(user_input: str, thread_uid: str, **kwargs) -> List[Dict]:
    return await get_memory().compose_context(user_input, thread_uid, **kwargs)


async def append_message(role: str, content: str, thread_uid: str = None, **kwargs) -> str:
    return await get_memory().append_message(role, content, thread_uid, **kwargs)


async def search(query: str, **kwargs) -> List[SearchResult]:
    return await get_memory().unified_search(query, **kwargs)
```

### 3.2 Update Cathedral Main to Use UnifiedMemory

```python
# cathedral/__init__.py changes

# Old imports
# from loom import Loom
# loom = Loom()

# New imports
from cathedral.Memory import get_memory, compose_context, append_message

# In process_input_stream:
async def process_input_stream(user_input: str, thread_uid: str):
    # ...

    # Old: await loom.append_async("user", user_input, thread_uid=thread_uid)
    # New:
    memory = get_memory()
    await memory.append_message("user", user_input, thread_uid)

    # Old: full_history = await loom.compose_prompt_context_async(user_input, thread_uid)
    # New:
    full_history = await memory.compose_context(
        user_input,
        thread_uid,
        include_knowledge=True
    )

    # ... rest of function
```

### 3.3 Add Unified Search Command

```python
# In process_input_stream, add new command:

if lowered.startswith("/usearch "):
    # Unified search across all memory
    query = command[9:].strip()
    if not query:
        yield "usage: /usearch <query>"
        return

    memory = get_memory()
    results = await memory.unified_search(query, limit_per_source=3)

    if not results:
        yield "No results found."
        return

    yield f"Found {len(results)} results:\n\n"
    for r in results:
        source_icon = {
            "conversation": "[C]",
            "observation": "[O]",
            "pattern": "[P]",
            "concept": "[K]",
            "document": "[D]"
        }.get(r.source.value, "[?]")

        yield f"{source_icon} [{r.similarity:.2f}] {r.content[:100]}...\n"
        yield f"    ref: {r.ref}\n\n"
    return
```

---

## 4. Phase 2: Migration to MemoryGate Backend

### 4.1 Prerequisites

1. MemoryGate Loom integration spec implemented (see `LOOM_INTEGRATION_SPEC.md`)
2. New MCP tools available: `conversation_*`
3. Migration script tested

### 4.2 Update UnifiedMemory Implementation

Replace Loom calls with MemoryGate MCP calls:

```python
class UnifiedMemory:
    def __init__(self):
        # Remove: self._loom = Loom()
        # Add: Initialize MCP client to MemoryGate
        self._mg_client = MemoryGateMCPClient()

    async def create_thread(self, thread_name: str = None, metadata: Dict = None) -> str:
        # Old: return self._loom.create_new_thread(thread_name)
        # New:
        result = await self._mg_client.call("conversation_create_thread", {
            "thread_name": thread_name,
            "metadata": metadata
        })
        return result["thread_uid"]

    async def append_message(self, role: str, content: str, thread_uid: str = None, **kwargs) -> str:
        # Old: return await self._loom.append_async(role, content, thread_uid)
        # New:
        result = await self._mg_client.call("conversation_append", {
            "thread_uid": thread_uid,
            "role": role,
            "content": content,
            **kwargs
        })
        return result["message_uid"]

    # ... etc for all methods
```

### 4.3 Data Migration

```python
# scripts/migrate_loom_to_memorygate.py

async def migrate():
    """Migrate existing Loom data to MemoryGate."""

    # 1. Export from Loom
    loom = Loom()
    threads = loom.list_all_threads()

    for thread in threads:
        # 2. Create thread in MemoryGate
        mg_thread = await mg.conversation_create_thread(
            thread_name=thread["thread_name"]
        )

        # 3. Migrate messages
        messages = await loom.recall_async(thread["thread_uid"])
        for msg in messages:
            await mg.conversation_append(
                thread_uid=mg_thread["thread_uid"],
                role=msg["role"],
                content=msg["content"]
            )

        # 4. Migrate summaries
        # ... similar pattern

    print(f"Migrated {len(threads)} threads")
```

### 4.4 Configuration Switch

```python
# cathedral/Config/memory.py

class MemoryBackend(Enum):
    LOOM = "loom"           # Original (Phase 1)
    MEMORYGATE = "memorygate"  # New (Phase 2+)
    HYBRID = "hybrid"       # Both during migration

MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "loom")
```

---

## 5. Phase 3: Cleanup

### 5.1 Remove Loom Module

```bash
# After migration complete and stable
rm -rf cathedral-v2/loom/
```

### 5.2 Update Dependencies

```diff
# requirements.txt
- # Loom dependencies (now in MemoryGate)
- asyncpg  # Now via MemoryGate
```

### 5.3 Simplify UnifiedMemory

Remove dual-backend support, keep only MemoryGate:

```python
class UnifiedMemory:
    """Unified memory backed by MemoryGate."""

    def __init__(self):
        self._client = MemoryGateMCPClient()

    # Clean single-backend implementation
```

---

## 6. File Changes Summary

### Phase 1 (New Files)

```
cathedral/
├── Memory/
│   ├── __init__.py        # UnifiedMemory class
│   └── types.py           # SearchResult, MemorySource
```

### Phase 1 (Modified Files)

```
cathedral/
├── __init__.py            # Use UnifiedMemory instead of direct Loom
```

### Phase 2 (Modified Files)

```
cathedral/
├── Memory/
│   └── __init__.py        # Switch to MemoryGate backend
├── Config/
│   └── memory.py          # Backend configuration (new)
```

### Phase 3 (Removed)

```
loom/                      # Entire module removed
├── __init__.py
├── models.py
├── db.py
├── embeddings.py
├── LoomMirror/
├── VectorGate/
└── CodexGate/
```

---

## 7. Commands Reference

### Current Commands (Loom)

| Command | Description | Phase 1 | Phase 2+ |
|---------|-------------|---------|----------|
| `/history` | Show thread history | Keep (wrapped) | MG backend |
| `/forget` | Clear thread | Keep (wrapped) | MG backend |
| `/loomsearch` | Semantic search | Keep (wrapped) | Rename to `/msearch` |
| `/backfill` | Generate embeddings | Keep (wrapped) | MG backend |

### New Commands

| Command | Description | Phase |
|---------|-------------|-------|
| `/usearch <query>` | Unified cross-memory search | 1 |
| `/memory` | Show memory stats (both systems) | 1 |
| `/extract` | Force memory extraction from thread | 2 |

### Deprecated Commands

| Command | Replacement | Phase |
|---------|-------------|-------|
| `/search` (MemoryGate) | `/usearch` | 2 |
| `/loomsearch` | `/usearch` | 2 |

---

## 8. Testing Strategy

### Phase 1 Tests

```python
# tests/test_unified_memory.py

async def test_unified_memory_wraps_loom():
    """UnifiedMemory should delegate to Loom correctly."""
    memory = UnifiedMemory()

    # Create thread
    thread_uid = await memory.create_thread("Test")
    assert thread_uid is not None

    # Append message
    msg_uid = await memory.append_message("user", "Hello", thread_uid)
    assert msg_uid is not None

    # Recall
    messages = await memory.recall_conversation(thread_uid)
    assert len(messages) == 1

async def test_unified_search_combines_sources():
    """Unified search should return results from both Loom and MemoryGate."""
    memory = UnifiedMemory()

    # Setup: add data to both systems
    await memory.append_message("user", "Python asyncio patterns", thread_uid)
    await memory.store_observation("asyncio uses event loops")

    # Search should find both
    results = await memory.unified_search("asyncio")
    sources = {r.source for r in results}

    assert MemorySource.CONVERSATION in sources
    assert MemorySource.OBSERVATION in sources
```

### Phase 2 Tests

```python
async def test_memorygate_backend():
    """Memory operations should work with MemoryGate backend."""
    # Similar tests but verify data in MemoryGate
    pass

async def test_migration():
    """Migration should preserve all data."""
    pass
```

---

## 9. Rollback Plan

### Phase 1 Rollback

```python
# Simply revert to direct Loom usage
# UnifiedMemory is additive, no data loss
```

### Phase 2 Rollback

```python
# Keep Loom running in parallel
MEMORY_BACKEND = "loom"  # Switch back

# Data in MemoryGate is preserved for future migration
```

---

## 10. Success Criteria

### Phase 1

- [ ] UnifiedMemory class implemented
- [ ] All existing functionality works through UnifiedMemory
- [ ] `/usearch` command works
- [ ] No performance regression

### Phase 2

- [ ] MemoryGate backend fully functional
- [ ] Migration script works
- [ ] Existing threads migrated successfully
- [ ] Cross-pollination (conversation ↔ knowledge) working

### Phase 3

- [ ] Loom module removed
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No orphaned dependencies

---

## 11. Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1 | 2-3 days | None |
| Phase 2 | 5-7 days | MemoryGate Loom spec implemented |
| Phase 3 | 1-2 days | Phase 2 stable for 1 week |

---

## 12. Open Questions

1. **Session identity**: Should Cathedral's thread_uid map to MemoryGate's session_id or create a new conversation concept?

2. **Embedding source**: Continue using OpenAI directly or route through MemoryGate's embedding service?

3. **Real-time sync**: Should MemoryGate observations be immediately available in conversation context, or use eventual consistency?

4. **Multi-tenant**: Cathedral is single-tenant; how to handle if MemoryGate is multi-tenant?

---

## Appendix A: Current Integration Points

### Files that import Loom

```
cathedral/__init__.py          # Main chat processing
cathedral/MetadataChannel/     # Metadata providers
altar/run.py                   # Web UI endpoints
```

### Loom methods used in Cathedral

```python
loom.create_new_thread()
loom.switch_to_thread()
loom.list_all_threads()
loom.append_async()
loom.recall_async()
loom.clear()
loom.compose_prompt_context_async()
loom.semantic_search()
loom.backfill_embeddings()
```

---

## Appendix B: Related Documents

- `F:/hexylab/lv_stack/MemoryGate/docs/LOOM_INTEGRATION_SPEC.md` - Spec for adding Loom to MemoryGate standalone
- `F:/hexylab/Cathedral-v2/loom/__init__.py` - Current Loom implementation
- `F:/hexylab/lv_stack/MemoryGate/core/models.py` - MemoryGate data models
