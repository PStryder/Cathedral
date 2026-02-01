from __future__ import annotations

from typing import AsyncGenerator, Optional
import json

from cathedral.commands.context import CommandContext
from cathedral.Memory import MemorySource
from cathedral.MemoryGate.auto_memory import format_search_results, format_recall_results
from cathedral.StarMirror import reflect_vision_stream, reflect_audio, transcribe_audio, ImageDetail

from cathedral import (
    MemoryGate,
    MetadataChannel,
    SubAgentGate,
    PersonalityGate,
    SecurityManager,
    FileSystemGate,
    ShellGate,
    BrowserGate,
    ScriptureGate,
)


async def _emit(ctx: CommandContext, event_type: str, message: str, **kwargs) -> None:
    await ctx.services.emit_event(event_type, message, **kwargs)


async def _agent_update(ctx: CommandContext, agent_id: str, message: str, status: str = "running") -> None:
    await ctx.services.record_agent_update(agent_id, message, status)


async def emit_completed_agents(ctx: CommandContext) -> AsyncGenerator[str, None]:
    """Yield notifications for completed sub-agents (if any)."""
    completed = SubAgentGate.check_completed()
    for agent_id in completed:
        agent_status = SubAgentGate.status(agent_id)
        if agent_status:
            status_str = agent_status.get("status", "?")
            yield f"[SubAgent {agent_id} {status_str}]\n"


async def handle_pre_command(
    command: str, lowered: str, ctx: CommandContext
) -> Optional[AsyncGenerator[str, None]]:
    """Handle commands that occur before sub-agent completion checks."""
    if lowered == "/history":
        async def _gen() -> AsyncGenerator[str, None]:
            history = await ctx.loom.recall_async(ctx.thread_uid)
            if not history:
                yield "no thread history yet"
                return
            history_text = "\n".join(f"{entry['role']}: {entry['content']}" for entry in history)
            yield history_text
        return _gen()

    if lowered == "/forget":
        async def _gen() -> AsyncGenerator[str, None]:
            ctx.loom.clear(ctx.thread_uid)
            yield "the memory fades—forgotten by the Cathedral"
        return _gen()

    if lowered.startswith("/export thread"):
        async def _gen() -> AsyncGenerator[str, None]:
            parts = command.split()
            if len(parts) >= 3:
                name = parts[2]
                history = await ctx.loom.recall_async(ctx.thread_uid)
                ScriptureGate.export_thread(history, name)
                yield f"thread exported to scripture as '{name}.thread.json'"
            else:
                yield "usage: /export thread <name>"
        return _gen()

    if lowered.startswith("/import bios"):
        async def _gen() -> AsyncGenerator[str, None]:
            parts = command.split("/import bios", 1)
            path = parts[1].strip()
            try:
                bios_text = ScriptureGate.import_bios(path)
                yield str(bios_text)
            except Exception as e:
                yield f"error loading bios: {e}"
        return _gen()

    if lowered.startswith("/import glyph"):
        async def _gen() -> AsyncGenerator[str, None]:
            parts = command.split("/import glyph", 1)
            path = parts[1].strip()
            try:
                glyph_data = ScriptureGate.import_glyph(path)
                yield str(glyph_data)
            except Exception as e:
                yield f"error loading glyph: {e}"
        return _gen()

    # === Memory Search (unified across all sources) ===

    if lowered.startswith("/search "):
        async def _gen() -> AsyncGenerator[str, None]:
            query = command[8:].strip()
            if not query:
                yield "usage: /search <query>"
                return
            await _emit(ctx, "memory", f"Searching: {query[:50]}...")

            # Use unified search across all sources (includes all conversation threads)
            results = await ctx.memory.unified_search(
                query,
                sources=None,  # All sources
                limit_per_source=3,
                min_similarity=0.3
            )

            if not results:
                yield "No results found across any memory source."
                return

            await _emit(ctx, "memory", f"Found {len(results)} results")
            yield f"Found {len(results)} results:\n\n"

            # Group by source type for display
            for r in results:
                source_icon = {
                    MemorySource.CONVERSATION: "[C]",
                    MemorySource.SUMMARY: "[S]",
                    MemorySource.OBSERVATION: "[O]",
                    MemorySource.PATTERN: "[P]",
                    MemorySource.CONCEPT: "[K]",
                    MemorySource.DOCUMENT: "[D]"
                }.get(r.source, "[?]")

                # Truncate content for display
                content_preview = r.content[:120]
                if len(r.content) > 120:
                    content_preview += "..."

                confidence_str = f" ({r.confidence:.0%})" if r.confidence else ""
                yield f"{source_icon} [{r.similarity:.2f}]{confidence_str} {content_preview}\n"
                yield f"    ref: {r.ref}\n\n"
        return _gen()

    # === Unified Search (alias for /search, kept for backwards compatibility) ===

    if lowered.startswith("/usearch "):
        async def _gen() -> AsyncGenerator[str, None]:
            query = command[9:].strip()
            if not query:
                yield "usage: /usearch <query> - Search across all memory sources"
                return
            await _emit(ctx, "memory", f"Unified search: {query[:50]}...")

            results = await ctx.memory.unified_search(
                query,
                sources=None,  # All sources
                limit_per_source=3,
                min_similarity=0.3
            )

            if not results:
                yield "No results found across any memory source."
                return

            yield f"Found {len(results)} results across all memory sources:\n\n"

            # Group by source type for display
            for r in results:
                source_icon = {
                    MemorySource.CONVERSATION: "[C]",
                    MemorySource.SUMMARY: "[S]",
                    MemorySource.OBSERVATION: "[O]",
                    MemorySource.PATTERN: "[P]",
                    MemorySource.CONCEPT: "[K]",
                    MemorySource.DOCUMENT: "[D]"
                }.get(r.source, "[?]")

                # Truncate content for display
                content_preview = r.content[:120]
                if len(r.content) > 120:
                    content_preview += "..."

                confidence_str = f" ({r.confidence:.0%})" if r.confidence else ""
                yield f"{source_icon} [{r.similarity:.2f}]{confidence_str} {content_preview}\n"
                yield f"    ref: {r.ref}\n\n"

            await _emit(ctx, "memory", f"Found {len(results)} unified results")
        return _gen()

    if lowered == "/memory" or lowered == "/memstat":
        async def _gen() -> AsyncGenerator[str, None]:
            stats = await ctx.memory.get_stats()
            yield "Memory Statistics:\n\n"
            yield "Conversation:\n"
            yield f"  Threads: {stats.thread_count}\n"
            yield f"  Available: {'Yes' if stats.conversation_available else 'No'}\n\n"
            yield "Knowledge (MemoryGate):\n"
            yield f"  Observations: {stats.observation_count}\n"
            yield f"  Patterns: {stats.pattern_count}\n"
            yield f"  Concepts: {stats.concept_count}\n"
            yield f"  Documents: {stats.document_count}\n"
            yield f"  Available: {'Yes' if stats.memorygate_available else 'No'}\n"
        return _gen()

    if lowered.startswith("/remember "):
        async def _gen() -> AsyncGenerator[str, None]:
            text = command[10:].strip()
            if not text:
                yield "usage: /remember <observation>"
                return
            await _emit(ctx, "memory", "Storing observation...")
            result = MemoryGate.store_observation(text, confidence=0.9, domain="explicit")
            if result:
                await _emit(ctx, "memory", f"Stored: obs:{result.get('id', '?')}")
                yield f"Stored observation:{result.get('id', '?')}"
            else:
                await _emit(ctx, "memory", "Store failed")
                yield "Failed to store observation (MemoryGate may not be configured)"
        return _gen()

    if lowered == "/memories" or lowered.startswith("/memories "):
        async def _gen() -> AsyncGenerator[str, None]:
            parts = command.split(maxsplit=1)
            domain = parts[1].strip() if len(parts) > 1 else None
            results = MemoryGate.recall(domain=domain, limit=10)
            yield format_recall_results(results)
        return _gen()

    if lowered.startswith("/concept "):
        async def _gen() -> AsyncGenerator[str, None]:
            name = command[9:].strip()
            if not name:
                yield "usage: /concept <name>"
                return
            result = MemoryGate.get_concept(name)
            if result and result.get("status") == "found":
                c = result
                yield f"Concept: {c.get('name', name)}\n"
                yield f"Type: {c.get('concept_type', '?')}\n"
                yield f"Description: {c.get('description', 'none')}\n"
                if c.get('domain'):
                    yield f"Domain: {c.get('domain')}\n"
            else:
                yield f"Concept '{name}' not found"
        return _gen()

    if lowered.startswith("/pattern "):
        async def _gen() -> AsyncGenerator[str, None]:
            parts = command[9:].strip().split("/", 1)
            if len(parts) != 2:
                yield "usage: /pattern <category>/<name>"
                return
            category, name = parts[0].strip(), parts[1].strip()
            result = MemoryGate.get_pattern(category, name)
            if result and result.get("status") == "found":
                p = result
                yield f"Pattern: {p.get('category', category)}/{p.get('pattern_name', name)}\n"
                yield f"Confidence: {p.get('confidence', 0):.2f}\n"
                yield f"Text: {p.get('pattern_text', 'none')}\n"
            else:
                yield f"Pattern '{category}/{name}' not found"
        return _gen()

    if lowered == "/memstats":
        async def _gen() -> AsyncGenerator[str, None]:
            stats = MemoryGate.get_stats()
            if stats:
                yield "Memory Stats:\n"
                counts = stats.get("counts", {})
                for k, v in counts.items():
                    yield f"  {k}: {v}\n"
            else:
                yield "MemoryGate not configured or unavailable"
        return _gen()

    # === Knowledge Discovery Commands ===

    if lowered.startswith("/discover "):
        async def _gen() -> AsyncGenerator[str, None]:
            ref = command[10:].strip()
            if not ref or ":" not in ref:
                yield "usage: /discover <type:id> (e.g., message:abc123, thread:xyz789, concept:42)"
                return
            await _emit(ctx, "memory", f"Discovering relationships for: {ref}")
            try:
                from cathedral.MemoryGate.discovery import discover_for_ref
                relationships = await discover_for_ref(ref)
                if not relationships:
                    yield f"No relationships discovered for {ref}"
                    return
                yield f"Discovered {len(relationships)} relationships for {ref}:\n\n"
                for rel in relationships:
                    yield f"  [{rel.similarity:.3f}] {rel.rel_type} -> {rel.to_ref}\n"
                await _emit(ctx, "memory", f"Discovered {len(relationships)} relationships")
            except Exception as e:
                yield f"Discovery error: {e}"
        return _gen()

    if lowered.startswith("/related "):
        async def _gen() -> AsyncGenerator[str, None]:
            ref = command[9:].strip()
            if not ref or ":" not in ref:
                yield "usage: /related <type:id> (e.g., message:abc123, concept:42)"
                return
            results = MemoryGate.related(ref, limit=10)
            if not results or not results.get("relationships"):
                yield f"No relationships found for {ref}"
                return
            rels = results["relationships"]
            yield f"Relationships for {ref} ({len(rels)}):\n\n"
            for r in rels:
                direction = "->" if r.get("direction") == "outgoing" else "<-"
                target = r.get("to_ref") if direction == "->" else r.get("from_ref")
                weight = r.get("weight", 0)
                rel_type = r.get("rel_type", "?")
                desc = r.get("description", "")
                if "discovered" in desc.lower():
                    tag = " [D]"
                else:
                    tag = ""
                yield f"  [{weight:.2f}]{tag} {rel_type} {direction} {target}\n"
        return _gen()

    if lowered == "/discovery":
        async def _gen() -> AsyncGenerator[str, None]:
            try:
                from cathedral.MemoryGate.discovery import get_discovery_service
                svc = get_discovery_service()
                yield "Discovery Service Status:\n"
                yield f"  Running: {svc._running}\n"
                yield f"  Enabled: {svc.config.enabled}\n"
                yield f"  Min similarity: {svc.config.min_similarity}\n"
                yield f"  Top K: {svc.config.top_k}\n"
                yield f"  Thread interval: {svc.config.thread_embedding_interval} messages\n"
                yield f"  Queue size: ~{svc._queue.qsize()}\n"
            except Exception as e:
                yield f"Discovery service not available: {e}"
        return _gen()

    # === Conversation Semantic Search ===

    if lowered.startswith("/loomsearch "):
        async def _gen() -> AsyncGenerator[str, None]:
            query = command[12:].strip()
            if not query:
                yield "usage: /loomsearch <query>"
                return
            results = await ctx.loom.semantic_search(query, thread_uid=ctx.thread_uid, limit=5, include_all_threads=True)
            if not results:
                yield "No semantically similar messages found."
                return
            yield "Semantically similar messages:\n"
            for r in results:
                sim = r.get("similarity", 0)
                role = r.get("role", "?")
                content = r.get("content", "")[:150]
                yield f"  [{sim:.2f}] {role}: {content}...\n"
        return _gen()

    if lowered == "/backfill":
        async def _gen() -> AsyncGenerator[str, None]:
            yield "Generating embeddings for messages without them...\n"
            count = await ctx.loom.backfill_embeddings(batch_size=50)
            yield f"Generated embeddings for {count} messages."
        return _gen()

    # === MetadataChannel Commands ===

    if lowered.startswith("/meta"):
        async def _gen() -> AsyncGenerator[str, None]:
            # Parse: /meta <target> [field1,field2,...]
            parts = command[5:].strip().split(maxsplit=1)
            if not parts or parts[0] == "":
                # Default to current state
                target = "current"
                fields = None
            else:
                target = parts[0]
                fields = None
                if len(parts) > 1:
                    fields = [f.strip() for f in parts[1].split(",")]

            context = {"loom": ctx.loom, "thread_uid": ctx.thread_uid}
            result = MetadataChannel.query(target, fields, context)
            yield json.dumps(result, separators=(",", ":"))
        return _gen()

    if lowered == "/metafields":
        async def _gen() -> AsyncGenerator[str, None]:
            channel = MetadataChannel.get_channel()
            yield f"providers: {','.join(channel.list_providers())}\n"
            yield f"fields: {','.join(channel.list_fields())}"
        return _gen()

    # === Multi-Modal Commands ===

    if lowered.startswith("/image "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /image <path> <prompt>
            parts = command[7:].strip().split(maxsplit=1)
            if len(parts) < 2:
                yield "usage: /image <path> <prompt>"
                return
            image_path, prompt = parts[0], parts[1]
            try:
                # Store user intent
                await ctx.loom.append_async("user", f"[Image: {image_path}] {prompt}", thread_uid=ctx.thread_uid)

                full_response = ""
                async for token in reflect_vision_stream(prompt, [image_path]):
                    full_response += token
                    yield token

                await ctx.loom.append_async("assistant", full_response, thread_uid=ctx.thread_uid)
            except FileNotFoundError:
                yield f"Image not found: {image_path}"
            except Exception as e:
                yield f"Error processing image: {e}"
        return _gen()

    if lowered.startswith("/describe "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /describe <path> [detail:low|high|auto]
            parts = command[10:].strip().split()
            if not parts:
                yield "usage: /describe <path> [detail:low|high|auto]"
                return
            image_path = parts[0]
            detail = ImageDetail.AUTO
            if len(parts) > 1 and parts[1].startswith("detail:"):
                detail_str = parts[1][7:].lower()
                detail = {"low": ImageDetail.LOW, "high": ImageDetail.HIGH}.get(detail_str, ImageDetail.AUTO)

            try:
                await ctx.loom.append_async("user", f"[Describe image: {image_path}]", thread_uid=ctx.thread_uid)

                full_response = ""
                async for token in reflect_vision_stream(
                    "Describe this image in detail.",
                    [image_path],
                    detail=detail
                ):
                    full_response += token
                    yield token

                await ctx.loom.append_async("assistant", full_response, thread_uid=ctx.thread_uid)
            except FileNotFoundError:
                yield f"Image not found: {image_path}"
            except Exception as e:
                yield f"Error: {e}"
        return _gen()

    if lowered.startswith("/compare "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /compare <path1> <path2> [prompt]
            parts = command[9:].strip().split(maxsplit=2)
            if len(parts) < 2:
                yield "usage: /compare <path1> <path2> [prompt]"
                return
            img1, img2 = parts[0], parts[1]
            prompt = parts[2] if len(parts) > 2 else "Compare these images and describe the differences."

            try:
                await ctx.loom.append_async("user", f"[Compare: {img1} vs {img2}] {prompt}", thread_uid=ctx.thread_uid)

                full_response = ""
                async for token in reflect_vision_stream(prompt, [img1, img2]):
                    full_response += token
                    yield token

                await ctx.loom.append_async("assistant", full_response, thread_uid=ctx.thread_uid)
            except FileNotFoundError as e:
                yield f"Image not found: {e}"
            except Exception as e:
                yield f"Error: {e}"
        return _gen()

    if lowered.startswith("/transcribe "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /transcribe <audio_path>
            audio_path = command[12:].strip()
            if not audio_path:
                yield "usage: /transcribe <audio_path>"
                return
            try:
                yield "Transcribing audio...\n"
                transcription = await transcribe_audio(audio_path)
                yield f"Transcription:\n{transcription}"
                await ctx.loom.append_async("user", f"[Transcribed audio: {audio_path}]\n{transcription}", thread_uid=ctx.thread_uid)
            except FileNotFoundError:
                yield f"Audio file not found: {audio_path}"
            except Exception as e:
                yield f"Error transcribing: {e}"
        return _gen()

    if lowered.startswith("/audio "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /audio <audio_path> <prompt>
            parts = command[7:].strip().split(maxsplit=1)
            if len(parts) < 2:
                yield "usage: /audio <audio_path> <prompt>"
                return
            audio_path, prompt = parts[0], parts[1]
            try:
                yield "Processing audio...\n"
                response = await reflect_audio(prompt, audio_path)
                yield response
                await ctx.loom.append_async("user", f"[Audio: {audio_path}] {prompt}", thread_uid=ctx.thread_uid)
                await ctx.loom.append_async("assistant", response, thread_uid=ctx.thread_uid)
            except FileNotFoundError:
                yield f"Audio file not found: {audio_path}"
            except Exception as e:
                yield f"Error: {e}"
        return _gen()

    return None


async def handle_post_command(
    command: str, lowered: str, ctx: CommandContext
) -> Optional[AsyncGenerator[str, None]]:
    """Handle commands that occur after sub-agent completion checks."""
    # === SubAgent Commands ===

    if lowered.startswith("/spawn "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /spawn <task description>
            task = command[7:].strip()
            if not task:
                yield "usage: /spawn <task description>"
                return
            try:
                await _emit(ctx, "agent", f"Spawning agent for: {task[:50]}...")
                agent_id = SubAgentGate.spawn(
                    task=task,
                    context={"thread_uid": ctx.thread_uid}
                )
                await _agent_update(ctx, agent_id, f"Started: {task[:40]}...", "running")
                yield f"Spawned sub-agent {agent_id}\n"
                yield f"Task: {task[:100]}{'...' if len(task) > 100 else ''}\n"
                yield "Use /agents to check status, /result <id> to get result"
            except Exception as e:
                await _emit(ctx, "agent", f"Spawn failed: {e}")
                yield f"Failed to spawn agent: {e}"
        return _gen()

    if lowered == "/agents":
        async def _gen() -> AsyncGenerator[str, None]:
            agents = SubAgentGate.list_agents()
            if not agents:
                yield "No sub-agents"
                return
            yield f"Sub-agents ({len(agents)}):\n"
            for a in agents:
                status_icon = {"running": "~", "completed": "+", "failed": "!", "cancelled": "x"}.get(a["status"], "?")
                task_short = a["task"][:40] + "..." if len(a["task"]) > 40 else a["task"]
                yield f"  [{status_icon}] {a['id']} | {task_short}\n"
        return _gen()

    if lowered.startswith("/agent "):
        async def _gen() -> AsyncGenerator[str, None]:
            agent_id = command[7:].strip()
            if not agent_id:
                yield "usage: /agent <id>"
                return
            agent_status = SubAgentGate.status(agent_id)
            if not agent_status:
                yield f"Agent {agent_id} not found"
                return
            yield json.dumps(agent_status, indent=2)
        return _gen()

    if lowered.startswith("/result "):
        async def _gen() -> AsyncGenerator[str, None]:
            agent_id = command[8:].strip()
            if not agent_id:
                yield "usage: /result <id>"
                return
            result = SubAgentGate.result(agent_id)
            if result is None:
                status = SubAgentGate.status(agent_id)
                if status:
                    yield f"Agent {agent_id} status: {status.get('status', '?')}"
                    if status.get("error"):
                        yield f"\nError: {status['error']}"
                else:
                    yield f"Agent {agent_id} not found"
                return
            yield f"[Result from {agent_id}]\n{result}"
        return _gen()

    if lowered.startswith("/cancel "):
        async def _gen() -> AsyncGenerator[str, None]:
            agent_id = command[8:].strip()
            if not agent_id:
                yield "usage: /cancel <id>"
                return
            if SubAgentGate.cancel(agent_id):
                yield f"Cancelled agent {agent_id}"
            else:
                yield f"Could not cancel agent {agent_id} (not running or not found)"
        return _gen()

    # === ScriptureGate Commands ===

    if lowered.startswith("/store "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /store <path> [title]
            parts = command[7:].strip().split(maxsplit=1)
            if not parts:
                yield "usage: /store <path> [title]"
                return
            file_path = parts[0]
            title = parts[1] if len(parts) > 1 else None
            try:
                result = await ScriptureGate.store(
                    source=file_path,
                    title=title,
                    source_type="upload"
                )
                yield f"Stored: {result.get('ref', '?')}\n"
                yield f"Type: {result.get('file_type', '?')}\n"
                yield "Indexing in background..."
            except FileNotFoundError:
                yield f"File not found: {file_path}"
            except Exception as e:
                yield f"Error storing: {e}"
        return _gen()

    if lowered.startswith("/scripture "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /scripture <ref_or_uid>
            ref = command[11:].strip()
            if not ref:
                yield "usage: /scripture <ref_or_uid>"
                return
            # Try as ref first, then as uid
            result = ScriptureGate.search_by_ref(ref) if ":" in ref else ScriptureGate.get(ref)
            if result:
                yield json.dumps(result, indent=2)
            else:
                yield f"Scripture not found: {ref}"
        return _gen()

    if lowered.startswith("/scriptsearch "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /scriptsearch <query> [type:document|image|audio]
            parts = command[14:].strip().split()
            if not parts:
                yield "usage: /scriptsearch <query> [type:document|image|audio]"
                return
            # Check for type filter
            file_type = None
            query_parts = []
            for p in parts:
                if p.startswith("type:"):
                    file_type = p[5:]
                else:
                    query_parts.append(p)
            query = " ".join(query_parts)
            if not query:
                yield "usage: /scriptsearch <query>"
                return
            results = await ScriptureGate.search(query, limit=5, file_type=file_type)
            if not results:
                yield "No matching scriptures found."
                return
            yield f"Found {len(results)} results:\n"
            for r in results:
                sim = r.get("similarity", 0)
                ref = r.get("ref", "?")
                title = r.get("title", "Untitled")
                yield f"  [{sim:.2f}] {ref} - {title}\n"
        return _gen()

    if lowered == "/scriptures" or lowered.startswith("/scriptures "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /scriptures [type]
            parts = command.split(maxsplit=1)
            file_type = parts[1].strip() if len(parts) > 1 else None
            results = ScriptureGate.list_scriptures(file_type=file_type, limit=10)
            if not results:
                yield "No scriptures found."
                return
            yield f"Scriptures ({len(results)}):\n"
            for r in results:
                ref = r.get("ref", "?")
                title = r.get("title", r.get("file_path", "?").split("/")[-1])
                indexed = "[+]" if r.get("is_indexed") else "[.]"
                yield f"  {indexed} {ref} - {title}\n"
        return _gen()

    if lowered == "/scriptstats":
        async def _gen() -> AsyncGenerator[str, None]:
            stats = ScriptureGate.stats()
            yield "Scripture Stats:\n"
            yield f"  Total: {stats.get('total', 0)}\n"
            yield f"  Indexed: {stats.get('indexed', 0)}\n"
            yield f"  Size: {stats.get('total_size_mb', 0)} MB\n"
            by_type = stats.get("by_type", {})
            if by_type:
                yield "  By type:\n"
                for t, count in by_type.items():
                    yield f"    {t}: {count}\n"
        return _gen()

    if lowered == "/scriptindex":
        async def _gen() -> AsyncGenerator[str, None]:
            yield "Indexing unindexed scriptures...\n"
            count = await ScriptureGate.backfill_index(batch_size=10)
            yield f"Indexed {count} scriptures."
        return _gen()

    # === Personality Commands ===

    if lowered == "/personalities":
        async def _gen() -> AsyncGenerator[str, None]:
            PersonalityGate.initialize()
            personalities = PersonalityGate.list_all()
            if not personalities:
                yield "No personalities available."
                return
            yield f"Personalities ({len(personalities)}):\n"
            for p in personalities:
                default_marker = "*" if p.get("is_default") else " "
                builtin_marker = "[B]" if p.get("is_builtin") else "   "
                yield f"  {default_marker} {builtin_marker} {p['id']} - {p['name']}\n"
                yield f"       {p['description'][:60]}{'...' if len(p.get('description', '')) > 60 else ''}\n"
        return _gen()

    if lowered.startswith("/personality ") and not lowered.startswith("/personality-"):
        async def _gen() -> AsyncGenerator[str, None]:
            # /personality <id> - Switch personality for this thread
            personality_id = command[13:].strip()
            if not personality_id:
                yield "usage: /personality <id>"
                return
            PersonalityGate.initialize()
            personality = PersonalityGate.load(personality_id)
            if not personality:
                yield f"Personality '{personality_id}' not found"
                return
            ctx.thread_personalities[ctx.thread_uid] = personality_id
            PersonalityGate.PersonalityManager.record_usage(personality_id)
            await _emit(ctx, "system", f"Personality: {personality.name} ({personality.llm_config.model})")
            yield f"Switched to personality: {personality.name}\n"
            yield f"Model: {personality.llm_config.model}\n"
            yield f"Temperature: {personality.llm_config.temperature}"
        return _gen()

    if lowered == "/personality":
        async def _gen() -> AsyncGenerator[str, None]:
            # Show current personality for this thread
            current_id = ctx.thread_personalities.get(ctx.thread_uid, "default")
            PersonalityGate.initialize()
            personality = PersonalityGate.load(current_id)
            if personality:
                yield f"Current personality: {personality.name} ({personality.id})\n"
                yield f"Model: {personality.llm_config.model}\n"
                yield f"Temperature: {personality.llm_config.temperature}\n"
                yield f"Style: {', '.join(personality.behavior.style_tags)}"
            else:
                yield "Using default personality"
        return _gen()

    if lowered.startswith("/personality-info "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /personality-info <id> - Show full details
            personality_id = command[18:].strip()
            if not personality_id:
                yield "usage: /personality-info <id>"
                return
            PersonalityGate.initialize()
            personality = PersonalityGate.load(personality_id)
            if not personality:
                yield f"Personality '{personality_id}' not found"
                return
            yield json.dumps(personality.to_dict(), indent=2, default=str)
        return _gen()

    if lowered.startswith("/personality-create "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /personality-create <name>
            name = command[20:].strip()
            if not name:
                yield "usage: /personality-create <name>"
                return
            PersonalityGate.initialize()
            try:
                personality = PersonalityGate.create(name=name)
                yield f"Created personality: {personality.id}\n"
                yield f"Edit at /personalities in the web UI or use /personality-info {personality.id}"
            except Exception as e:
                yield f"Error creating personality: {e}"
        return _gen()

    if lowered.startswith("/personality-delete "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /personality-delete <id>
            personality_id = command[20:].strip()
            if not personality_id:
                yield "usage: /personality-delete <id>"
                return
            PersonalityGate.initialize()
            try:
                if PersonalityGate.PersonalityManager.delete(personality_id):
                    yield f"Deleted personality: {personality_id}"
                else:
                    yield f"Personality '{personality_id}' not found"
            except ValueError as e:
                yield str(e)  # e.g., "Cannot delete builtin personalities"
        return _gen()

    if lowered.startswith("/personality-export "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /personality-export <id>
            personality_id = command[20:].strip()
            if not personality_id:
                yield "usage: /personality-export <id>"
                return
            PersonalityGate.initialize()
            personality = PersonalityGate.load(personality_id)
            if not personality:
                yield f"Personality '{personality_id}' not found"
                return
            # Output as JSON for copy-paste
            export_data = personality.to_dict()
            export_data["metadata"]["usage_count"] = 0
            yield json.dumps(export_data, indent=2, default=str)
        return _gen()

    if lowered.startswith("/personality-copy "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /personality-copy <id> <new_name>
            parts = command[18:].strip().split(maxsplit=1)
            if len(parts) < 2:
                yield "usage: /personality-copy <id> <new_name>"
                return
            personality_id, new_name = parts[0], parts[1]
            PersonalityGate.initialize()
            try:
                new_personality = PersonalityGate.PersonalityManager.duplicate(personality_id, new_name)
                if new_personality:
                    yield f"Created copy: {new_personality.id} ({new_personality.name})"
                else:
                    yield f"Personality '{personality_id}' not found"
            except Exception as e:
                yield f"Error: {e}"
        return _gen()

    # === FileSystemGate Commands ===

    if lowered == "/sources":
        async def _gen() -> AsyncGenerator[str, None]:
            FileSystemGate.initialize()
            folders = FileSystemGate.list_folders()
            if not folders:
                yield "No folders configured. Use /sources-add to add one."
                return
            yield f"Configured folders ({len(folders)}):\n"
            for f in folders:
                perm = f.get("permission", "?")
                perm_icon = {"read_only": "R", "read_write": "RW", "no_access": "-"}
                yield f"  [{perm_icon.get(perm, '?')}] {f['id']}: {f['name']}\n"
                yield f"      {f['path']}\n"
        return _gen()

    if lowered.startswith("/sources-add "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /sources-add <path> <name> [permission]
            parts = command[13:].strip().split(maxsplit=2)
            if len(parts) < 2:
                yield "usage: /sources-add <path> <name> [read_only|read_write]"
                return
            path = parts[0]
            name = parts[1]
            permission = parts[2] if len(parts) > 2 else "read_only"
            # Generate folder ID from name
            import re
            folder_id = re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))[:20]
            FileSystemGate.initialize()
            success, message = FileSystemGate.add_folder(folder_id, path, name, permission)
            if success:
                await _emit(ctx, "filesystem", message)
                yield message
            else:
                yield f"Error: {message}"
        return _gen()

    if lowered.startswith("/ls "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /ls [folder:]<path>
            path_spec = command[4:].strip()
            if not path_spec:
                yield "usage: /ls [folder:]<path>"
                return
            FileSystemGate.initialize()
            if ":" in path_spec:
                folder_id, rel_path = path_spec.split(":", 1)
            else:
                # Use first configured folder
                folders = FileSystemGate.list_folders()
                if not folders:
                    yield "No folders configured. Use /sources-add first."
                    return
                folder_id = folders[0]["id"]
                rel_path = path_spec
            result = FileSystemGate.list_dir(folder_id, rel_path)
            if not result.success:
                yield f"Error: {result.error}"
                return
            files = result.data or []
            if not files:
                yield "Directory is empty."
                return
            yield f"Contents of {result.path}:\n"
            for f in files:
                icon = "/" if f.get("is_directory") else ""
                size = f"{f.get('size_bytes', 0):,}" if not f.get("is_directory") else "-"
                yield f"  {f['name']}{icon}\t{size}\n"
        return _gen()

    if lowered.startswith("/cat "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /cat [folder:]<path>
            path_spec = command[5:].strip()
            if not path_spec:
                yield "usage: /cat [folder:]<path>"
                return
            FileSystemGate.initialize()
            if ":" in path_spec:
                folder_id, rel_path = path_spec.split(":", 1)
            else:
                folders = FileSystemGate.list_folders()
                if not folders:
                    yield "No folders configured."
                    return
                folder_id = folders[0]["id"]
                rel_path = path_spec
            result = FileSystemGate.read_file(folder_id, rel_path)
            if not result.success:
                yield f"Error: {result.error}"
                return
            yield result.data
        return _gen()

    if lowered.startswith("/writefile "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /writefile [folder:]<path> <content>
            parts = command[11:].strip().split(maxsplit=1)
            if len(parts) < 2:
                yield "usage: /writefile [folder:]<path> <content>"
                return
            path_spec, content = parts[0], parts[1]
            FileSystemGate.initialize()
            if ":" in path_spec:
                folder_id, rel_path = path_spec.split(":", 1)
            else:
                folders = FileSystemGate.list_folders()
                if not folders:
                    yield "No folders configured."
                    return
                folder_id = folders[0]["id"]
                rel_path = path_spec
            result = FileSystemGate.write_file(folder_id, rel_path, content, create_dirs=True)
            if not result.success:
                yield f"Error: {result.error}"
                return
            await _emit(ctx, "filesystem", result.message)
            yield result.message
            if result.backup_id:
                yield f"\nBackup created: {result.backup_id}"
        return _gen()

    if lowered.startswith("/mkdir "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /mkdir [folder:]<path>
            path_spec = command[7:].strip()
            if not path_spec:
                yield "usage: /mkdir [folder:]<path>"
                return
            FileSystemGate.initialize()
            if ":" in path_spec:
                folder_id, rel_path = path_spec.split(":", 1)
            else:
                folders = FileSystemGate.list_folders()
                if not folders:
                    yield "No folders configured."
                    return
                folder_id = folders[0]["id"]
                rel_path = path_spec
            result = FileSystemGate.mkdir(folder_id, rel_path, parents=True)
            if not result.success:
                yield f"Error: {result.error}"
                return
            await _emit(ctx, "filesystem", result.message)
            yield result.message
        return _gen()

    if lowered.startswith("/rm "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /rm [folder:]<path>
            path_spec = command[4:].strip()
            if not path_spec:
                yield "usage: /rm [folder:]<path>"
                return
            FileSystemGate.initialize()
            if ":" in path_spec:
                folder_id, rel_path = path_spec.split(":", 1)
            else:
                folders = FileSystemGate.list_folders()
                if not folders:
                    yield "No folders configured."
                    return
                folder_id = folders[0]["id"]
                rel_path = path_spec
            result = FileSystemGate.delete(folder_id, rel_path, recursive=True)
            if not result.success:
                yield f"Error: {result.error}"
                return
            await _emit(ctx, "filesystem", result.message)
            yield result.message
            if result.backup_id:
                yield f"\nBackup created: {result.backup_id}"
        return _gen()

    if lowered == "/backups" or lowered.startswith("/backups "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /backups [folder]
            parts = command.split(maxsplit=1)
            folder_id = parts[1].strip() if len(parts) > 1 else None
            FileSystemGate.initialize()
            backups = FileSystemGate.list_backups(folder_id, limit=20)
            if not backups:
                yield "No backups found."
                return
            yield f"Backups ({len(backups)}):\n"
            for b in backups:
                created = b.get("created_at", "?")[:16]
                op = b.get("operation", "?")
                orig = b.get("original_path", "?").split("/")[-1]
                yield f"  {b['id']} | {created} | {op} | {orig}\n"
        return _gen()

    if lowered.startswith("/restore "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /restore <backup_id>
            backup_id = command[9:].strip()
            if not backup_id:
                yield "usage: /restore <backup_id>"
                return
            FileSystemGate.initialize()
            success, message = FileSystemGate.restore_backup(backup_id)
            if success:
                await _emit(ctx, "filesystem", message)
                yield message
            else:
                yield f"Error: {message}"
        return _gen()

    # === ShellGate Commands ===

    if lowered.startswith("/shell "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /shell <command>
            cmd = command[7:].strip()
            if not cmd:
                yield "usage: /shell <command>"
                return
            ShellGate.initialize()

            # Validate command first
            is_valid, error = ShellGate.validate_command(cmd)
            if not is_valid:
                yield f"Blocked: {error}"
                return

            # Show warnings
            risk = ShellGate.estimate_risk(cmd)
            if risk.get("warnings"):
                yield f"Warnings: {', '.join(risk['warnings'])}\n\n"

            # Execute and stream output
            async for line in ShellGate.execute_stream(cmd):
                yield line
        return _gen()

    if lowered.startswith("/shellbg "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /shellbg <command>
            cmd = command[9:].strip()
            if not cmd:
                yield "usage: /shellbg <command>"
                return
            ShellGate.initialize()
            execution = ShellGate.execute_background(cmd)
            if execution.status.value == "failed":
                yield f"Error: {execution.stderr}"
                return
            await _emit(ctx, "shell", f"Background: {cmd[:40]}...")
            yield f"Started background command: {execution.id}\n"
            yield f"Command: {cmd[:80]}{'...' if len(cmd) > 80 else ''}\n"
            yield "Use /shellstatus <id> to check status"
        return _gen()

    if lowered.startswith("/shellstatus "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /shellstatus <id>
            exec_id = command[13:].strip()
            if not exec_id:
                yield "usage: /shellstatus <id>"
                return
            ShellGate.initialize()
            execution = ShellGate.get_status(exec_id)
            if not execution:
                yield f"Execution {exec_id} not found"
                return
            yield f"Status: {execution.status.value}\n"
            if execution.exit_code is not None:
                yield f"Exit code: {execution.exit_code}\n"
            if execution.stdout:
                yield f"STDOUT:\n{execution.stdout[:2000]}\n"
            if execution.stderr:
                yield f"STDERR:\n{execution.stderr[:2000]}\n"
        return _gen()

    if lowered.startswith("/shellkill "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /shellkill <id>
            exec_id = command[11:].strip()
            if not exec_id:
                yield "usage: /shellkill <id>"
                return
            ShellGate.initialize()
            if ShellGate.cancel(exec_id):
                await _emit(ctx, "shell", f"Cancelled: {exec_id}")
                yield f"Cancelled: {exec_id}"
            else:
                yield f"Could not cancel {exec_id} (not found or already complete)"
        return _gen()

    if lowered == "/shellhistory":
        async def _gen() -> AsyncGenerator[str, None]:
            ShellGate.initialize()
            history = ShellGate.get_history(limit=20)
            if not history:
                yield "No command history."
                return
            yield f"Recent commands ({len(history)}):\n"
            for h in history:
                status = "+" if h.get("success") else "-"
                cmd = h.get("command", "?")[:50]
                duration = h.get("duration_seconds", 0)
                yield f"  [{status}] {cmd}... ({duration:.1f}s)\n"
        return _gen()

    # === BrowserGate Commands ===

    if lowered.startswith("/websearch "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /websearch <query> [max:N]
            parts = command[11:].strip().split()
            if not parts:
                yield "usage: /websearch <query> [max:N]"
                return
            # Parse max results option
            max_results = 10
            query_parts = []
            for p in parts:
                if p.startswith("max:"):
                    try:
                        max_results = int(p[4:])
                    except ValueError:
                        pass
                else:
                    query_parts.append(p)
            query = " ".join(query_parts)
            if not query:
                yield "usage: /websearch <query>"
                return
            try:
                await _emit(ctx, "browser", f"Searching: {query[:50]}...")
                response = await BrowserGate.search(query, max_results=max_results)
                await _emit(ctx, "browser", f"Found {len(response.results)} results ({response.search_time_ms}ms)")
                yield f"Search: {query}\n"
                yield f"Provider: {response.provider.value} | Results: {len(response.results)} | Time: {response.search_time_ms}ms\n\n"
                for r in response.results:
                    yield f"{r.position}. [{r.title}]({r.url})\n"
                    yield f"   {r.snippet[:150]}{'...' if len(r.snippet) > 150 else ''}\n\n"
            except Exception as e:
                yield f"Search error: {e}"
        return _gen()

    if lowered.startswith("/fetch "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /fetch <url> [mode:simple|headless] [format:markdown|text|html]
            parts = command[7:].strip().split()
            if not parts:
                yield "usage: /fetch <url> [mode:simple|headless] [format:markdown|text|html]"
                return
            url = parts[0]
            mode = None
            output_format = None
            for p in parts[1:]:
                if p.startswith("mode:"):
                    mode_str = p[5:].lower()
                    if mode_str == "headless":
                        mode = BrowserGate.FetchMode.HEADLESS
                    else:
                        mode = BrowserGate.FetchMode.SIMPLE
                elif p.startswith("format:"):
                    fmt_str = p[7:].lower()
                    if fmt_str == "text":
                        output_format = BrowserGate.ContentFormat.TEXT
                    elif fmt_str == "html":
                        output_format = BrowserGate.ContentFormat.HTML
                    else:
                        output_format = BrowserGate.ContentFormat.MARKDOWN
            try:
                await _emit(ctx, "browser", f"Fetching: {url[:50]}...")
                page = await BrowserGate.fetch(url, mode=mode, output_format=output_format)
                await _emit(ctx, "browser", f"Fetched: {page.word_count} words ({page.fetch_time_ms}ms)")
                yield f"# {page.title}\n\n"
                yield f"URL: {page.url}\n"
                yield f"Words: {page.word_count} | Mode: {page.fetch_mode.value} | Time: {page.fetch_time_ms}ms\n\n"
                yield "---\n\n"
                yield page.content
            except Exception as e:
                yield f"Fetch error: {e}"
        return _gen()

    if lowered.startswith("/browse "):
        async def _gen() -> AsyncGenerator[str, None]:
            # /browse <query> [fetch:N] - search and fetch top N results
            parts = command[8:].strip().split()
            if not parts:
                yield "usage: /browse <query> [fetch:N]"
                return
            fetch_top = 1
            query_parts = []
            for p in parts:
                if p.startswith("fetch:"):
                    try:
                        fetch_top = int(p[6:])
                    except ValueError:
                        pass
                else:
                    query_parts.append(p)
            query = " ".join(query_parts)
            if not query:
                yield "usage: /browse <query>"
                return
            try:
                await _emit(ctx, "browser", f"Browsing: {query[:50]}...")
                response, pages = await BrowserGate.search_and_fetch(query, fetch_top=fetch_top)
                await _emit(ctx, "browser", f"Fetched {len(pages)} pages")
                yield f"Search: {query}\n"
                yield f"Results: {len(response.results)} | Fetched: {len(pages)}\n\n"
                # Show search results
                yield "## Search Results\n\n"
                for r in response.results[:5]:
                    fetched = "[+]" if r.position <= fetch_top else "[ ]"
                    yield f"{fetched} {r.position}. [{r.title}]({r.url})\n"
                yield "\n---\n\n"
                # Show fetched content
                for page in pages:
                    yield f"## {page.title}\n\n"
                    yield f"URL: {page.url}\n"
                    yield f"Words: {page.word_count}\n\n"
                    # Truncate content for display
                    content = page.content[:5000]
                    if len(page.content) > 5000:
                        content += "\n\n[Content truncated...]"
                    yield content
                    yield "\n\n---\n\n"
            except Exception as e:
                yield f"Browse error: {e}"
        return _gen()

    # === Security Commands ===

    if lowered == "/lock":
        async def _gen() -> AsyncGenerator[str, None]:
            await _emit(ctx, "security", "Session locked by user")
            SecurityManager.lock()
            yield "Cathedral locked. Use /unlock or the web UI to unlock."
        return _gen()

    if lowered == "/security":
        async def _gen() -> AsyncGenerator[str, None]:
            status = SecurityManager.get_status()
            if not status.get("encryption_enabled"):
                yield "Encryption: Disabled\n"
                yield "Visit /security in the web UI to enable encryption."
            else:
                session = status.get("session", {})
                is_locked = session.get("is_locked", True)
                tier = status.get("security_tier", "basic")
                yield f"Encryption: Enabled ({tier})\n"
                yield f"Session: {'Locked' if is_locked else 'Unlocked'}\n"
                if not is_locked:
                    time_left = session.get("time_until_lock")
                    if time_left:
                        yield f"Auto-lock in: {time_left // 60} minutes"
                    else:
                        yield "Auto-lock: Disabled"
        return _gen()

    if lowered == "/security-status":
        async def _gen() -> AsyncGenerator[str, None]:
            yield json.dumps(SecurityManager.get_status(), indent=2)
        return _gen()

    return None


__all__ = ["handle_pre_command", "emit_completed_agents", "handle_post_command"]
