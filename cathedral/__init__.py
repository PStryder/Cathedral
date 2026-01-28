from typing import AsyncGenerator
import json
from loom import Loom
from cathedral.StarMirror import (
    reflect,
    reflect_stream,
    reflect_vision_stream,
    reflect_audio,
    describe_image,
    compare_images,
    transcribe_audio,
    build_multimodal_message,
    ContentBuilder,
    ImageDetail,
)
from cathedral import ScriptureGate
from cathedral.ScriptureGate import export_thread, import_bios, import_glyph
from cathedral import MemoryGate
from cathedral.MemoryGate.auto_memory import (
    extract_from_exchange,
    build_memory_context,
    format_search_results,
    format_recall_results
)
from cathedral import MetadataChannel
from cathedral import SubAgentGate
from cathedral import Config
from cathedral import PersonalityGate
from cathedral import SecurityManager
from cathedral import FileSystemGate
from cathedral import ShellGate
from cathedral import BrowserGate
from rich.text import Text

# Thread-personality associations (in-memory, could be persisted)
_thread_personalities: dict = {}

# Placeholders for event emission (injected by run.py)
emit_event = None  # Will be set by run.py
record_agent_update = None  # Will be set by run.py

import asyncio

async def _emit(event_type: str, message: str, **kwargs):
    """Safe event emission helper - works even if emit_event not injected."""
    global emit_event
    if emit_event and asyncio.iscoroutinefunction(emit_event):
        await emit_event(event_type, message, **kwargs)


async def _agent_update(agent_id: str, message: str, status: str = "running"):
    """Safe agent update helper."""
    global record_agent_update
    if record_agent_update and asyncio.iscoroutinefunction(record_agent_update):
        await record_agent_update(agent_id, message, status)


loom = Loom()

def process_input(user_input: str, thread_uid: str) -> str:
    """Synchronous version (legacy, prefer process_input_stream)."""
    command = user_input.strip()
    lowered = command.lower()

    if lowered == "/history":
        history = loom.recall(thread_uid)
        if not history:
            return Text.assemble("no thread history yet")

        history_text = "\n".join(f"{entry['role']}: {entry['content']}" for entry in history)
        return Text.assemble(history_text)

    if lowered == "/forget":
        loom.clear(thread_uid)
        return "the memory fades—forgotten by the Cathedral"

    if lowered.startswith("/export thread"):
        parts = command.split()
        if len(parts) >= 3:
            name = parts[2]
            history = loom.recall(thread_uid)
            export_thread(history, name)
            return f"thread exported to scripture as '{name}.thread.json'"
        else:
            return "usage: /export thread <name>"

    if lowered.startswith("/import bios"):
        parts = command.split("/import bios", 1)
        path = parts[1].strip()
        try:
            bios_text = import_bios(path)
            return Text.assemble(bios_text)
        except Exception as e:
            return f"error loading bios: {e}"

    if lowered.startswith("/import glyph"):
        parts = command.split("/import glyph", 1)
        path = parts[1].strip()
        try:
            glyph_data = import_glyph(path)
            return Text.assemble(str(glyph_data))
        except Exception as e:
            return f"error loading glyph: {e}"

    # Append user message to memory
    loom.append("user", user_input, thread_uid=thread_uid)

    # Build full context with summarization and vector recall
    full_history = loom.compose_prompt_context(user_input, thread_uid)

    # Query StarMirror with enriched context
    response = reflect(full_history)

    # Store assistant response
    loom.append("assistant", response, thread_uid=thread_uid)

    return response


async def process_input_stream(user_input: str, thread_uid: str) -> AsyncGenerator[str, None]:
    """Process input and stream response tokens. Stores complete response after streaming."""
    command = user_input.strip()
    lowered = command.lower()

    # Handle commands (no streaming for these)
    if lowered == "/history":
        history = await loom.recall_async(thread_uid)
        if not history:
            yield "no thread history yet"
            return
        history_text = "\n".join(f"{entry['role']}: {entry['content']}" for entry in history)
        yield history_text
        return

    if lowered == "/forget":
        loom.clear(thread_uid)
        yield "the memory fades—forgotten by the Cathedral"
        return

    if lowered.startswith("/export thread"):
        parts = command.split()
        if len(parts) >= 3:
            name = parts[2]
            history = await loom.recall_async(thread_uid)
            export_thread(history, name)
            yield f"thread exported to scripture as '{name}.thread.json'"
        else:
            yield "usage: /export thread <name>"
        return

    if lowered.startswith("/import bios"):
        parts = command.split("/import bios", 1)
        path = parts[1].strip()
        try:
            bios_text = import_bios(path)
            yield str(bios_text)
        except Exception as e:
            yield f"error loading bios: {e}"
        return

    if lowered.startswith("/import glyph"):
        parts = command.split("/import glyph", 1)
        path = parts[1].strip()
        try:
            glyph_data = import_glyph(path)
            yield str(glyph_data)
        except Exception as e:
            yield f"error loading glyph: {e}"
        return

    # === MemoryGate Commands ===

    if lowered.startswith("/search "):
        query = command[8:].strip()
        if not query:
            yield "usage: /search <query>"
            return
        await _emit("memory", f"Searching: {query[:50]}...")
        results = MemoryGate.search(query, limit=5)
        result_count = len(results) if results else 0
        await _emit("memory", f"Found {result_count} results")
        yield format_search_results(results)
        return

    if lowered.startswith("/remember "):
        text = command[10:].strip()
        if not text:
            yield "usage: /remember <observation>"
            return
        await _emit("memory", f"Storing observation...")
        result = MemoryGate.store_observation(text, confidence=0.9, domain="explicit")
        if result:
            await _emit("memory", f"Stored: obs:{result.get('id', '?')}")
            yield f"Stored observation:{result.get('id', '?')}"
        else:
            await _emit("memory", "Store failed")
            yield "Failed to store observation (MemoryGate may not be configured)"
        return

    if lowered == "/memories" or lowered.startswith("/memories "):
        parts = command.split(maxsplit=1)
        domain = parts[1].strip() if len(parts) > 1 else None
        results = MemoryGate.recall(domain=domain, limit=10)
        yield format_recall_results(results)
        return

    if lowered.startswith("/concept "):
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
        return

    if lowered.startswith("/pattern "):
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
        return

    if lowered == "/memstats":
        stats = MemoryGate.get_stats()
        if stats:
            yield f"Memory Stats:\n"
            counts = stats.get("counts", {})
            for k, v in counts.items():
                yield f"  {k}: {v}\n"
        else:
            yield "MemoryGate not configured or unavailable"
        return

    # === Knowledge Discovery Commands ===

    if lowered.startswith("/discover "):
        ref = command[10:].strip()
        if not ref or ":" not in ref:
            yield "usage: /discover <type:id> (e.g., message:abc123, thread:xyz789, concept:42)"
            return
        await _emit("memory", f"Discovering relationships for: {ref}")
        try:
            from cathedral.MemoryGate.discovery import discover_for_ref
            relationships = await discover_for_ref(ref)
            if not relationships:
                yield f"No relationships discovered for {ref}"
                return
            yield f"Discovered {len(relationships)} relationships for {ref}:\n\n"
            for rel in relationships:
                yield f"  [{rel.similarity:.3f}] {rel.rel_type} -> {rel.to_ref}\n"
            await _emit("memory", f"Discovered {len(relationships)} relationships")
        except Exception as e:
            yield f"Discovery error: {e}"
        return

    if lowered.startswith("/related "):
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
        return

    if lowered == "/discovery":
        # Show discovery service status
        try:
            from cathedral.MemoryGate.discovery import get_discovery_service
            svc = get_discovery_service()
            yield f"Discovery Service Status:\n"
            yield f"  Running: {svc._running}\n"
            yield f"  Enabled: {svc.config.enabled}\n"
            yield f"  Min similarity: {svc.config.min_similarity}\n"
            yield f"  Top K: {svc.config.top_k}\n"
            yield f"  Thread interval: {svc.config.thread_embedding_interval} messages\n"
            yield f"  Queue size: ~{svc._queue.qsize()}\n"
        except Exception as e:
            yield f"Discovery service not available: {e}"
        return

    # === Loom Semantic Search ===

    if lowered.startswith("/loomsearch "):
        query = command[12:].strip()
        if not query:
            yield "usage: /loomsearch <query>"
            return
        results = await loom.semantic_search(query, thread_uid=thread_uid, limit=5, include_all_threads=True)
        if not results:
            yield "No semantically similar messages found."
            return
        yield "Semantically similar messages:\n"
        for r in results:
            sim = r.get("similarity", 0)
            role = r.get("role", "?")
            content = r.get("content", "")[:150]
            yield f"  [{sim:.2f}] {role}: {content}...\n"
        return

    if lowered == "/backfill":
        yield "Generating embeddings for messages without them...\n"
        count = await loom.backfill_embeddings(batch_size=50)
        yield f"Generated embeddings for {count} messages."
        return

    # === MetadataChannel Commands ===

    if lowered.startswith("/meta"):
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

        context = {"loom": loom, "thread_uid": thread_uid}
        result = MetadataChannel.query(target, fields, context)
        yield json.dumps(result, separators=(",", ":"))
        return

    if lowered == "/metafields":
        channel = MetadataChannel.get_channel()
        yield f"providers: {','.join(channel.list_providers())}\n"
        yield f"fields: {','.join(channel.list_fields())}"
        return

    # === Multi-Modal Commands ===

    if lowered.startswith("/image "):
        # /image <path> <prompt>
        parts = command[7:].strip().split(maxsplit=1)
        if len(parts) < 2:
            yield "usage: /image <path> <prompt>"
            return
        image_path, prompt = parts[0], parts[1]
        try:
            # Store user intent
            await loom.append_async("user", f"[Image: {image_path}] {prompt}", thread_uid=thread_uid)

            full_response = ""
            async for token in reflect_vision_stream(prompt, [image_path]):
                full_response += token
                yield token

            await loom.append_async("assistant", full_response, thread_uid=thread_uid)
        except FileNotFoundError:
            yield f"Image not found: {image_path}"
        except Exception as e:
            yield f"Error processing image: {e}"
        return

    if lowered.startswith("/describe "):
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
            await loom.append_async("user", f"[Describe image: {image_path}]", thread_uid=thread_uid)

            full_response = ""
            async for token in reflect_vision_stream(
                "Describe this image in detail.",
                [image_path],
                detail=detail
            ):
                full_response += token
                yield token

            await loom.append_async("assistant", full_response, thread_uid=thread_uid)
        except FileNotFoundError:
            yield f"Image not found: {image_path}"
        except Exception as e:
            yield f"Error: {e}"
        return

    if lowered.startswith("/compare "):
        # /compare <path1> <path2> [prompt]
        parts = command[9:].strip().split(maxsplit=2)
        if len(parts) < 2:
            yield "usage: /compare <path1> <path2> [prompt]"
            return
        img1, img2 = parts[0], parts[1]
        prompt = parts[2] if len(parts) > 2 else "Compare these images and describe the differences."

        try:
            await loom.append_async("user", f"[Compare: {img1} vs {img2}] {prompt}", thread_uid=thread_uid)

            full_response = ""
            async for token in reflect_vision_stream(prompt, [img1, img2]):
                full_response += token
                yield token

            await loom.append_async("assistant", full_response, thread_uid=thread_uid)
        except FileNotFoundError as e:
            yield f"Image not found: {e}"
        except Exception as e:
            yield f"Error: {e}"
        return

    if lowered.startswith("/transcribe "):
        # /transcribe <audio_path>
        audio_path = command[12:].strip()
        if not audio_path:
            yield "usage: /transcribe <audio_path>"
            return
        try:
            yield "Transcribing audio...\n"
            transcription = await transcribe_audio(audio_path)
            yield f"Transcription:\n{transcription}"
            await loom.append_async("user", f"[Transcribed audio: {audio_path}]\n{transcription}", thread_uid=thread_uid)
        except FileNotFoundError:
            yield f"Audio file not found: {audio_path}"
        except Exception as e:
            yield f"Error transcribing: {e}"
        return

    if lowered.startswith("/audio "):
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
            await loom.append_async("user", f"[Audio: {audio_path}] {prompt}", thread_uid=thread_uid)
            await loom.append_async("assistant", response, thread_uid=thread_uid)
        except FileNotFoundError:
            yield f"Audio file not found: {audio_path}"
        except Exception as e:
            yield f"Error: {e}"
        return

    # === SubAgent Commands ===

    # Check for completed agents and notify
    completed = SubAgentGate.check_completed()
    for agent_id in completed:
        agent_status = SubAgentGate.status(agent_id)
        if agent_status:
            status_str = agent_status.get("status", "?")
            yield f"[SubAgent {agent_id} {status_str}]\n"

    if lowered.startswith("/spawn "):
        # /spawn <task description>
        task = command[7:].strip()
        if not task:
            yield "usage: /spawn <task description>"
            return
        try:
            await _emit("agent", f"Spawning agent for: {task[:50]}...")
            agent_id = SubAgentGate.spawn(
                task=task,
                context={"thread_uid": thread_uid}
            )
            await _agent_update(agent_id, f"Started: {task[:40]}...", "running")
            yield f"Spawned sub-agent {agent_id}\n"
            yield f"Task: {task[:100]}{'...' if len(task) > 100 else ''}\n"
            yield "Use /agents to check status, /result <id> to get result"
        except Exception as e:
            await _emit("agent", f"Spawn failed: {e}")
            yield f"Failed to spawn agent: {e}"
        return

    if lowered == "/agents":
        agents = SubAgentGate.list_agents()
        if not agents:
            yield "No sub-agents"
            return
        yield f"Sub-agents ({len(agents)}):\n"
        for a in agents:
            status_icon = {"running": "~", "completed": "+", "failed": "!", "cancelled": "x"}.get(a["status"], "?")
            task_short = a["task"][:40] + "..." if len(a["task"]) > 40 else a["task"]
            yield f"  [{status_icon}] {a['id']} | {task_short}\n"
        return

    if lowered.startswith("/agent "):
        agent_id = command[7:].strip()
        if not agent_id:
            yield "usage: /agent <id>"
            return
        agent_status = SubAgentGate.status(agent_id)
        if not agent_status:
            yield f"Agent {agent_id} not found"
            return
        yield json.dumps(agent_status, indent=2)
        return

    if lowered.startswith("/result "):
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
        return

    if lowered.startswith("/cancel "):
        agent_id = command[8:].strip()
        if not agent_id:
            yield "usage: /cancel <id>"
            return
        if SubAgentGate.cancel(agent_id):
            yield f"Cancelled agent {agent_id}"
        else:
            yield f"Could not cancel agent {agent_id} (not running or not found)"
        return

    # === ScriptureGate Commands ===

    if lowered.startswith("/store "):
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
            yield f"Indexing in background..."
        except FileNotFoundError:
            yield f"File not found: {file_path}"
        except Exception as e:
            yield f"Error storing: {e}"
        return

    if lowered.startswith("/scripture "):
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
        return

    if lowered.startswith("/scriptsearch "):
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
        return

    if lowered == "/scriptures" or lowered.startswith("/scriptures "):
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
        return

    if lowered == "/scriptstats":
        stats = ScriptureGate.stats()
        yield f"Scripture Stats:\n"
        yield f"  Total: {stats.get('total', 0)}\n"
        yield f"  Indexed: {stats.get('indexed', 0)}\n"
        yield f"  Size: {stats.get('total_size_mb', 0)} MB\n"
        by_type = stats.get("by_type", {})
        if by_type:
            yield "  By type:\n"
            for t, count in by_type.items():
                yield f"    {t}: {count}\n"
        return

    if lowered == "/scriptindex":
        yield "Indexing unindexed scriptures...\n"
        count = await ScriptureGate.backfill_index(batch_size=10)
        yield f"Indexed {count} scriptures."
        return

    # === Personality Commands ===

    if lowered == "/personalities":
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
        return

    if lowered.startswith("/personality ") and not lowered.startswith("/personality-"):
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
        _thread_personalities[thread_uid] = personality_id
        PersonalityGate.PersonalityManager.record_usage(personality_id)
        await _emit("system", f"Personality: {personality.name} ({personality.llm_config.model})")
        yield f"Switched to personality: {personality.name}\n"
        yield f"Model: {personality.llm_config.model}\n"
        yield f"Temperature: {personality.llm_config.temperature}"
        return

    if lowered == "/personality":
        # Show current personality for this thread
        current_id = _thread_personalities.get(thread_uid, "default")
        PersonalityGate.initialize()
        personality = PersonalityGate.load(current_id)
        if personality:
            yield f"Current personality: {personality.name} ({personality.id})\n"
            yield f"Model: {personality.llm_config.model}\n"
            yield f"Temperature: {personality.llm_config.temperature}\n"
            yield f"Style: {', '.join(personality.behavior.style_tags)}"
        else:
            yield "Using default personality"
        return

    if lowered.startswith("/personality-info "):
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
        return

    if lowered.startswith("/personality-create "):
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
        return

    if lowered.startswith("/personality-delete "):
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
        return

    if lowered.startswith("/personality-export "):
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
        return

    if lowered.startswith("/personality-copy "):
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
        return

    # === FileSystemGate Commands ===

    if lowered == "/sources":
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
        return

    if lowered.startswith("/sources-add "):
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
            await _emit("filesystem", message)
            yield message
        else:
            yield f"Error: {message}"
        return

    if lowered.startswith("/ls "):
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
        return

    if lowered.startswith("/cat "):
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
        return

    if lowered.startswith("/writefile "):
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
        await _emit("filesystem", result.message)
        yield result.message
        if result.backup_id:
            yield f"\nBackup created: {result.backup_id}"
        return

    if lowered.startswith("/mkdir "):
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
        await _emit("filesystem", result.message)
        yield result.message
        return

    if lowered.startswith("/rm "):
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
        await _emit("filesystem", result.message)
        yield result.message
        if result.backup_id:
            yield f"\nBackup created: {result.backup_id}"
        return

    if lowered == "/backups" or lowered.startswith("/backups "):
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
        return

    if lowered.startswith("/restore "):
        # /restore <backup_id>
        backup_id = command[9:].strip()
        if not backup_id:
            yield "usage: /restore <backup_id>"
            return
        FileSystemGate.initialize()
        success, message = FileSystemGate.restore_backup(backup_id)
        if success:
            await _emit("filesystem", message)
            yield message
        else:
            yield f"Error: {message}"
        return

    # === ShellGate Commands ===

    if lowered.startswith("/shell "):
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
        return

    if lowered.startswith("/shellbg "):
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
        await _emit("shell", f"Background: {cmd[:40]}...")
        yield f"Started background command: {execution.id}\n"
        yield f"Command: {cmd[:80]}{'...' if len(cmd) > 80 else ''}\n"
        yield "Use /shellstatus <id> to check status"
        return

    if lowered.startswith("/shellstatus "):
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
        return

    if lowered.startswith("/shellkill "):
        # /shellkill <id>
        exec_id = command[11:].strip()
        if not exec_id:
            yield "usage: /shellkill <id>"
            return
        ShellGate.initialize()
        if ShellGate.cancel(exec_id):
            await _emit("shell", f"Cancelled: {exec_id}")
            yield f"Cancelled: {exec_id}"
        else:
            yield f"Could not cancel {exec_id} (not found or already complete)"
        return

    if lowered == "/shellhistory":
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
        return

    # === BrowserGate Commands ===

    if lowered.startswith("/websearch "):
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
            await _emit("browser", f"Searching: {query[:50]}...")
            response = await BrowserGate.search(query, max_results=max_results)
            await _emit("browser", f"Found {len(response.results)} results ({response.search_time_ms}ms)")
            yield f"Search: {query}\n"
            yield f"Provider: {response.provider.value} | Results: {len(response.results)} | Time: {response.search_time_ms}ms\n\n"
            for r in response.results:
                yield f"{r.position}. [{r.title}]({r.url})\n"
                yield f"   {r.snippet[:150]}{'...' if len(r.snippet) > 150 else ''}\n\n"
        except Exception as e:
            yield f"Search error: {e}"
        return

    if lowered.startswith("/fetch "):
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
            await _emit("browser", f"Fetching: {url[:50]}...")
            page = await BrowserGate.fetch(url, mode=mode, output_format=output_format)
            await _emit("browser", f"Fetched: {page.word_count} words ({page.fetch_time_ms}ms)")
            yield f"# {page.title}\n\n"
            yield f"URL: {page.url}\n"
            yield f"Words: {page.word_count} | Mode: {page.fetch_mode.value} | Time: {page.fetch_time_ms}ms\n\n"
            yield "---\n\n"
            yield page.content
        except Exception as e:
            yield f"Fetch error: {e}"
        return

    if lowered.startswith("/browse "):
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
            await _emit("browser", f"Browsing: {query[:50]}...")
            response, pages = await BrowserGate.search_and_fetch(query, fetch_top=fetch_top)
            await _emit("browser", f"Fetched {len(pages)} pages")
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
        return

    # === Security Commands ===

    if lowered == "/lock":
        await _emit("security", "Session locked by user")
        SecurityManager.lock()
        yield "Cathedral locked. Use /unlock or the web UI to unlock."
        return

    if lowered == "/security":
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
        return

    if lowered == "/security-status":
        yield json.dumps(SecurityManager.get_status(), indent=2)
        return

    # Check if session is locked - if so, can't process chat
    if SecurityManager.is_locked():
        yield "Session is locked. Please unlock at /lock to continue."
        return

    # Append user message to Loom thread history (async with embedding)
    await loom.append_async("user", user_input, thread_uid=thread_uid)

    # Load personality for this thread
    PersonalityGate.initialize()
    personality_id = _thread_personalities.get(thread_uid, "default")
    personality = PersonalityGate.load(personality_id) or PersonalityGate.get_default()

    # Build full context with summarization and semantic search
    full_history = await loom.compose_prompt_context_async(user_input, thread_uid)

    # Inject personality system prompt at the start
    system_prompt = personality.get_system_prompt()
    full_history.insert(0, {"role": "system", "content": system_prompt})

    # Inject semantically relevant memories from MemoryGate
    memory_context = build_memory_context(user_input, limit=3, min_confidence=0.5)
    if memory_context:
        await _emit("memory", "Injecting relevant memories")
        full_history.insert(1, {"role": "system", "content": memory_context})

    # Inject relevant documents from ScriptureGate (RAG)
    scripture_context = await ScriptureGate.build_context(user_input, limit=2, min_similarity=0.4)
    if scripture_context:
        await _emit("system", "RAG context loaded")
        full_history.insert(1, {"role": "system", "content": scripture_context})

    # Stream tokens from StarMirror using personality settings
    full_response = ""
    async for token in reflect_stream(
        full_history,
        model=personality.llm_config.model,
        temperature=personality.llm_config.temperature
    ):
        full_response += token
        yield token

    # Store complete assistant response in Loom thread (async with embedding)
    await loom.append_async("assistant", full_response, thread_uid=thread_uid)

    # Extract and store memory from this exchange
    memory_result = extract_from_exchange(user_input, full_response, thread_uid)
    if memory_result:
        await _emit("memory", "Memory extracted from exchange")
