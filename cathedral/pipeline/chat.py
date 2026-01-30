from __future__ import annotations

from typing import AsyncGenerator

from rich.text import Text

from cathedral.commands import CommandContext, emit_completed_agents, handle_post_command, handle_pre_command
from cathedral.runtime import loom, memory, thread_personalities
from cathedral.services import ServiceRegistry
from cathedral.MemoryGate.auto_memory import extract_from_exchange, build_memory_context
from cathedral.StarMirror import reflect, reflect_stream
from cathedral import ScriptureGate, PersonalityGate, SecurityManager, ToolGate
from cathedral.ToolGate import PolicyClass, is_tool_response


async def _emit(services: ServiceRegistry, event_type: str, message: str, **kwargs) -> None:
    await services.emit_event(event_type, message, **kwargs)


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
            ScriptureGate.export_thread(history, name)
            return f"thread exported to scripture as '{name}.thread.json'"
        else:
            return "usage: /export thread <name>"

    if lowered.startswith("/import bios"):
        parts = command.split("/import bios", 1)
        path = parts[1].strip()
        try:
            bios_text = ScriptureGate.import_bios(path)
            return Text.assemble(bios_text)
        except Exception as e:
            return f"error loading bios: {e}"

    if lowered.startswith("/import glyph"):
        parts = command.split("/import glyph", 1)
        path = parts[1].strip()
        try:
            glyph_data = ScriptureGate.import_glyph(path)
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


async def process_input_stream(
    user_input: str,
    thread_uid: str,
    services: ServiceRegistry | None = None,
    enable_tools: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Process input and stream response tokens.

    Args:
        user_input: User's message
        thread_uid: Thread identifier
        services: Service registry for events
        enable_tools: Enable tool calling (ToolGate orchestration)

    Yields:
        Response tokens
    """
    command = user_input.strip()
    lowered = command.lower()

    if services is None:
        services = ServiceRegistry()

    ctx = CommandContext(
        loom=loom,
        memory=memory,
        thread_uid=thread_uid,
        thread_personalities=thread_personalities,
        services=services,
    )

    # Block commands and chat while locked (allow security/lock helpers)
    if SecurityManager.is_locked():
        allowed = ("/lock", "/security", "/security-status")
        if not any(lowered == cmd or lowered.startswith(cmd + " ") for cmd in allowed):
            yield "Session is locked. Please unlock at /lock to continue."
            return

    # Handle pre-completion commands (no streaming for these)
    handler = await handle_pre_command(command, lowered, ctx)
    if handler:
        async for token in handler:
            yield token
        return

    # SubAgent completion notifications (only if no pre-command matched)
    async for line in emit_completed_agents(ctx):
        yield line

    # Handle post-completion commands
    handler = await handle_post_command(command, lowered, ctx)
    if handler:
        async for token in handler:
            yield token
        return

    # Load personality for this thread
    PersonalityGate.initialize()
    personality_id = thread_personalities.get(thread_uid, "default")
    personality = PersonalityGate.load(personality_id) or PersonalityGate.get_default()

    # ==========================================================================
    # CATHEDRAL CONTEXT ASSEMBLY ORDER (Normative Specification v1.0)
    # ==========================================================================
    #
    # This ordering is load-bearing. Deviations cause subtle failures.
    #
    # INSTRUCTION LAYER (System):
    #   0. Tool Protocol Kernel     (if tools enabled, immutable)
    #   1. Personality / Guidance   (editable)
    #
    # INTENT LAYER (User):
    #   2. Current User Message     (NON-NEGOTIABLE POSITION)
    #
    # EVIDENCE LAYER (Context):
    #   3. Memory Context           (labeled, lower priority than user intent)
    #   4. Scripture / RAG Context  (labeled, lowest priority)
    #
    # CONTINUITY LAYER (History):
    #   5+. Prior Assistant / User Turns
    #
    # Prohibited:
    #   ❌ Memory or RAG before current user message
    #   ❌ Retrieved context ahead of tool protocol
    #   ❌ Interleaving memory inside assistant history
    # ==========================================================================

    # Build prior conversation history (BEFORE appending current message)
    # This gives us positions 5+ (continuity layer)
    prior_history = await loom.compose_prompt_context_async(user_input, thread_uid)

    # Now append current user message to storage (for future turns)
    await loom.append_async("user", user_input, thread_uid=thread_uid)

    # Gather evidence layer content
    memory_context = build_memory_context(user_input, limit=3, min_confidence=0.5)
    scripture_context = await ScriptureGate.build_context(user_input, limit=2, min_similarity=0.4)

    # === ASSEMBLE IN SPEC ORDER ===
    messages = []

    # Position 0: Tool Protocol Kernel (if enabled)
    if enable_tools:
        ToolGate.initialize()
        tool_prompt = ToolGate.build_tool_prompt(
            enabled_policies={PolicyClass.READ_ONLY}
        )
        if tool_prompt:
            await _emit(services, "tool", "Tool calling enabled")
            messages.append({"role": "system", "content": tool_prompt})

    # Position 1: Personality / Guidance
    system_prompt = personality.get_system_prompt()
    messages.append({"role": "system", "content": system_prompt})

    # Position 2: Current User Message (NON-NEGOTIABLE)
    messages.append({"role": "user", "content": user_input})

    # Position 3: Memory Context (labeled)
    if memory_context:
        await _emit(services, "memory", "Injecting relevant memories")
        messages.append({
            "role": "system",
            "content": f"[Memory Context]\n{memory_context}"
        })

    # Position 4: Scripture / RAG Context (labeled)
    if scripture_context:
        await _emit(services, "system", "RAG context loaded")
        messages.append({
            "role": "system",
            "content": f"[Relevant Documents]\n{scripture_context}"
        })

    # Position 5+: Prior History (continuity layer)
    messages.extend(prior_history)

    full_history = messages

    # Get model response
    model = personality.llm_config.model
    temperature = personality.llm_config.temperature

    if enable_tools:
        # Tool mode: collect full response, then run orchestration loop
        full_response = ""
        async for token in reflect_stream(full_history, model=model, temperature=temperature):
            full_response += token

        # Check if response contains tool calls
        if is_tool_response(full_response):
            await _emit(services, "tool", "Processing tool calls")

            # Create orchestrator with event emission
            async def emit_wrapper(event_type: str, message: str, **kwargs):
                await _emit(services, event_type, message, **kwargs)

            orchestrator = ToolGate.get_orchestrator(
                enabled_policies=[PolicyClass.READ_ONLY],
                emit_event=emit_wrapper,
            )

            # Run tool loop and yield tokens
            async for token in orchestrator.execute_loop(
                initial_response=full_response,
                messages=full_history,
                model=model,
                temperature=temperature,
            ):
                yield token
                full_response = token  # Track final response for storage

        else:
            # No tool calls, yield the response
            yield full_response

    else:
        # Standard mode: stream directly
        full_response = ""
        async for token in reflect_stream(full_history, model=model, temperature=temperature):
            full_response += token
            yield token

    # Store complete assistant response in conversation history (async with embedding)
    await loom.append_async("assistant", full_response, thread_uid=thread_uid)

    # Extract and store memory from this exchange
    memory_result = extract_from_exchange(user_input, full_response, thread_uid)
    if memory_result:
        await _emit(services, "memory", "Memory extracted from exchange")


__all__ = ["process_input", "process_input_stream"]
