from __future__ import annotations

import asyncio
from functools import partial
from typing import AsyncGenerator

from rich.text import Text

from cathedral.commands import CommandContext, emit_completed_agents, handle_post_command, handle_pre_command
from cathedral.runtime import loom, memory, thread_personalities
from cathedral.services import ServiceRegistry
from cathedral.MemoryGate.auto_memory import extract_from_exchange, build_memory_context
from cathedral.StarMirror import reflect, reflect_stream
from cathedral import ScriptureGate, PersonalityGate, SecurityManager, ToolGate
from cathedral.ToolGate import PolicyClass, is_tool_response, get_policy_manager


def _get_enabled_policies() -> set[PolicyClass]:
    """
    Get the set of policies that are actually enabled by PolicyManager.

    This checks SecurityManager state to avoid advertising capabilities
    that will be rejected at execution time (e.g., WRITE when locked).
    """
    pm = get_policy_manager()
    enabled = set()

    # READ_ONLY is always available
    enabled.add(PolicyClass.READ_ONLY)

    # Try to enable each policy and track successes
    for policy in [PolicyClass.NETWORK, PolicyClass.WRITE, PolicyClass.PRIVILEGED, PolicyClass.DESTRUCTIVE]:
        success, _ = pm.enable_policy(policy)
        if success:
            enabled.add(policy)

    return enabled


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
    enabled_gates: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Process input and stream response tokens.

    Args:
        user_input: User's message
        thread_uid: Thread identifier
        services: Service registry for events
        enable_tools: Enable tool calling (ToolGate orchestration)
        enabled_gates: List of gate names to enable (e.g., ["MemoryGate", "ShellGate"])
                       If None and enable_tools=True, all gates are enabled

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
    # CATHEDRAL CONTEXT ASSEMBLY ORDER (Normative Specification v1.1)
    # ==========================================================================
    #
    # This ordering is load-bearing. Deviations cause subtle failures.
    #
    # INSTRUCTION LAYER (System):
    #   0. Tool Protocol Kernel     (if tools enabled, immutable)
    #   1. Personality / Guidance   (editable)
    #
    # EVIDENCE LAYER (System - labeled context):
    #   2. Scripture / RAG Context  (labeled, background documents)
    #   3. Memory Context           (labeled, higher priority than RAG)
    #
    # CONTINUITY LAYER (History):
    #   4+. Prior Assistant / User Turns (chronological)
    #
    # INTENT LAYER (User - LAST):
    #   N. Current User Message     (NON-NEGOTIABLE FINAL POSITION)
    #
    # Rationale: User message last ensures the model focuses on the current
    # request. Context injected before history provides background without
    # overwhelming recency. Tool protocol first ensures instructions are seen.
    #
    # Prohibited:
    #   ❌ Current user message anywhere except LAST
    #   ❌ Retrieved context ahead of tool protocol
    #   ❌ Interleaving memory inside assistant history
    # ==========================================================================

    # Build prior conversation history (BEFORE appending current message)
    # This gives us positions 5+ (continuity layer)
    prior_history = await loom.compose_prompt_context_async(user_input, thread_uid)

    # Now append current user message to storage (for future turns)
    await loom.append_async("user", user_input, thread_uid=thread_uid)

    # Gather evidence layer content
    # Run sync memory search in executor to avoid greenlet errors
    loop = asyncio.get_running_loop()
    memory_context = await loop.run_in_executor(
        None,
        partial(build_memory_context, user_input, limit=3, min_confidence=0.5)
    )
    scripture_context = await ScriptureGate.build_context(user_input, limit=2, min_similarity=0.4)

    # === ASSEMBLE IN CHRONOLOGICAL ORDER ===
    # LLMs expect: system prompts -> conversation history -> current message
    messages = []

    # Position 0: Tool Protocol Kernel (if enabled)
    # Determine which policies are actually enabled (not just requested)
    active_policies = _get_enabled_policies() if enable_tools else set()

    if enable_tools:
        ToolGate.initialize()
        tool_prompt = ToolGate.build_tool_prompt(
            enabled_policies=active_policies,
            gate_filter=enabled_gates,
        )
        if tool_prompt:
            gates_msg = f" ({', '.join(enabled_gates)})" if enabled_gates else " (all gates)"
            policy_names = sorted(p.value for p in active_policies)
            await _emit(services, "tool", f"Tool calling enabled{gates_msg}, policies: {', '.join(policy_names)}")
            messages.append({"role": "system", "content": tool_prompt})

    # Position 1: Personality / Guidance
    system_prompt = personality.get_system_prompt()
    messages.append({"role": "system", "content": system_prompt})

    # Position 2: Scripture / RAG Context (labeled, lower priority background docs)
    if scripture_context:
        await _emit(services, "system", "RAG context loaded")
        messages.append({
            "role": "system",
            "content": f"[Relevant Documents]\n{scripture_context}"
        })

    # Position 3: Memory Context (labeled, higher priority than RAG)
    if memory_context:
        await _emit(services, "memory", "Injecting relevant memories")
        messages.append({
            "role": "system",
            "content": f"[Memory Context]\n{memory_context}"
        })

    # Position 4+: Prior History (conversation continuity - chronological)
    messages.extend(prior_history)

    # Position LAST: Current User Message (what we're responding to)
    messages.append({"role": "user", "content": user_input})

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
                enabled_policies=list(active_policies),
                emit_event=emit_wrapper,
                gate_filter=enabled_gates,
            )

            # Run tool loop and yield tokens
            tool_output = ""
            async for token in orchestrator.execute_loop(
                initial_response=full_response,
                messages=full_history,
                model=model,
                temperature=temperature,
            ):
                yield token
                tool_output += token  # Accumulate for storage
            full_response = tool_output if tool_output else full_response

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
