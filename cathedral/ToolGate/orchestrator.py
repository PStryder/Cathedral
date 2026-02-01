"""
ToolGate Orchestrator.

Manages the tool execution loop within the chat pipeline.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from cathedral.shared.gate import GateLogger
from cathedral.ToolGate.models import (
    PolicyClass,
    ToolBudget,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from cathedral.ToolGate.policy import PolicyManager, get_policy_manager
from cathedral.ToolGate.protocol import (
    format_tool_results,
    is_tool_response,
    parse_tool_calls,
    validate_args,
)
from cathedral.ToolGate.prompt import get_json_correction_message
from cathedral.ToolGate.registry import ToolRegistry

_log = GateLogger.get("ToolGate")


# =============================================================================
# Gate Dispatch
# =============================================================================


def _get_gate_module(gate_name: str):
    """Dynamically import and return a gate module."""
    if gate_name == "MemoryGate":
        from cathedral import MemoryGate
        return MemoryGate
    elif gate_name == "FileSystemGate":
        from cathedral import FileSystemGate
        return FileSystemGate
    elif gate_name == "ShellGate":
        from cathedral import ShellGate
        return ShellGate
    elif gate_name == "ScriptureGate":
        from cathedral import ScriptureGate
        return ScriptureGate
    elif gate_name == "BrowserGate":
        from cathedral import BrowserGate
        return BrowserGate
    elif gate_name == "SubAgentGate":
        from cathedral import SubAgentGate
        return SubAgentGate
    elif gate_name == "MCPClient":
        from cathedral import MCPClient
        return MCPClient
    else:
        raise ValueError(f"Unknown gate: {gate_name}")


async def _dispatch_tool(tool: ToolDefinition, args: Dict[str, Any]) -> Any:
    """
    Dispatch a tool call to its Gate method.

    Args:
        tool: Tool definition
        args: Validated arguments

    Returns:
        Tool execution result
    """
    # Special handling for MCP tools
    if tool.gate == "MCPClient":
        return await _dispatch_mcp_tool(tool, args)

    gate = _get_gate_module(tool.gate)
    method = getattr(gate, tool.method)

    if tool.is_async:
        result = await method(**args)
    else:
        result = method(**args)

    return result


async def _dispatch_mcp_tool(tool: ToolDefinition, args: Dict[str, Any]) -> Any:
    """
    Dispatch an MCP tool call.

    Tool name format: MCP.{server_id}.{tool_name}

    Args:
        tool: Tool definition
        args: Tool arguments

    Returns:
        Tool execution result
    """
    from cathedral import MCPClient

    # Parse tool name to extract server_id and tool_name
    parts = tool.name.split(".", 2)
    if len(parts) != 3 or parts[0] != "MCP":
        raise ValueError(f"Invalid MCP tool name format: {tool.name}")

    server_id = parts[1]
    tool_name = parts[2]

    # Call the MCP tool
    result = await MCPClient.call_mcp_tool(server_id, tool_name, args)

    if result.success:
        return result.content
    else:
        raise RuntimeError(result.error or "MCP tool call failed")


# =============================================================================
# Tool Orchestrator
# =============================================================================


class ToolOrchestrator:
    """
    Manages tool execution loop within chat pipeline.

    Handles parsing, validation, policy enforcement, execution, and result injection.
    """

    def __init__(
        self,
        policy_manager: Optional[PolicyManager] = None,
        budget: Optional[ToolBudget] = None,
        emit_event: Optional[Callable] = None,
        enabled_gates: Optional[List[str]] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            policy_manager: Policy manager instance (uses global if None)
            budget: Execution budget (uses defaults if None)
            emit_event: Optional async callback for events
            enabled_gates: Optional list of gate names to enable (None = all gates)
        """
        self.policy_manager = policy_manager or get_policy_manager()
        self.budget = budget or ToolBudget()
        self.emit_event = emit_event
        self.enabled_gates = enabled_gates  # None means all gates enabled

        # Initialize registry
        ToolRegistry.initialize()

    async def _emit(self, event_type: str, message: str, **kwargs) -> None:
        """Emit an event if callback is configured."""
        if self.emit_event:
            try:
                await self.emit_event(event_type, message, **kwargs)
            except Exception as e:
                _log.warning(f"Event emission failed: {e}")

    async def execute_single(self, call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            call: Tool call to execute

        Returns:
            Tool execution result
        """
        # Look up tool
        tool = ToolRegistry.get_tool(call.tool)
        if not tool:
            return ToolResult.failure(call.id, f"Unknown tool: {call.tool}")

        # Check if gate is enabled
        if self.enabled_gates is not None and tool.gate not in self.enabled_gates:
            return ToolResult.failure(
                call.id,
                f"Gate {tool.gate} is not enabled. Enabled gates: {', '.join(self.enabled_gates)}"
            )

        # Validate arguments
        valid, error = validate_args(tool, call.args)
        if not valid:
            return ToolResult.failure(call.id, f"Invalid arguments: {error}")

        # Check policy
        allowed, reason = self.policy_manager.check(tool, call.args)
        if not allowed:
            return ToolResult.failure(call.id, f"Policy denied: {reason}")

        # Execute
        try:
            await self._emit("tool", f"Executing {call.tool}")
            result = await _dispatch_tool(tool, call.args)
            _log.info(f"Tool {call.tool} executed successfully")
            return ToolResult.success(call.id, result)

        except Exception as e:
            _log.error(f"Tool {call.tool} failed: {e}")
            return ToolResult.failure(call.id, str(e))

    async def execute_batch(self, calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute a batch of tool calls.

        Args:
            calls: List of tool calls

        Returns:
            List of results in same order
        """
        results = []
        for call in calls:
            result = await self.execute_single(call)
            results.append(result)
            self.budget.use_calls(1)

            # Stop if budget exceeded
            if self.budget.exceeded():
                break

        return results

    async def execute_loop(
        self,
        initial_response: str,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Run the tool execution loop.

        This is the main agentic loop:
        1. Parse tool calls from response
        2. Execute tools, collect results
        3. Inject results into conversation
        4. Get next model response
        5. Repeat until no more tool calls or budget exceeded

        Args:
            initial_response: First model response to check for tool calls
            messages: Conversation history (will be modified)
            model: Model name for follow-up calls
            temperature: Temperature for follow-up calls
            stream_callback: Optional callback for streaming tokens

        Yields:
            Response text tokens
        """
        from cathedral.StarMirror import reflect_stream

        current_response = initial_response

        while self.budget.can_iterate():
            self.budget.use_iteration()

            # Parse tool calls
            remaining_text, tool_calls = parse_tool_calls(current_response)

            # If no tool calls, check if it was a malformed attempt
            if not tool_calls:
                # Check if response looked like a tool call attempt but failed to parse
                if is_tool_response(current_response):
                    # Malformed JSON - inject correction message and retry
                    await self._emit("tool", "Malformed tool call JSON, requesting correction")
                    _log.warning("Model emitted malformed tool call JSON, requesting retry")

                    # Add assistant's malformed response to history
                    messages.append({
                        "role": "assistant",
                        "content": current_response,
                    })

                    # Inject correction message
                    messages.append({
                        "role": "user",
                        "content": get_json_correction_message(),
                    })

                    # Get corrected response
                    current_response = ""
                    async for token in reflect_stream(
                        messages,
                        model=model,
                        temperature=temperature,
                    ):
                        current_response += token

                    # Continue loop to re-parse the corrected response
                    continue

                # Genuine non-tool response - yield and exit
                if remaining_text:
                    yield remaining_text
                break

            # Enforce per-step limit
            if len(tool_calls) > self.budget.max_calls_per_step:
                await self._emit(
                    "tool",
                    f"Truncating to {self.budget.max_calls_per_step} calls"
                )
                tool_calls = tool_calls[: self.budget.max_calls_per_step]

            # Yield any text before tool calls
            if remaining_text.strip():
                yield remaining_text + "\n"

            # Execute tools
            await self._emit("tool", f"Executing {len(tool_calls)} tool(s)")
            results = await self.execute_batch(tool_calls)

            # Format results for injection
            result_content = format_tool_results(results)

            # Add assistant message with tool calls (for history)
            messages.append({
                "role": "assistant",
                "content": current_response,
            })

            # Add tool results as user message
            # NOTE: We use "user" role because most OpenAI-compatible APIs don't
            # support a dedicated "tool" role. This pattern works but may cause
            # issues with models that treat user messages as higher authority.
            # If tool-loop issues occur, consider "system" role for providers
            # that support it, or the native tool_result format for Anthropic API.
            messages.append({
                "role": "user",
                "content": result_content,
            })

            # Check budget before next iteration
            if self.budget.exceeded():
                yield "\n[Tool execution budget exceeded]"
                break

            # Get next model response
            current_response = ""
            async for token in reflect_stream(
                messages,
                model=model,
                temperature=temperature,
            ):
                current_response += token

        # Yield any remaining response that wasn't yielded in the loop
        # This handles cases where:
        # 1. Loop exited due to can_iterate() returning False
        # 2. Final response was fetched but not yet parsed/yielded
        if current_response:
            remaining, final_calls = parse_tool_calls(current_response)
            if not final_calls and remaining:
                yield remaining

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            "iterations_used": self.budget.iterations_used,
            "iterations_max": self.budget.max_iterations,
            "calls_used": self.budget.calls_used,
            "calls_max": self.budget.max_total_calls,
            "exceeded": self.budget.exceeded(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_orchestrator(
    enabled_policies: Optional[List[PolicyClass]] = None,
    max_iterations: int = 6,
    max_calls_per_step: int = 5,
    emit_event: Optional[Callable] = None,
    gate_filter: Optional[List[str]] = None,
) -> ToolOrchestrator:
    """
    Create a configured orchestrator.

    Args:
        enabled_policies: Policies to enable (default: READ_ONLY only)
        max_iterations: Max loop iterations
        max_calls_per_step: Max calls per step
        emit_event: Event callback
        gate_filter: Optional list of gate names to enable

    Returns:
        Configured ToolOrchestrator
    """
    policy_manager = PolicyManager()

    if enabled_policies:
        for policy in enabled_policies:
            policy_manager.enable_policy(policy)

    budget = ToolBudget(
        max_iterations=max_iterations,
        max_calls_per_step=max_calls_per_step,
    )

    return ToolOrchestrator(
        policy_manager=policy_manager,
        budget=budget,
        emit_event=emit_event,
        enabled_gates=gate_filter,
    )


__all__ = [
    "ToolOrchestrator",
    "create_orchestrator",
]
