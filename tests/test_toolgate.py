"""
Tests for ToolGate - Cathedral Tool Call Protocol.
"""

import pytest
from cathedral.ToolGate import (
    PolicyClass,
    ToolCall,
    ToolCallBatch,
    ToolResult,
    ToolDefinition,
    ToolBudget,
    ArgSchema,
    ToolRegistry,
    parse_tool_calls,
    validate_args,
    format_tool_results,
    is_tool_response,
    generate_call_id,
    PolicyManager,
    build_tool_prompt,
    build_minimal_tool_prompt,
    get_tool_count,
    initialize,
    is_initialized,
    is_healthy,
    get_health_status,
    list_tools,
    list_tool_names,
    get_tool,
)


class TestModels:
    """Tests for protocol models."""

    def test_tool_call_creation(self):
        """Should create a valid tool call."""
        call = ToolCall(
            id="tc_123",
            tool="MemoryGate.search",
            args={"query": "test"}
        )
        assert call.type == "tool_call"
        assert call.id == "tc_123"
        assert call.tool == "MemoryGate.search"
        assert call.args == {"query": "test"}

    def test_tool_call_batch_creation(self):
        """Should create a batch of tool calls."""
        calls = [
            ToolCall(id="tc_1", tool="MemoryGate.search", args={"query": "a"}),
            ToolCall(id="tc_2", tool="MemoryGate.recall", args={}),
        ]
        batch = ToolCallBatch(calls=calls)
        assert batch.type == "tool_calls"
        assert len(batch) == 2

    def test_tool_result_success(self):
        """Should create a success result."""
        result = ToolResult.success("tc_123", {"data": "value"})
        assert result.ok is True
        assert result.result == {"data": "value"}
        assert result.error is None

    def test_tool_result_failure(self):
        """Should create a failure result."""
        result = ToolResult.failure("tc_123", "Something went wrong")
        assert result.ok is False
        assert result.error == "Something went wrong"
        assert result.result is None

    def test_tool_budget_tracking(self):
        """Should track iterations and calls."""
        budget = ToolBudget(max_iterations=3, max_calls_per_step=2)

        assert budget.can_iterate() is True
        assert budget.can_call(1) is True

        budget.use_iteration()
        budget.use_calls(5)

        assert budget.iterations_used == 1
        assert budget.calls_used == 5

    def test_tool_budget_exceeded(self):
        """Should detect when budget is exceeded."""
        budget = ToolBudget(max_iterations=2, max_total_calls=3)

        budget.use_iteration()
        budget.use_iteration()

        assert budget.exceeded() is True


class TestRegistry:
    """Tests for tool registry."""

    def test_registry_initialize(self):
        """Should initialize with tools from all gates."""
        ToolRegistry.reset()
        ToolRegistry.initialize()

        tools = ToolRegistry.list_tools()
        assert len(tools) > 0

    def test_registry_get_tool(self):
        """Should retrieve tool by name."""
        ToolRegistry.initialize()

        tool = ToolRegistry.get_tool("MemoryGate.search")
        assert tool is not None
        assert tool.name == "MemoryGate.search"
        assert tool.gate == "MemoryGate"
        assert tool.method == "search"

    def test_registry_list_by_policy(self):
        """Should filter tools by policy class."""
        ToolRegistry.initialize()

        read_only = ToolRegistry.list_tools({PolicyClass.READ_ONLY})
        privileged = ToolRegistry.list_tools({PolicyClass.PRIVILEGED})

        # All read-only tools should have READ_ONLY policy
        for tool in read_only:
            assert tool.policy_class == PolicyClass.READ_ONLY

        # All privileged tools should have PRIVILEGED policy
        for tool in privileged:
            assert tool.policy_class == PolicyClass.PRIVILEGED

    def test_registry_list_by_gate(self):
        """Should filter tools by gate."""
        ToolRegistry.initialize()

        memory_tools = ToolRegistry.list_tools(gate_filter="MemoryGate")
        shell_tools = ToolRegistry.list_tools(gate_filter="ShellGate")

        for tool in memory_tools:
            assert tool.gate == "MemoryGate"

        for tool in shell_tools:
            assert tool.gate == "ShellGate"


class TestProtocol:
    """Tests for protocol parsing."""

    def test_parse_single_tool_call(self):
        """Should parse a single tool call from text."""
        text = '''
Here's what I'll do:
{"type": "tool_call", "id": "tc_abc", "tool": "MemoryGate.search", "args": {"query": "test"}}
'''
        remaining, calls = parse_tool_calls(text)

        assert len(calls) == 1
        assert calls[0].id == "tc_abc"
        assert calls[0].tool == "MemoryGate.search"
        assert "query" in calls[0].args

    def test_parse_tool_call_in_code_block(self):
        """Should parse tool call from markdown code block."""
        text = '''
I'll search for that:

```json
{"type": "tool_call", "id": "tc_123", "tool": "MemoryGate.search", "args": {"query": "cathedral"}}
```
'''
        remaining, calls = parse_tool_calls(text)

        assert len(calls) == 1
        assert calls[0].tool == "MemoryGate.search"

    def test_parse_batch_tool_calls(self):
        """Should parse batch of tool calls."""
        text = '''
{"type": "tool_calls", "calls": [
  {"type": "tool_call", "id": "tc_1", "tool": "MemoryGate.search", "args": {"query": "a"}},
  {"type": "tool_call", "id": "tc_2", "tool": "MemoryGate.recall", "args": {}}
]}
'''
        remaining, calls = parse_tool_calls(text)

        assert len(calls) == 2

    def test_parse_no_tool_calls(self):
        """Should return empty list when no tool calls."""
        text = "Just a normal response without any tool calls."
        remaining, calls = parse_tool_calls(text)

        assert len(calls) == 0
        assert remaining == text

    def test_is_tool_response(self):
        """Should detect tool call markers."""
        assert is_tool_response('{"type": "tool_call"') is True
        assert is_tool_response("Normal text") is False

    def test_generate_call_id(self):
        """Should generate unique IDs."""
        id1 = generate_call_id()
        id2 = generate_call_id()

        assert id1.startswith("tc_")
        assert id2.startswith("tc_")
        assert id1 != id2

    def test_validate_args_valid(self):
        """Should validate correct arguments."""
        tool = ToolDefinition(
            name="Test.method",
            description="Test",
            gate="Test",
            method="method",
            args_schema={
                "query": ArgSchema(type="string", required=True),
                "limit": ArgSchema(type="integer", required=False, default=10),
            }
        )

        valid, error = validate_args(tool, {"query": "test"})
        assert valid is True
        assert error is None

    def test_validate_args_missing_required(self):
        """Should reject missing required arguments."""
        tool = ToolDefinition(
            name="Test.method",
            description="Test",
            gate="Test",
            method="method",
            args_schema={
                "query": ArgSchema(type="string", required=True),
            }
        )

        valid, error = validate_args(tool, {})
        assert valid is False
        assert "required" in error.lower()

    def test_validate_args_wrong_type(self):
        """Should reject wrong argument type."""
        tool = ToolDefinition(
            name="Test.method",
            description="Test",
            gate="Test",
            method="method",
            args_schema={
                "limit": ArgSchema(type="integer", required=True),
            }
        )

        valid, error = validate_args(tool, {"limit": "not an integer"})
        assert valid is False
        assert "integer" in error.lower()

    def test_format_tool_results(self):
        """Should format results for injection."""
        results = [
            ToolResult.success("tc_1", {"data": "value"}),
            ToolResult.failure("tc_2", "Error message"),
        ]

        formatted = format_tool_results(results)

        assert "[TOOL RESULTS]" in formatted
        assert "tc_1" in formatted
        assert "SUCCESS" in formatted
        assert "tc_2" in formatted
        assert "FAILED" in formatted
        assert "Error message" in formatted


class TestPolicy:
    """Tests for policy management."""

    def test_default_policy_read_only(self):
        """Should have READ_ONLY enabled by default."""
        manager = PolicyManager()

        assert manager.is_policy_enabled(PolicyClass.READ_ONLY) is True
        assert manager.is_policy_enabled(PolicyClass.WRITE) is False
        assert manager.is_policy_enabled(PolicyClass.PRIVILEGED) is False

    def test_enable_policy(self):
        """Should enable policy classes."""
        manager = PolicyManager()

        # Enable network (always allowed)
        success, error = manager.enable_policy(PolicyClass.NETWORK)
        assert success is True
        assert manager.is_policy_enabled(PolicyClass.NETWORK) is True

    def test_check_tool_policy(self):
        """Should check tool against enabled policies."""
        manager = PolicyManager()

        read_tool = ToolDefinition(
            name="Test.read",
            description="Test",
            gate="Test",
            method="read",
            policy_class=PolicyClass.READ_ONLY,
        )

        write_tool = ToolDefinition(
            name="Test.write",
            description="Test",
            gate="Test",
            method="write",
            policy_class=PolicyClass.WRITE,
        )

        # Read should be allowed
        allowed, reason = manager.check(read_tool, {})
        assert allowed is True

        # Write should be blocked
        allowed, reason = manager.check(write_tool, {})
        assert allowed is False
        assert "not enabled" in reason


class TestPrompt:
    """Tests for prompt building."""

    def test_build_tool_prompt(self):
        """Should build tool prompt with instructions."""
        prompt = build_tool_prompt(
            enabled_policies={PolicyClass.READ_ONLY},
            max_calls=5,
        )

        assert "Available Tools" in prompt
        assert "tool_call" in prompt
        assert "MemoryGate" in prompt

    def test_build_minimal_tool_prompt(self):
        """Should build minimal prompt."""
        prompt = build_minimal_tool_prompt()

        assert "tool_call" in prompt
        # Should be shorter than full prompt
        assert len(prompt) < 2000

    def test_get_tool_count(self):
        """Should return count of available tools."""
        count = get_tool_count()
        assert count > 0

        read_only_count = get_tool_count({PolicyClass.READ_ONLY})
        privileged_count = get_tool_count({PolicyClass.PRIVILEGED})

        # Should have more read-only than privileged
        assert read_only_count > privileged_count


class TestPublicAPI:
    """Tests for public API functions."""

    def test_initialize(self):
        """Should initialize ToolGate."""
        result = initialize()
        assert result is True
        assert is_initialized() is True

    def test_is_healthy(self):
        """Should report health status."""
        initialize()
        assert is_healthy() is True

    def test_get_health_status(self):
        """Should return health status dict."""
        initialize()
        status = get_health_status()

        assert "gate" in status
        assert status["gate"] == "ToolGate"
        assert "healthy" in status
        assert "initialized" in status
        assert "checks" in status

    def test_list_tools(self):
        """Should list available tools."""
        initialize()
        tools = list_tools()

        assert len(tools) > 0
        assert all(isinstance(t, ToolDefinition) for t in tools)

    def test_list_tool_names(self):
        """Should list tool names."""
        initialize()
        names = list_tool_names()

        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
        assert any("MemoryGate" in n for n in names)

    def test_get_tool(self):
        """Should get specific tool."""
        initialize()
        tool = get_tool("MemoryGate.search")

        assert tool is not None
        assert tool.name == "MemoryGate.search"


class TestPromptConfiguration:
    """Tests for prompt configuration system."""

    def test_get_prompt_config(self):
        """Should return prompt configuration."""
        from cathedral.ToolGate import get_prompt_config

        config = get_prompt_config()

        assert "prompt" in config
        assert "is_custom" in config
        assert "current_version" in config
        assert "is_using_fallback" in config

    def test_get_prompt_version(self):
        """Should return prompt version."""
        from cathedral.ToolGate import get_prompt_version, PROMPT_VERSION

        version = get_prompt_version()
        assert version == PROMPT_VERSION
        assert "." in version  # Semver-ish

    def test_validate_prompt_valid(self):
        """Should validate a valid prompt."""
        from cathedral.ToolGate import validate_prompt, DEFAULT_TOOL_PROTOCOL_PROMPT

        valid, missing = validate_prompt(DEFAULT_TOOL_PROTOCOL_PROMPT)

        assert valid is True
        assert len(missing) == 0

    def test_validate_prompt_invalid(self):
        """Should detect missing markers in invalid prompt."""
        from cathedral.ToolGate import validate_prompt

        # Missing required markers
        invalid_prompt = "Just a regular prompt without tool instructions."

        valid, missing = validate_prompt(invalid_prompt)

        assert valid is False
        assert len(missing) > 0
        assert '"type"' in missing

    def test_is_prompt_functional(self):
        """Should check if prompt is functional."""
        from cathedral.ToolGate import is_prompt_functional, DEFAULT_TOOL_PROTOCOL_PROMPT

        assert is_prompt_functional(DEFAULT_TOOL_PROTOCOL_PROMPT) is True
        assert is_prompt_functional("broken prompt") is False

    def test_set_custom_prompt_requires_acknowledgment(self):
        """Should require risk acknowledgment."""
        from cathedral.ToolGate import set_custom_prompt, restore_default_prompt

        # Reset to known state
        restore_default_prompt()

        # Try without acknowledgment
        success, message = set_custom_prompt("new prompt", acknowledge_risk=False)

        assert success is False
        assert "acknowledge" in message.lower()

    def test_set_custom_prompt_validates(self):
        """Should validate prompt before saving."""
        from cathedral.ToolGate import set_custom_prompt, restore_default_prompt

        # Reset to known state
        restore_default_prompt()

        # Try with invalid prompt (missing markers)
        success, message = set_custom_prompt(
            "This prompt has no tool markers.",
            acknowledge_risk=True
        )

        assert success is False
        assert "missing" in message.lower()

    def test_set_and_restore_custom_prompt(self):
        """Should set custom prompt and restore default."""
        from cathedral.ToolGate import (
            set_custom_prompt,
            restore_default_prompt,
            get_prompt_config,
            DEFAULT_TOOL_PROTOCOL_PROMPT,
        )

        # Reset first
        restore_default_prompt()

        # Create a valid custom prompt
        custom_prompt = DEFAULT_TOOL_PROTOCOL_PROMPT + "\n\n## Custom Section\nThis is custom."

        success, message = set_custom_prompt(custom_prompt, acknowledge_risk=True)

        assert success is True

        config = get_prompt_config()
        assert config["is_custom"] is True
        assert "Custom Section" in config["prompt"]

        # Restore default
        success, message = restore_default_prompt()

        assert success is True

        config = get_prompt_config()
        assert config["is_custom"] is False
        assert "Custom Section" not in config["prompt"]

    def test_get_edit_warning(self):
        """Should return warning message."""
        from cathedral.ToolGate import get_edit_warning

        warning = get_edit_warning()

        assert "WARNING" in warning
        assert "break" in warning.lower() or "risk" in warning.lower()

    def test_default_prompt_has_required_elements(self):
        """Default prompt should have all required elements."""
        from cathedral.ToolGate import DEFAULT_TOOL_PROTOCOL_PROMPT

        # Must have JSON format examples
        assert '"type"' in DEFAULT_TOOL_PROTOCOL_PROMPT
        assert '"tool_call"' in DEFAULT_TOOL_PROTOCOL_PROMPT
        assert '"id"' in DEFAULT_TOOL_PROTOCOL_PROMPT
        assert '"tool"' in DEFAULT_TOOL_PROTOCOL_PROMPT
        assert '"args"' in DEFAULT_TOOL_PROTOCOL_PROMPT

        # Must have result format
        assert "TOOL RESULTS" in DEFAULT_TOOL_PROTOCOL_PROMPT

        # Should explain when to use tools
        assert "when" in DEFAULT_TOOL_PROTOCOL_PROMPT.lower()
