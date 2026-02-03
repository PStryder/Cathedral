"""
Tests for OpenClaw Agent StarMirror provider.

Tests the hooks-based agent endpoint integration for hybrid Cathedral+OpenClaw flow.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock


class TestOpenClawAgentImport:
    """Test that OpenClaw Agent provider can be imported."""

    def test_import_openclaw_agent_module(self):
        """Should import openclaw_agent provider module."""
        from cathedral.StarMirror.providers import openclaw_agent
        assert hasattr(openclaw_agent, "stream")
        assert hasattr(openclaw_agent, "transmit")
        assert hasattr(openclaw_agent, "transmit_async")
        assert hasattr(openclaw_agent, "HANDLES_TOOLS_INTERNALLY")

    def test_openclaw_agent_in_supported_backends(self):
        """OpenClaw Agent should be in SUPPORTED_BACKENDS."""
        from cathedral.StarMirror.router import SUPPORTED_BACKENDS
        assert "openclaw_agent" in SUPPORTED_BACKENDS

    def test_handles_tools_internally_flag(self):
        """Should have HANDLES_TOOLS_INTERNALLY = True."""
        from cathedral.StarMirror.providers import openclaw_agent
        assert openclaw_agent.HANDLES_TOOLS_INTERNALLY is True


class TestOpenClawAgentConfiguration:
    """Test OpenClaw Agent configuration defaults."""

    def test_default_agent_url(self):
        """Should have correct default agent URL."""
        from cathedral.StarMirror.providers import openclaw_agent
        # Default should be hooks/agent endpoint
        assert "hooks/agent" in openclaw_agent.API_URL

    def test_default_session_key(self):
        """Should have default session key."""
        from cathedral.StarMirror.providers import openclaw_agent
        assert openclaw_agent.DEFAULT_SESSION_KEY == "cathedral"

    def test_agent_timeout_longer_than_chat(self):
        """Agent timeout should be longer than standard chat (tools take time)."""
        from cathedral.StarMirror.providers import openclaw_agent
        assert openclaw_agent.AGENT_TIMEOUT >= 120  # At least 2 minutes


class TestContextExtraction:
    """Test extraction of user message and context from Cathedral's message array."""

    def test_extract_simple_message(self):
        """Should extract user message from simple conversation."""
        from cathedral.StarMirror.providers.openclaw_agent import _extract_user_message_and_context

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        user_msg, context = _extract_user_message_and_context(messages)
        assert user_msg == "What is 2+2?"
        assert "You are helpful." in context

    def test_extract_with_cathedral_context(self):
        """Should extract Cathedral's injected context blocks."""
        from cathedral.StarMirror.providers.openclaw_agent import _extract_user_message_and_context

        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "system", "content": "[Relevant Documents]\nDoc about X..."},
            {"role": "system", "content": "[Memory Context]\nFact about Y..."},
            {"role": "assistant", "content": "Previous response"},
            {"role": "user", "content": "Follow-up question"},
        ]

        user_msg, context = _extract_user_message_and_context(messages)
        assert user_msg == "Follow-up question"
        assert "[Relevant Documents]" in context
        assert "[Memory Context]" in context
        assert "[ASSISTANT]:" in context

    def test_extract_empty_messages(self):
        """Should handle empty message list."""
        from cathedral.StarMirror.providers.openclaw_agent import _extract_user_message_and_context

        user_msg, context = _extract_user_message_and_context([])
        assert user_msg == ""
        assert context == ""

    def test_extract_multimodal_content(self):
        """Should handle multimodal content (extract text parts)."""
        from cathedral.StarMirror.providers.openclaw_agent import _extract_user_message_and_context

        messages = [
            {"role": "system", "content": "Vision model."},
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]},
        ]

        user_msg, context = _extract_user_message_and_context(messages)
        assert user_msg == "What's in this image?"


class TestAgentPayloadBuilding:
    """Test payload building for /hooks/agent endpoint."""

    def test_build_payload_basic(self):
        """Should build basic agent payload."""
        from cathedral.StarMirror.providers.openclaw_agent import _build_agent_payload

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]

        payload = _build_agent_payload(messages, session_key="test-session")
        assert payload["message"] == "Hello"
        assert payload["sessionKey"] == "test-session"
        assert payload["stream"] is True
        assert "context" in payload

    def test_build_payload_with_label(self):
        """Should include agent label if provided."""
        from cathedral.StarMirror.providers.openclaw_agent import _build_agent_payload

        messages = [{"role": "user", "content": "Hi"}]
        payload = _build_agent_payload(messages, agent_label="assistant")
        assert payload["label"] == "assistant"


class TestRouterIntegration:
    """Test that OpenClaw Agent integrates with StarMirror router."""

    def test_router_can_load_openclaw_agent(self):
        """Router should be able to load OpenClaw Agent provider."""
        from cathedral.StarMirror.router import _load_provider
        provider = _load_provider("openclaw_agent")
        assert hasattr(provider, "stream")
        assert hasattr(provider, "HANDLES_TOOLS_INTERNALLY")

    def test_backend_handles_tools_detection(self):
        """Router should detect that openclaw_agent handles tools internally."""
        from cathedral.StarMirror import router
        with patch.object(router, "get_backend", return_value="openclaw_agent"):
            assert router.backend_handles_tools() is True

    def test_regular_backend_no_internal_tools(self):
        """Router should detect that openrouter does NOT handle tools internally."""
        from cathedral.StarMirror import router
        with patch.object(router, "get_backend", return_value="openrouter"):
            assert router.backend_handles_tools() is False


class TestMessageValidation:
    """Test message format validation."""

    def test_empty_messages_raises(self):
        """Should raise on empty message list."""
        from cathedral.StarMirror.providers import openclaw_agent

        with pytest.raises(ValueError, match="empty or invalid"):
            openclaw_agent.transmit([])

    @pytest.mark.asyncio
    async def test_stream_empty_messages_raises(self):
        """Stream should raise on empty message list."""
        from cathedral.StarMirror.providers import openclaw_agent

        with pytest.raises(ValueError, match="empty or invalid"):
            async for _ in openclaw_agent.stream([]):
                pass


class TestSSEParsing:
    """Test SSE parsing for agent responses."""

    @pytest.mark.asyncio
    async def test_stream_parses_multiple_formats(self):
        """Should parse multiple response formats from OpenClaw."""
        from cathedral.StarMirror.providers import openclaw_agent
        import httpx

        # Test different response formats OpenClaw might use
        sse_lines = [
            'data: {"content": "Hello"}',  # Direct content
            'data: {"text": " from"}',  # Text field
            'data: {"choices": [{"delta": {"content": " agent"}}]}',  # OpenAI format
            'data: [DONE]',
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        class MockStreamContext:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                pass

        class MockClient:
            def stream(self, *args, **kwargs):
                return MockStreamContext()

        class MockAsyncClient:
            async def __aenter__(self):
                return MockClient()
            async def __aexit__(self, *args):
                pass

        with patch.object(httpx, "AsyncClient", return_value=MockAsyncClient()):
            messages = [{"role": "user", "content": "Hi"}]
            chunks = []
            async for chunk in openclaw_agent.stream(messages):
                chunks.append(chunk)

            assert "Hello" in chunks
            assert " from" in chunks
            assert " agent" in chunks


# Example configuration for hybrid Cathedral + OpenClaw flow
EXAMPLE_HYBRID_CONFIG = """
# .env configuration for hybrid Cathedral + OpenClaw agent flow

# Use OpenClaw Agent backend (Cathedral injects context, OpenClaw runs tools)
LLM_BACKEND=openclaw_agent

# OpenClaw Agent endpoint
OPENCLAW_AGENT_URL=http://127.0.0.1:18789/hooks/agent

# Session key for OpenClaw
OPENCLAW_SESSION_KEY=cathedral-hybrid

# Optional auth token (for remote gateways)
# OPENCLAW_TOKEN=your-token-here

# Enable context injection (Cathedral does RAG, OpenClaw does tools)
# In API call: enable_context=true, enable_tools=true (but Cathedral skips ToolGate)

# Optional: Import OpenClaw's memory files into Cathedral's ScriptureGate
SCRIPTURE_OPENCLAW_ENABLED=true
OPENCLAW_WORKSPACE_PATHS=/path/to/openclaw/workspace
"""


class TestExampleConfig:
    """Verify example configuration is valid."""

    def test_example_hybrid_config_parseable(self):
        """Example config should be parseable."""
        lines = EXAMPLE_HYBRID_CONFIG.strip().split("\n")
        config = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key] = value

        assert config.get("LLM_BACKEND") == "openclaw_agent"
        assert "hooks/agent" in config.get("OPENCLAW_AGENT_URL", "")
        assert config.get("SCRIPTURE_OPENCLAW_ENABLED") == "true"
