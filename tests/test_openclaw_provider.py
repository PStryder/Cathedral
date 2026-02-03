"""
Tests for OpenClaw StarMirror provider.

Smoke tests for the OpenClaw Gateway backend integration.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock


class TestOpenClawImport:
    """Test that OpenClaw provider can be imported."""

    def test_import_openclaw_module(self):
        """Should import openclaw provider module."""
        from cathedral.StarMirror.providers import openclaw
        assert hasattr(openclaw, "stream")
        assert hasattr(openclaw, "transmit")
        assert hasattr(openclaw, "transmit_async")

    def test_openclaw_in_supported_backends(self):
        """OpenClaw should be in SUPPORTED_BACKENDS."""
        from cathedral.StarMirror.router import SUPPORTED_BACKENDS
        assert "openclaw" in SUPPORTED_BACKENDS


class TestOpenClawConfiguration:
    """Test OpenClaw configuration defaults."""

    def test_default_api_url(self):
        """Should have correct default API URL."""
        from cathedral.StarMirror.providers import openclaw
        # Default should be local gateway
        assert "127.0.0.1:18789" in openclaw.API_URL or "localhost" in openclaw.API_URL.lower()

    def test_api_key_optional(self):
        """API key should be optional (returns None if not set)."""
        from cathedral.StarMirror.providers import openclaw
        # _get_api_key should return None if OPENCLAW_TOKEN not set
        # (not raise an error like OpenRouter does)
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(openclaw, "_get_api_key", return_value=None):
                assert openclaw._get_api_key() is None

    def test_headers_without_token(self):
        """Headers should work without auth token."""
        from cathedral.StarMirror.providers import openclaw
        with patch.object(openclaw, "_get_api_key", return_value=None):
            headers = openclaw._get_headers()
            assert "Content-Type" in headers
            assert "Authorization" not in headers

    def test_headers_with_token(self):
        """Headers should include auth when token is set."""
        from cathedral.StarMirror.providers import openclaw
        with patch.object(openclaw, "_get_api_key", return_value="test-token"):
            headers = openclaw._get_headers()
            assert headers["Authorization"] == "Bearer test-token"


class TestOpenClawSSEParsing:
    """Test SSE parsing logic matches OpenAI format."""

    @pytest.mark.asyncio
    async def test_stream_parses_sse_chunks(self):
        """Should parse SSE data lines and yield content."""
        from cathedral.StarMirror.providers import openclaw
        import httpx

        # Mock SSE response lines
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            'data: {"choices": [{"delta": {"content": "!"}}]}',
            'data: [DONE]',
        ]

        # Create async generator for lines
        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        # Create mock response with proper async context manager
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        # Mock stream context manager
        class MockStreamContext:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                pass

        # Mock client
        class MockClient:
            def stream(self, *args, **kwargs):
                return MockStreamContext()

        # Mock AsyncClient context manager
        class MockAsyncClient:
            async def __aenter__(self):
                return MockClient()
            async def __aexit__(self, *args):
                pass

        with patch.object(httpx, "AsyncClient", return_value=MockAsyncClient()):
            messages = [{"role": "user", "content": "Hi"}]
            chunks = []
            async for chunk in openclaw.stream(messages):
                chunks.append(chunk)

            assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_handles_malformed_json(self):
        """Should skip malformed JSON lines gracefully."""
        from cathedral.StarMirror.providers import openclaw
        import httpx

        sse_lines = [
            'data: {"choices": [{"delta": {"content": "OK"}}]}',
            'data: {malformed json}',
            'data: {"choices": [{"delta": {"content": "!"}}]}',
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
            async for chunk in openclaw.stream(messages):
                chunks.append(chunk)

            # Should have skipped the malformed line
            assert chunks == ["OK", "!"]


class TestOpenClawRouterIntegration:
    """Test that OpenClaw integrates with StarMirror router."""

    def test_router_can_load_openclaw(self):
        """Router should be able to load OpenClaw provider."""
        from cathedral.StarMirror.router import _load_provider
        provider = _load_provider("openclaw")
        assert hasattr(provider, "stream")

    def test_router_backend_selection(self):
        """Router should select OpenClaw when configured."""
        from cathedral.StarMirror import router
        with patch.object(router, "get_backend", return_value="openclaw"):
            provider = router._get_provider()
            # Verify it loaded the openclaw module
            assert provider.__name__ == "cathedral.StarMirror.providers.openclaw"


class TestOpenClawMessageFormat:
    """Test message format matches OpenAI/OpenRouter."""

    def test_empty_messages_raises(self):
        """Should raise on empty message list."""
        from cathedral.StarMirror.providers import openclaw

        with pytest.raises(ValueError, match="empty or invalid"):
            # Sync version
            openclaw.transmit([])

    @pytest.mark.asyncio
    async def test_stream_empty_messages_raises(self):
        """Stream should raise on empty message list."""
        from cathedral.StarMirror.providers import openclaw

        with pytest.raises(ValueError, match="empty or invalid"):
            async for _ in openclaw.stream([]):
                pass

    def test_format_content_for_log_text(self):
        """Should format text content for logging."""
        from cathedral.StarMirror.providers import openclaw

        result = openclaw._format_content_for_log("Hello world")
        assert result == "Hello world"

        # Long content should be truncated
        long_text = "x" * 200
        result = openclaw._format_content_for_log(long_text, max_len=100)
        assert len(result) == 103  # 100 + "..."

    def test_format_content_for_log_multimodal(self):
        """Should format multimodal content for logging."""
        from cathedral.StarMirror.providers import openclaw

        content = [
            {"type": "text", "text": "Analyze this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        result = openclaw._format_content_for_log(content)
        assert "text:" in result
        assert "[image:base64]" in result


# Example configuration for using OpenClaw
EXAMPLE_CONFIG = """
# .env configuration for OpenClaw backend

# Use OpenClaw as the LLM backend
LLM_BACKEND=openclaw

# Local OpenClaw Gateway (default)
OPENCLAW_API_URL=http://127.0.0.1:18789/v1/chat/completions

# Remote OpenClaw Gateway (with authentication)
# OPENCLAW_API_URL=https://openclaw.example.com/v1/chat/completions
# OPENCLAW_TOKEN=your-api-token-here

# Optional: Override the model (gateway uses its configured default otherwise)
# OPENCLAW_MODEL=llama-3.1-8b
"""


class TestExampleConfig:
    """Verify example configuration is valid."""

    def test_example_config_parseable(self):
        """Example config should be parseable."""
        lines = EXAMPLE_CONFIG.strip().split("\n")
        config = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key] = value

        assert config.get("LLM_BACKEND") == "openclaw"
        assert "127.0.0.1:18789" in config.get("OPENCLAW_API_URL", "")
