"""
Tests for BrowserGate search providers.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cathedral.BrowserGate.models import (
    SearchProvider,
    SearchResult,
    ProviderConfig,
)
from cathedral.BrowserGate.providers import (
    get_provider,
    PROVIDERS,
    DuckDuckGoProvider,
    SearXNGProvider,
    BraveProvider,
)


class TestProviderRegistry:
    """Test provider registration and factory."""

    def test_all_providers_registered(self):
        """All SearchProvider enum values should have implementations."""
        assert SearchProvider.DUCKDUCKGO in PROVIDERS
        assert SearchProvider.SEARXNG in PROVIDERS
        assert SearchProvider.BRAVE in PROVIDERS

    def test_get_provider_duckduckgo(self):
        """Should return DuckDuckGo provider instance."""
        provider = get_provider(SearchProvider.DUCKDUCKGO)
        assert isinstance(provider, DuckDuckGoProvider)
        assert provider.name == "duckduckgo"

    def test_get_provider_searxng(self):
        """Should return SearXNG provider instance."""
        config = ProviderConfig(base_url="https://searx.example.com")
        provider = get_provider(SearchProvider.SEARXNG, config)
        assert isinstance(provider, SearXNGProvider)
        assert provider.name == "searxng"

    def test_get_provider_brave(self):
        """Should return Brave provider instance."""
        config = ProviderConfig(api_key="test-api-key")
        provider = get_provider(SearchProvider.BRAVE, config)
        assert isinstance(provider, BraveProvider)
        assert provider.name == "brave"

    def test_get_provider_uses_default_config(self):
        """Should use default config if none provided."""
        provider = get_provider(SearchProvider.DUCKDUCKGO)
        assert provider.config.max_results == 10
        assert provider.config.enabled is True


class TestDuckDuckGoProvider:
    """Tests for DuckDuckGo search provider."""

    def test_provider_name(self):
        """Provider should have correct name."""
        provider = DuckDuckGoProvider(ProviderConfig())
        assert provider.name == "duckduckgo"

    def test_is_available_when_enabled(self):
        """Provider should be available when enabled."""
        provider = DuckDuckGoProvider(ProviderConfig(enabled=True))
        assert provider.is_available() is True

    def test_is_not_available_when_disabled(self):
        """Provider should not be available when disabled."""
        provider = DuckDuckGoProvider(ProviderConfig(enabled=False))
        assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Search should return list of SearchResult objects."""
        provider = DuckDuckGoProvider(ProviderConfig())

        mock_results = [
            {"title": "Test 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Test 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]

        with patch.object(provider, '_sync_search', return_value=[
            SearchResult(title="Test 1", url="https://example.com/1", snippet="Snippet 1", position=1, source="duckduckgo"),
            SearchResult(title="Test 2", url="https://example.com/2", snippet="Snippet 2", position=2, source="duckduckgo"),
        ]):
            results = await provider.search("test query", max_results=2)

            assert len(results) == 2
            assert all(isinstance(r, SearchResult) for r in results)
            assert results[0].title == "Test 1"
            assert results[0].source == "duckduckgo"


class TestSearXNGProvider:
    """Tests for SearXNG search provider."""

    def test_provider_name(self):
        """Provider should have correct name."""
        provider = SearXNGProvider(ProviderConfig(base_url="https://searx.example.com"))
        assert provider.name == "searxng"

    def test_is_available_requires_base_url(self):
        """Provider should not be available without base_url."""
        provider = SearXNGProvider(ProviderConfig(enabled=True))
        assert provider.is_available() is False

    def test_is_available_with_base_url(self):
        """Provider should be available with base_url."""
        provider = SearXNGProvider(ProviderConfig(
            enabled=True,
            base_url="https://searx.example.com"
        ))
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_search_requires_base_url(self):
        """Search should raise error without base_url."""
        provider = SearXNGProvider(ProviderConfig())

        with pytest.raises(ValueError, match="requires base_url"):
            await provider.search("test query")

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Search should return properly formatted results."""
        provider = SearXNGProvider(ProviderConfig(
            base_url="https://searx.example.com"
        ))

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1", "engine": "google"},
                {"title": "Result 2", "url": "https://example.com/2", "content": "Content 2", "engine": "bing"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            results = await provider.search("test query", max_results=2)

            assert len(results) == 2
            assert results[0].title == "Result 1"
            assert results[0].source == "searxng:google"
            assert results[1].source == "searxng:bing"


class TestBraveProvider:
    """Tests for Brave Search provider."""

    def test_provider_name(self):
        """Provider should have correct name."""
        provider = BraveProvider(ProviderConfig(api_key="test-key"))
        assert provider.name == "brave"

    def test_is_available_requires_api_key(self):
        """Provider should not be available without api_key."""
        provider = BraveProvider(ProviderConfig(enabled=True))
        assert provider.is_available() is False

    def test_is_available_with_api_key(self):
        """Provider should be available with api_key."""
        provider = BraveProvider(ProviderConfig(
            enabled=True,
            api_key="test-api-key"
        ))
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_search_requires_api_key(self):
        """Search should raise error without api_key."""
        provider = BraveProvider(ProviderConfig())

        with pytest.raises(ValueError, match="requires api_key"):
            await provider.search("test query")

    @pytest.mark.asyncio
    async def test_search_caps_results_at_20(self):
        """Search should cap max_results at 20 (free tier limit)."""
        provider = BraveProvider(ProviderConfig(api_key="test-key"))

        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            await provider.search("test", max_results=50)

            # Verify the count parameter was capped at 20
            call_kwargs = mock_instance.get.call_args
            assert call_kwargs[1]["params"]["count"] == 20

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Search should return properly formatted results."""
        provider = BraveProvider(ProviderConfig(api_key="test-key"))

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Brave Result", "url": "https://example.com", "description": "Description"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            results = await provider.search("test query")

            assert len(results) == 1
            assert results[0].title == "Brave Result"
            assert results[0].source == "brave"


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Should create SearchResult with all fields."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            position=1,
            source="test"
        )

        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.position == 1
        assert result.source == "test"

    def test_search_result_defaults(self):
        """SearchResult should have sensible defaults."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet",
            position=1
        )

        assert result.source == ""
