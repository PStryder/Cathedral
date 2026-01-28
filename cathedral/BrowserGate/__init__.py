"""
BrowserGate - Web search and page fetching for Cathedral agents.

Provides local browser automation capabilities:
- Web search via multiple providers (DuckDuckGo, SearXNG, Brave)
- Page content fetching (simple or headless mode)
- Content conversion to markdown/text/html
- Browser extension integration via WebSocket

Usage:
    from cathedral.BrowserGate import search, fetch

    # Search the web
    results = await search("python asyncio tutorial")

    # Fetch page content
    page = await fetch("https://example.com")
    print(page.content)

    # Start WebSocket server for browser extension
    from cathedral.BrowserGate import start_extension_server
    await start_extension_server()
"""

import time
from typing import Optional, List
from .models import (
    SearchProvider,
    FetchMode,
    ContentFormat,
    SearchResult,
    SearchResponse,
    PageContent,
    ProviderConfig,
    BrowserConfig,
)
from .providers import get_provider, PROVIDERS
from .fetcher import PageFetcher, get_fetcher
from .websocket_server import (
    WebSocketServer,
    ExtensionHandler,
    get_server,
    start_server as start_extension_server,
    stop_server as stop_extension_server,
)
from .models import ExtensionMessage, ExtensionResponse

__all__ = [
    # Models
    "SearchProvider",
    "FetchMode",
    "ContentFormat",
    "SearchResult",
    "SearchResponse",
    "PageContent",
    "ProviderConfig",
    "BrowserConfig",
    "ExtensionMessage",
    "ExtensionResponse",
    # Main class
    "BrowserGate",
    "get_browser",
    # Convenience functions
    "search",
    "fetch",
    # WebSocket server
    "WebSocketServer",
    "ExtensionHandler",
    "get_server",
    "start_extension_server",
    "stop_extension_server",
]


class BrowserGate:
    """
    Main interface for web browsing capabilities.

    Combines search and fetch functionality with configurable
    providers and modes.
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        self.config = config or BrowserConfig()
        self._fetcher = PageFetcher(
            default_mode=self.config.default_fetch_mode,
            default_format=self.config.default_format,
            max_content_length=self.config.max_content_length,
        )

    async def search(
        self,
        query: str,
        provider: Optional[SearchProvider] = None,
        max_results: Optional[int] = None,
        **kwargs
    ) -> SearchResponse:
        """
        Search the web.

        Args:
            query: Search query
            provider: Search provider (default from config)
            max_results: Maximum results to return
            **kwargs: Provider-specific options

        Returns:
            SearchResponse with results
        """
        provider = provider or self.config.default_provider
        provider_config = self._get_provider_config(provider)

        if max_results is None:
            max_results = provider_config.max_results

        start_time = time.time()

        search_provider = get_provider(provider, provider_config)
        results = await search_provider.search(query, max_results=max_results, **kwargs)

        search_time_ms = int((time.time() - start_time) * 1000)

        return SearchResponse(
            query=query,
            provider=provider,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms,
        )

    async def fetch(
        self,
        url: str,
        mode: Optional[FetchMode] = None,
        output_format: Optional[ContentFormat] = None,
        **kwargs
    ) -> PageContent:
        """
        Fetch page content.

        Args:
            url: URL to fetch
            mode: Fetch mode (SIMPLE or HEADLESS)
            output_format: Output format (MARKDOWN, TEXT, HTML)
            **kwargs: Additional fetch options

        Returns:
            PageContent with fetched data
        """
        return await self._fetcher.fetch(
            url,
            mode=mode,
            output_format=output_format,
            **kwargs
        )

    async def search_and_fetch(
        self,
        query: str,
        fetch_top: int = 1,
        provider: Optional[SearchProvider] = None,
        fetch_mode: Optional[FetchMode] = None,
        **kwargs
    ) -> tuple[SearchResponse, List[PageContent]]:
        """
        Search and fetch top results.

        Args:
            query: Search query
            fetch_top: Number of top results to fetch
            provider: Search provider
            fetch_mode: Fetch mode for pages

        Returns:
            Tuple of (SearchResponse, list of PageContent)
        """
        response = await self.search(query, provider=provider, **kwargs)

        pages = []
        for result in response.results[:fetch_top]:
            try:
                page = await self.fetch(result.url, mode=fetch_mode)
                pages.append(page)
            except Exception as e:
                # Create error placeholder
                pages.append(PageContent(
                    url=result.url,
                    title=result.title,
                    content=f"[Fetch failed: {e}]",
                    format=self.config.default_format,
                    fetch_mode=fetch_mode or self.config.default_fetch_mode,
                ))

        return response, pages

    def _get_provider_config(self, provider: SearchProvider) -> ProviderConfig:
        """Get config for a specific provider."""
        provider_name = provider.value
        if provider_name in self.config.providers:
            return self.config.providers[provider_name]
        return ProviderConfig()

    def list_providers(self) -> List[str]:
        """List available search providers."""
        return [p.value for p in PROVIDERS.keys()]


# Global instance
_browser: Optional[BrowserGate] = None


def get_browser(config: Optional[BrowserConfig] = None) -> BrowserGate:
    """Get or create global BrowserGate instance."""
    global _browser
    if _browser is None or config is not None:
        _browser = BrowserGate(config)
    return _browser


# Convenience functions
async def search(query: str, **kwargs) -> SearchResponse:
    """Search the web. See BrowserGate.search() for options."""
    return await get_browser().search(query, **kwargs)


async def fetch(url: str, **kwargs) -> PageContent:
    """Fetch page content. See BrowserGate.fetch() for options."""
    return await get_browser().fetch(url, **kwargs)


async def search_and_fetch(query: str, **kwargs) -> tuple[SearchResponse, List[PageContent]]:
    """Search and fetch top results. See BrowserGate.search_and_fetch() for options."""
    return await get_browser().search_and_fetch(query, **kwargs)
