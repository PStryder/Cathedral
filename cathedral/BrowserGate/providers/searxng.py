"""
SearXNG search provider.

SearXNG is a self-hosted, privacy-respecting metasearch engine.
Requires a configured SearXNG instance URL.
"""

import httpx
from typing import List, Optional
from .base import SearchProviderBase
from ..models import SearchResult, ProviderConfig


class SearXNGProvider(SearchProviderBase):
    """SearXNG search provider for self-hosted search."""

    DEFAULT_TIMEOUT = 30

    @property
    def name(self) -> str:
        return "searxng"

    def is_available(self) -> bool:
        """Check if provider is available (requires base_url)."""
        return self.config.enabled and bool(self.config.base_url)

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        categories: Optional[str] = None,
        engines: Optional[str] = None,
        language: str = "en",
        safesearch: int = 1,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search using SearXNG instance.

        Args:
            query: Search query
            max_results: Max results (default from config)
            categories: Comma-separated categories (general, images, news, etc.)
            engines: Comma-separated engines to use
            language: Language code (en, de, fr, etc.)
            safesearch: 0=off, 1=moderate, 2=strict
        """
        if not self.config.base_url:
            raise ValueError(
                "SearXNG requires base_url in config. "
                "Set to your SearXNG instance URL (e.g., https://searx.example.com)"
            )

        if max_results is None:
            max_results = self.config.max_results

        # Build request URL
        base_url = self.config.base_url.rstrip("/")
        search_url = f"{base_url}/search"

        params = {
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": safesearch,
        }

        if categories:
            params["categories"] = categories
        if engines:
            params["engines"] = engines

        timeout = self.config.timeout_seconds or self.DEFAULT_TIMEOUT

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        raw_results = data.get("results", [])[:max_results]

        for i, r in enumerate(raw_results):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                position=i + 1,
                source=f"searxng:{r.get('engine', 'unknown')}"
            ))

        return results
