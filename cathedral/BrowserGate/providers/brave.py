"""
Brave Search provider.

Uses the Brave Search API for privacy-focused web search.
Requires an API key from https://brave.com/search/api/
"""

import httpx
from typing import List, Optional
from .base import SearchProviderBase
from ..models import SearchResult, ProviderConfig


class BraveProvider(SearchProviderBase):
    """Brave Search API provider."""

    API_BASE = "https://api.search.brave.com/res/v1"
    DEFAULT_TIMEOUT = 30

    @property
    def name(self) -> str:
        return "brave"

    def is_available(self) -> bool:
        """Check if provider is available (requires api_key)."""
        return self.config.enabled and bool(self.config.api_key)

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        country: str = "us",
        search_lang: str = "en",
        safesearch: str = "moderate",
        freshness: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search using Brave Search API.

        Args:
            query: Search query
            max_results: Max results (default from config, max 20 for free tier)
            country: Country code for results
            search_lang: Search language
            safesearch: off, moderate, strict
            freshness: Time filter: pd (past day), pw (past week), pm (past month), py (past year)
        """
        if not self.config.api_key:
            raise ValueError(
                "Brave Search requires api_key in config. "
                "Get one at https://brave.com/search/api/"
            )

        if max_results is None:
            max_results = self.config.max_results

        # Brave free tier caps at 20 results
        max_results = min(max_results, 20)

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.config.api_key,
        }

        params = {
            "q": query,
            "count": max_results,
            "country": country,
            "search_lang": search_lang,
            "safesearch": safesearch,
        }

        if freshness:
            params["freshness"] = freshness

        timeout = self.config.timeout_seconds or self.DEFAULT_TIMEOUT

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{self.API_BASE}/web/search",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()

        results = []
        web_results = data.get("web", {}).get("results", [])

        for i, r in enumerate(web_results):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                position=i + 1,
                source="brave"
            ))

        return results
