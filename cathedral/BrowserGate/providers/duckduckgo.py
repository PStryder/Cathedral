"""
DuckDuckGo search provider.

Uses the duckduckgo-search library for privacy-focused searching.
"""

import asyncio
from typing import List, Optional
from .base import SearchProviderBase
from ..models import SearchResult, ProviderConfig


class DuckDuckGoProvider(SearchProviderBase):
    """DuckDuckGo search provider."""

    @property
    def name(self) -> str:
        return "duckduckgo"

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        **kwargs
    ) -> List[SearchResult]:
        """
        Search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Max results (default from config)
            region: Region code (wt-wt = no region)
            safesearch: off, moderate, strict
        """
        if max_results is None:
            max_results = self.config.max_results

        # Run in thread pool since duckduckgo-search is sync
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._sync_search(query, max_results, region, safesearch)
        )
        return results

    def _sync_search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str
    ) -> List[SearchResult]:
        """Synchronous search implementation."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search not installed. "
                "Install with: pip install duckduckgo-search"
            )

        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results
            )

            for i, r in enumerate(search_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                    position=i + 1,
                    source="duckduckgo"
                ))

        return results
