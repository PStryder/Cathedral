"""
Base search provider interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..models import SearchResult, ProviderConfig


class SearchProviderBase(ABC):
    """Abstract base class for search providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Execute a search query.

        Args:
            query: Search query string
            max_results: Maximum results to return (default from config)
            **kwargs: Provider-specific options

        Returns:
            List of SearchResult objects
        """
        pass

    def is_available(self) -> bool:
        """Check if provider is available and configured."""
        return self.config.enabled
