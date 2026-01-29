"""
BrowserGate search providers.

Exports provider base class and implementations.
"""

from .base import SearchProviderBase
from .duckduckgo import DuckDuckGoProvider
from .searxng import SearXNGProvider
from .brave import BraveProvider
from ..models import SearchProvider, ProviderConfig

__all__ = [
    "SearchProviderBase",
    "DuckDuckGoProvider",
    "SearXNGProvider",
    "BraveProvider",
    "get_provider",
    "PROVIDERS",
]

# Registry of available providers
PROVIDERS = {
    SearchProvider.DUCKDUCKGO: DuckDuckGoProvider,
    SearchProvider.SEARXNG: SearXNGProvider,
    SearchProvider.BRAVE: BraveProvider,
}


def get_provider(
    provider: SearchProvider,
    config: ProviderConfig = None
) -> SearchProviderBase:
    """
    Factory function to get a search provider instance.

    Args:
        provider: Which provider to use
        config: Provider configuration (uses defaults if None)

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider not implemented
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Provider '{provider.value}' not implemented. "
            f"Available: {[p.value for p in PROVIDERS.keys()]}"
        )

    if config is None:
        config = ProviderConfig()

    return PROVIDERS[provider](config)
