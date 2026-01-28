"""
BrowserGate data models.

Pydantic models for search, page fetching, and browser configuration.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SearchProvider(str, Enum):
    """Available search providers."""
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    BRAVE = "brave"


class FetchMode(str, Enum):
    """Page fetching mode."""
    SIMPLE = "simple"      # requests + BeautifulSoup
    HEADLESS = "headless"  # Playwright for JS-rendered content


class ContentFormat(str, Enum):
    """Output content format."""
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"


class SearchResult(BaseModel):
    """Individual search result."""
    title: str
    url: str
    snippet: str
    position: int
    source: str = ""  # Additional source info


class SearchResponse(BaseModel):
    """Search response with results."""
    query: str
    provider: SearchProvider
    results: List[SearchResult]
    total_results: Optional[int] = None
    search_time_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PageContent(BaseModel):
    """Fetched page content."""
    url: str
    title: str
    content: str
    format: ContentFormat
    fetch_mode: FetchMode
    word_count: int = 0
    fetch_time_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for a search provider."""
    enabled: bool = True
    api_key: Optional[str] = None  # For providers that need it (Brave)
    base_url: Optional[str] = None  # For self-hosted (SearXNG)
    timeout_seconds: int = 30
    max_results: int = 10


class BrowserConfig(BaseModel):
    """BrowserGate configuration."""
    default_provider: SearchProvider = SearchProvider.DUCKDUCKGO
    default_fetch_mode: FetchMode = FetchMode.SIMPLE
    default_format: ContentFormat = ContentFormat.MARKDOWN

    # Provider-specific configs
    providers: Dict[str, ProviderConfig] = Field(default_factory=lambda: {
        "duckduckgo": ProviderConfig(),
        "searxng": ProviderConfig(enabled=False),
        "brave": ProviderConfig(enabled=False)
    })

    # Headless browser settings
    headless_timeout_ms: int = 30000
    headless_wait_for_idle: bool = True

    # Content processing
    max_content_length: int = 100000  # Max chars to return
    include_images: bool = False
    include_links: bool = True

    # WebSocket settings
    websocket_enabled: bool = True
    websocket_port: int = 8765


class ExtensionMessage(BaseModel):
    """Message from browser extension."""
    type: str  # "page", "selection", "search"
    url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    selection: Optional[str] = None
    action: Optional[str] = None  # "send_to_cathedral", "send_to_scripture"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExtensionResponse(BaseModel):
    """Response to browser extension."""
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None
