# BrowserGate Design Document

## Executive Summary

BrowserGate provides Cathedral with autonomous web browsing and search capabilities through a privacy-focused architecture. The system consists of four major components:

1. **BrowserGate Core** - Python module for headless browsing, web search (DuckDuckGo default), and content processing
2. **Chrome Extension** - Browser integration for sending pages/selections to Cathedral
3. **Windows Shell Extension** - System-wide context menu for "Send to Cathedral"
4. **WebSocket Bridge** - Real-time bidirectional communication between all components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER                                        │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐     │
│   │ Chrome + Ext │    │ Any Windows  │    │ Cathedral Web UI     │     │
│   │              │    │ Application  │    │                      │     │
│   └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘     │
│          │                   │                       │                  │
│          │ WebSocket         │ HTTP POST             │ SSE/HTTP         │
│          ▼                   ▼                       ▼                  │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    Cathedral Server (FastAPI)                 │     │
│   │  ┌────────────────────────────────────────────────────────┐  │     │
│   │  │                   WebSocket Hub                         │  │     │
│   │  │  - Extension connections                                │  │     │
│   │  │  - Agent browse requests                                │  │     │
│   │  │  - Bidirectional message routing                        │  │     │
│   │  └────────────────────────────────────────────────────────┘  │     │
│   │                            │                                  │     │
│   │  ┌─────────────────────────┴────────────────────────────┐    │     │
│   │  │                    BrowserGate                        │    │     │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │    │     │
│   │  │  │   Fetcher   │  │   Parser    │  │   Search     │  │    │     │
│   │  │  │ (Playwright)│  │ (HTML→MD)   │  │ (Providers)  │  │    │     │
│   │  │  └─────────────┘  └─────────────┘  └──────────────┘  │    │     │
│   │  └───────────────────────────────────────────────────────┘    │     │
│   │                            │                                  │     │
│   │  ┌─────────────────────────┴────────────────────────────┐    │     │
│   │  │              ScriptureGate (Optional Storage)         │    │     │
│   │  └───────────────────────────────────────────────────────┘    │     │
│   └──────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: BrowserGate Core

### 1.1 Module Structure

```
cathedral/
  BrowserGate/
    __init__.py          # BrowserManager class + convenience functions
    models.py            # Pydantic: PageContent, SearchResult, BrowseRequest
    fetcher.py           # Playwright-based page fetching
    parser.py            # HTML → Markdown conversion
    searcher.py          # Search provider abstraction + implementations
    providers/
      __init__.py        # Provider registry
      base.py            # Abstract SearchProvider base class
      duckduckgo.py      # DuckDuckGo provider (default)
      searxng.py         # SearXNG provider (self-hosted)
      brave.py           # Brave Search provider (API key)
    config.py            # Provider selection, timeouts, user agent settings
```

### 1.2 Data Models

```python
class BrowseRequest(BaseModel):
    """Request to fetch a web page."""
    url: str
    wait_for_js: bool = True           # Wait for JS rendering
    wait_timeout_ms: int = 10000       # Max wait for page load
    screenshot: bool = False           # Capture screenshot
    extract_links: bool = True         # Extract all links
    store_to_scripture: bool = False   # Auto-store in ScriptureGate

class PageContent(BaseModel):
    """Fetched and processed page content."""
    url: str
    final_url: str                     # After redirects
    title: str
    markdown: str                      # Converted content
    links: List[LinkInfo]              # Extracted links
    metadata: PageMetadata             # OG tags, description, etc.
    screenshot_path: Optional[str]     # If screenshot requested
    fetched_at: datetime
    fetch_duration_ms: int
    scripture_ref: Optional[str]       # If stored to ScriptureGate

class SearchRequest(BaseModel):
    """Request to perform web search."""
    query: str
    num_results: int = 10
    categories: List[str] = ["general"]  # general, images, news, etc.
    language: str = "en"
    safe_search: int = 1               # 0=off, 1=moderate, 2=strict

class SearchResult(BaseModel):
    """Single search result."""
    title: str
    url: str
    snippet: str
    engine: str                        # Which search engine provided this

class SearchResponse(BaseModel):
    """Complete search response."""
    query: str
    results: List[SearchResult]
    suggestions: List[str]
    total_results: int
    search_duration_ms: int
```

### 1.3 Fetcher (Playwright)

```python
# fetcher.py

class PageFetcher:
    """Headless browser for JavaScript-rendered pages."""

    def __init__(self, config: BrowserConfig):
        self.config = config
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def initialize(self):
        """Launch browser (call on startup)."""
        playwright = await async_playwright().start()
        self._browser = await playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        self._context = await self._browser.new_context(
            user_agent=self.config.user_agent,
            viewport={'width': 1280, 'height': 720}
        )

    async def fetch(self, request: BrowseRequest) -> PageContent:
        """Fetch a page with full JS rendering."""
        page = await self._context.new_page()

        try:
            # Navigate with timeout
            response = await page.goto(
                request.url,
                wait_until='networkidle' if request.wait_for_js else 'domcontentloaded',
                timeout=request.wait_timeout_ms
            )

            # Get final URL after redirects
            final_url = page.url

            # Extract content
            html = await page.content()
            title = await page.title()

            # Screenshot if requested
            screenshot_path = None
            if request.screenshot:
                screenshot_path = await self._take_screenshot(page, request.url)

            # Extract links if requested
            links = []
            if request.extract_links:
                links = await self._extract_links(page)

            return PageContent(
                url=request.url,
                final_url=final_url,
                title=title,
                html=html,  # Raw HTML, will be converted later
                links=links,
                screenshot_path=screenshot_path,
                # ... other fields
            )
        finally:
            await page.close()

    async def _take_screenshot(self, page: Page, url: str) -> str:
        """Save screenshot to scripture storage."""
        # Generate filename from URL hash
        filename = f"screenshot_{hash_url(url)}_{timestamp()}.png"
        path = SCRIPTURE_PATH / "screenshots" / filename
        await page.screenshot(path=str(path), full_page=True)
        return str(path)

    async def _extract_links(self, page: Page) -> List[LinkInfo]:
        """Extract all links from page."""
        return await page.evaluate('''() => {
            return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                text: a.innerText.trim(),
                href: a.href,
                rel: a.rel
            })).filter(l => l.href.startsWith('http'));
        }''')
```

### 1.4 Parser (HTML → Markdown)

```python
# parser.py

from markdownify import markdownify as md
from bs4 import BeautifulSoup
from readability import Document  # mozilla/readability port

class ContentParser:
    """Convert HTML to clean Markdown."""

    def parse(self, html: str, url: str) -> ParsedContent:
        """Extract main content and convert to Markdown."""

        # Use Readability to extract main content
        doc = Document(html)
        title = doc.title()
        main_html = doc.summary()

        # Parse with BeautifulSoup for cleanup
        soup = BeautifulSoup(main_html, 'html.parser')

        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose()

        # Extract metadata
        metadata = self._extract_metadata(BeautifulSoup(html, 'html.parser'))

        # Convert to Markdown
        markdown = md(
            str(soup),
            heading_style='atx',
            bullets='-',
            strip=['script', 'style']
        )

        # Clean up excessive whitespace
        markdown = self._clean_markdown(markdown)

        return ParsedContent(
            title=title,
            markdown=markdown,
            metadata=metadata,
            word_count=len(markdown.split())
        )

    def _extract_metadata(self, soup: BeautifulSoup) -> PageMetadata:
        """Extract Open Graph and meta tags."""
        meta = PageMetadata()

        # Open Graph
        og_title = soup.find('meta', property='og:title')
        og_desc = soup.find('meta', property='og:description')
        og_image = soup.find('meta', property='og:image')

        if og_title:
            meta.og_title = og_title.get('content')
        if og_desc:
            meta.og_description = og_desc.get('content')
        if og_image:
            meta.og_image = og_image.get('content')

        # Standard meta
        desc = soup.find('meta', attrs={'name': 'description'})
        if desc:
            meta.description = desc.get('content')

        # Canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical:
            meta.canonical_url = canonical.get('href')

        return meta

    def _clean_markdown(self, md: str) -> str:
        """Clean up markdown output."""
        import re
        # Collapse multiple blank lines
        md = re.sub(r'\n{3,}', '\n\n', md)
        # Remove trailing whitespace
        md = '\n'.join(line.rstrip() for line in md.split('\n'))
        return md.strip()
```

### 1.5 Search Provider System

The search system uses a provider abstraction allowing users to choose their preferred search backend.

#### Provider Base Class

```python
# providers/base.py

from abc import ABC, abstractmethod
from typing import List
from ..models import SearchRequest, SearchResponse

class SearchProvider(ABC):
    """Abstract base class for search providers."""

    name: str = "base"
    requires_api_key: bool = False

    @abstractmethod
    async def initialize(self):
        """Initialize the provider (create HTTP clients, etc.)."""
        pass

    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform a search and return results."""
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass
```

#### DuckDuckGo Provider (Default)

```python
# providers/duckduckgo.py

from duckduckgo_search import AsyncDDGS
import time

class DuckDuckGoProvider(SearchProvider):
    """Privacy-focused search via DuckDuckGo (no API key required)."""

    name = "duckduckgo"
    requires_api_key = False

    def __init__(self, config: SearchConfig):
        self.timeout = config.search_timeout
        self._ddgs: Optional[AsyncDDGS] = None

    async def initialize(self):
        """Create DuckDuckGo client."""
        self._ddgs = AsyncDDGS()

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform search via DuckDuckGo."""
        start = time.time()

        # Map safe_search: 0=off, 1=moderate, 2=strict
        safesearch = ["off", "moderate", "strict"][min(request.safe_search, 2)]

        # Perform search
        raw_results = await self._ddgs.text(
            keywords=request.query,
            region=request.language,
            safesearch=safesearch,
            max_results=request.num_results
        )

        duration_ms = int((time.time() - start) * 1000)

        results = [
            SearchResult(
                title=r.get('title', ''),
                url=r.get('href', ''),
                snippet=r.get('body', ''),
                engine='duckduckgo'
            )
            for r in raw_results
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            suggestions=[],
            total_results=len(results),
            search_duration_ms=duration_ms
        )

    async def close(self):
        pass  # AsyncDDGS handles cleanup
```

#### SearXNG Provider (Self-Hosted)

```python
# providers/searxng.py

import httpx

class SearXNGProvider(SearchProvider):
    """Privacy-focused search via self-hosted SearXNG instance."""

    name = "searxng"
    requires_api_key = False

    def __init__(self, config: SearchConfig):
        self.base_url = config.searxng_url  # e.g., "http://localhost:8080"
        self.timeout = config.search_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={'Accept': 'application/json'}
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        params = {
            'q': request.query,
            'format': 'json',
            'categories': ','.join(request.categories),
            'language': request.language,
            'safesearch': request.safe_search,
        }

        start = time.time()
        response = await self._client.get(f"{self.base_url}/search", params=params)
        response.raise_for_status()
        data = response.json()
        duration_ms = int((time.time() - start) * 1000)

        results = [
            SearchResult(
                title=r.get('title', ''),
                url=r.get('url', ''),
                snippet=r.get('content', ''),
                engine=r.get('engine', 'searxng')
            )
            for r in data.get('results', [])[:request.num_results]
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            suggestions=data.get('suggestions', []),
            total_results=len(results),
            search_duration_ms=duration_ms
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
```

#### Brave Search Provider (API Key)

```python
# providers/brave.py

import httpx

class BraveSearchProvider(SearchProvider):
    """Web search via Brave Search API (requires API key)."""

    name = "brave"
    requires_api_key = True

    def __init__(self, config: SearchConfig):
        self.api_key = config.brave_api_key
        self.timeout = config.search_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        if not self.api_key:
            raise ValueError("Brave Search requires an API key")
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                'Accept': 'application/json',
                'X-Subscription-Token': self.api_key
            }
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        params = {
            'q': request.query,
            'count': request.num_results,
        }

        start = time.time()
        response = await self._client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params
        )
        response.raise_for_status()
        data = response.json()
        duration_ms = int((time.time() - start) * 1000)

        results = [
            SearchResult(
                title=r.get('title', ''),
                url=r.get('url', ''),
                snippet=r.get('description', ''),
                engine='brave'
            )
            for r in data.get('web', {}).get('results', [])
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            suggestions=[],
            total_results=len(results),
            search_duration_ms=duration_ms
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
```

#### Provider Registry

```python
# providers/__init__.py

from typing import Dict, Type
from .base import SearchProvider
from .duckduckgo import DuckDuckGoProvider
from .searxng import SearXNGProvider
from .brave import BraveSearchProvider

PROVIDERS: Dict[str, Type[SearchProvider]] = {
    "duckduckgo": DuckDuckGoProvider,
    "searxng": SearXNGProvider,
    "brave": BraveSearchProvider,
}

def get_provider(name: str, config) -> SearchProvider:
    """Get a search provider instance by name."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown search provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name](config)
```

#### Searcher Wrapper

```python
# searcher.py

from .providers import get_provider, PROVIDERS
from .models import SearchRequest, SearchResponse

class WebSearcher:
    """Unified search interface with configurable provider."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.provider_name = config.provider  # Default: "duckduckgo"
        self._provider: Optional[SearchProvider] = None

    async def initialize(self):
        """Initialize the configured search provider."""
        self._provider = get_provider(self.provider_name, self.config)
        await self._provider.initialize()

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform search using configured provider."""
        return await self._provider.search(request)

    async def switch_provider(self, provider_name: str):
        """Switch to a different search provider."""
        if self._provider:
            await self._provider.close()
        self.provider_name = provider_name
        self._provider = get_provider(provider_name, self.config)
        await self._provider.initialize()

    @staticmethod
    def list_providers() -> list:
        """List available search providers."""
        return [
            {
                "name": name,
                "requires_api_key": cls.requires_api_key
            }
            for name, cls in PROVIDERS.items()
        ]

    async def close(self):
        if self._provider:
            await self._provider.close()
```

### 1.6 Main Interface

```python
# __init__.py

class BrowserGate:
    """Main interface for web browsing capabilities."""

    @classmethod
    async def initialize(cls):
        """Initialize browser and search systems."""
        global _fetcher, _parser, _searcher

        config = cls._load_config()

        _fetcher = PageFetcher(config.browser)
        await _fetcher.initialize()

        _parser = ContentParser()

        # Initialize search with configured provider (default: DuckDuckGo)
        _searcher = WebSearcher(config.search)
        await _searcher.initialize()

    @classmethod
    async def fetch(
        cls,
        url: str,
        wait_for_js: bool = True,
        screenshot: bool = False,
        store_to_scripture: bool = False
    ) -> PageContent:
        """Fetch and parse a web page."""
        request = BrowseRequest(
            url=url,
            wait_for_js=wait_for_js,
            screenshot=screenshot,
            store_to_scripture=store_to_scripture
        )

        # Fetch with Playwright
        raw = await _fetcher.fetch(request)

        # Parse HTML to Markdown
        parsed = _parser.parse(raw.html, url)

        # Build result
        content = PageContent(
            url=request.url,
            final_url=raw.final_url,
            title=parsed.title,
            markdown=parsed.markdown,
            links=raw.links,
            metadata=parsed.metadata,
            screenshot_path=raw.screenshot_path,
            fetched_at=datetime.utcnow(),
            fetch_duration_ms=raw.fetch_duration_ms
        )

        # Store to ScriptureGate if requested
        if store_to_scripture:
            ref = await cls._store_to_scripture(content)
            content.scripture_ref = ref

        return content

    @classmethod
    async def search(
        cls,
        query: str,
        num_results: int = 10,
        categories: List[str] = None
    ) -> SearchResponse:
        """Perform web search using configured provider."""
        request = SearchRequest(
            query=query,
            num_results=num_results,
            categories=categories or ["general"]
        )
        return await _searcher.search(request)

    @classmethod
    async def switch_search_provider(cls, provider: str):
        """Switch to a different search provider (duckduckgo, searxng, brave)."""
        await _searcher.switch_provider(provider)

    @classmethod
    def list_search_providers(cls) -> list:
        """List available search providers."""
        return WebSearcher.list_providers()

    @classmethod
    def get_current_provider(cls) -> str:
        """Get the current search provider name."""
        return _searcher.provider_name

    @classmethod
    async def search_and_fetch(
        cls,
        query: str,
        fetch_top_n: int = 3
    ) -> Tuple[SearchResponse, List[PageContent]]:
        """Search and automatically fetch top results."""
        search_results = await cls.search(query)

        pages = []
        for result in search_results.results[:fetch_top_n]:
            try:
                page = await cls.fetch(result.url)
                pages.append(page)
            except Exception as e:
                # Log but continue with other results
                pass

        return search_results, pages

    @classmethod
    async def _store_to_scripture(cls, content: PageContent) -> str:
        """Store page content in ScriptureGate."""
        from cathedral import ScriptureGate

        # Create temporary markdown file
        temp_path = Path(tempfile.mktemp(suffix='.md'))
        temp_path.write_text(content.markdown, encoding='utf-8')

        try:
            result = await ScriptureGate.store(
                source=str(temp_path),
                title=content.title,
                source_type="web",
                tags=["web", "browsing", urlparse(content.url).netloc],
                extra_metadata={
                    "url": content.url,
                    "final_url": content.final_url,
                    "fetched_at": content.fetched_at.isoformat()
                }
            )
            return result.get('ref')
        finally:
            temp_path.unlink(missing_ok=True)
```

### 1.7 Chat Commands

```
| Command | Description |
|---------|-------------|
| `/browse <url>` | Fetch page, display markdown summary |
| `/browse <url> --store` | Fetch and store to ScriptureGate |
| `/browse <url> --screenshot` | Fetch with screenshot |
| `/search <query>` | Web search (uses configured provider) |
| `/search <query> --fetch` | Search and fetch top 3 results |
| `/search <query> --images` | Image search |
| `/search <query> --news` | News search |
| `/search-provider` | Show current search provider |
| `/search-provider <name>` | Switch provider (duckduckgo, searxng, brave) |
| `/search-providers` | List available providers |
```

### 1.8 API Endpoints

```
GET    /api/browse/config          # Get browser configuration
PUT    /api/browse/config          # Update configuration

POST   /api/browse/fetch           # Fetch a URL
  Body: { url, wait_for_js, screenshot, store_to_scripture }
  Returns: PageContent

POST   /api/browse/search          # Web search
  Body: { query, num_results, categories }
  Returns: SearchResponse

POST   /api/browse/search-fetch    # Search and fetch top results
  Body: { query, fetch_top_n }
  Returns: { search: SearchResponse, pages: PageContent[] }

GET    /api/browse/search/providers      # List available search providers
GET    /api/browse/search/provider       # Get current provider
PUT    /api/browse/search/provider       # Switch provider
  Body: { provider: "duckduckgo" | "searxng" | "brave" }

GET    /api/browse/history         # Recent browsing history
DELETE /api/browse/history         # Clear history

WS     /ws/browser                 # WebSocket for extension communication
```

---

## Part 2: Chrome Extension

### 2.1 Extension Structure

```
cathedral-extension/
  manifest.json           # Extension manifest (v3)
  background.js           # Service worker for WebSocket
  content.js              # Page content script
  popup/
    popup.html            # Extension popup UI
    popup.js              # Popup logic
    popup.css             # Popup styles
  icons/
    icon16.png
    icon48.png
    icon128.png
  lib/
    websocket-client.js   # WebSocket wrapper
```

### 2.2 Manifest (v3)

```json
{
  "manifest_version": 3,
  "name": "Cathedral Browser Bridge",
  "version": "1.0.0",
  "description": "Send web pages and selections to Cathedral",

  "permissions": [
    "activeTab",
    "contextMenus",
    "storage",
    "scripting"
  ],

  "host_permissions": [
    "http://localhost:8000/*",
    "<all_urls>"
  ],

  "background": {
    "service_worker": "background.js",
    "type": "module"
  },

  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"],
      "run_at": "document_idle"
    }
  ],

  "action": {
    "default_popup": "popup/popup.html",
    "default_icon": {
      "16": "icons/icon16.png",
      "48": "icons/icon48.png",
      "128": "icons/icon128.png"
    }
  },

  "icons": {
    "16": "icons/icon16.png",
    "48": "icons/icon48.png",
    "128": "icons/icon128.png"
  }
}
```

### 2.3 Background Service Worker

```javascript
// background.js

class CathedralBridge {
  constructor() {
    this.ws = null;
    this.serverUrl = 'ws://localhost:8000/ws/browser';
    this.reconnectInterval = 5000;
    this.connected = false;
  }

  async connect() {
    try {
      this.ws = new WebSocket(this.serverUrl);

      this.ws.onopen = () => {
        console.log('[Cathedral] Connected to server');
        this.connected = true;
        this.sendIdentify();
        this.updateBadge('connected');
      };

      this.ws.onclose = () => {
        console.log('[Cathedral] Disconnected from server');
        this.connected = false;
        this.updateBadge('disconnected');
        setTimeout(() => this.connect(), this.reconnectInterval);
      };

      this.ws.onerror = (error) => {
        console.error('[Cathedral] WebSocket error:', error);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(JSON.parse(event.data));
      };

    } catch (error) {
      console.error('[Cathedral] Connection failed:', error);
      setTimeout(() => this.connect(), this.reconnectInterval);
    }
  }

  sendIdentify() {
    this.send({
      type: 'identify',
      client: 'chrome-extension',
      version: chrome.runtime.getManifest().version
    });
  }

  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  handleMessage(message) {
    switch (message.type) {
      case 'fetch_request':
        // Server is asking us to fetch a page
        this.handleFetchRequest(message);
        break;
      case 'search_request':
        // Server wants us to perform a search
        this.handleSearchRequest(message);
        break;
      case 'ack':
        console.log('[Cathedral] Server acknowledged:', message.id);
        break;
    }
  }

  async handleFetchRequest(message) {
    // Get current tab or open new one
    const tab = await this.getOrCreateTab(message.url);

    // Wait for page load
    await this.waitForPageLoad(tab.id);

    // Execute content script to get page content
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractPageContent
    });

    // Send content back to server
    this.send({
      type: 'fetch_response',
      request_id: message.request_id,
      content: results[0].result
    });
  }

  updateBadge(status) {
    const color = status === 'connected' ? '#4CAF50' : '#9E9E9E';
    const text = status === 'connected' ? '' : '!';
    chrome.action.setBadgeBackgroundColor({ color });
    chrome.action.setBadgeText({ text });
  }
}

// Content extraction function (injected into page)
function extractPageContent() {
  return {
    url: window.location.href,
    title: document.title,
    html: document.documentElement.outerHTML,
    selection: window.getSelection().toString()
  };
}

// Context menu setup
chrome.runtime.onInstalled.addListener(() => {
  // Send page to Cathedral
  chrome.contextMenus.create({
    id: 'send-page-to-cathedral',
    title: 'Send Page to Cathedral',
    contexts: ['page']
  });

  // Send selection to Cathedral
  chrome.contextMenus.create({
    id: 'send-selection-to-cathedral',
    title: 'Send Selection to Cathedral',
    contexts: ['selection']
  });

  // Send page to ScriptureGate
  chrome.contextMenus.create({
    id: 'send-page-to-scripture',
    title: 'Send Page to ScriptureGate',
    contexts: ['page']
  });

  // Send selection to ScriptureGate
  chrome.contextMenus.create({
    id: 'send-selection-to-scripture',
    title: 'Send Selection to ScriptureGate',
    contexts: ['selection']
  });

  // Separator
  chrome.contextMenus.create({
    id: 'separator',
    type: 'separator',
    contexts: ['page', 'selection']
  });

  // Ask Cathedral about selection
  chrome.contextMenus.create({
    id: 'ask-cathedral-about',
    title: 'Ask Cathedral About This',
    contexts: ['selection']
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  switch (info.menuItemId) {
    case 'send-page-to-cathedral':
      await bridge.sendPageToCathedral(tab, false);
      break;
    case 'send-selection-to-cathedral':
      await bridge.sendSelectionToCathedral(info.selectionText, tab.url, false);
      break;
    case 'send-page-to-scripture':
      await bridge.sendPageToCathedral(tab, true);
      break;
    case 'send-selection-to-scripture':
      await bridge.sendSelectionToCathedral(info.selectionText, tab.url, true);
      break;
    case 'ask-cathedral-about':
      await bridge.openCathedralWithContext(info.selectionText, tab.url);
      break;
  }
});

// Initialize bridge
const bridge = new CathedralBridge();
bridge.connect();
```

### 2.4 Content Script

```javascript
// content.js

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  switch (request.action) {
    case 'getPageContent':
      sendResponse({
        url: window.location.href,
        title: document.title,
        html: document.documentElement.outerHTML,
        selection: window.getSelection().toString()
      });
      break;

    case 'getSelection':
      sendResponse({
        selection: window.getSelection().toString(),
        url: window.location.href
      });
      break;
  }
  return true;  // Keep channel open for async response
});
```

### 2.5 Popup UI

```html
<!-- popup/popup.html -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <link rel="stylesheet" href="popup.css">
</head>
<body>
  <div class="popup">
    <div class="header">
      <span class="logo">Cathedral</span>
      <span id="status" class="status disconnected">Disconnected</span>
    </div>

    <div class="actions">
      <button id="send-page" class="btn primary">
        Send This Page
      </button>
      <button id="send-to-scripture" class="btn secondary">
        Store in ScriptureGate
      </button>
    </div>

    <div class="quick-search">
      <input type="text" id="search-input" placeholder="Quick search...">
      <button id="search-btn" class="btn small">Search</button>
    </div>

    <div class="footer">
      <a href="http://localhost:8000" target="_blank">Open Cathedral</a>
      <button id="settings-btn" class="btn-icon">Settings</button>
    </div>
  </div>

  <script src="popup.js"></script>
</body>
</html>
```

---

## Part 3: Windows Shell Extension

### 3.1 Overview

A Windows Shell Extension (COM component) that adds "Send to Cathedral" to the system-wide right-click context menu. When triggered, it opens Cathedral in the default browser with the selected text pre-filled.

### 3.2 Structure

```
cathedral-shell/
  src/
    CathedralShellExt.cpp    # Main extension implementation
    ContextMenuHandler.cpp    # IContextMenu implementation
    ContextMenuHandler.h
    dllmain.cpp              # DLL entry point
    resource.h
    resource.rc              # Icons and strings
  include/
    CathedralShellExt.h
  CathedralShellExt.def      # Export definitions
  CMakeLists.txt             # Build configuration
  register.bat               # Registration script
  unregister.bat             # Unregistration script
```

### 3.3 Core Implementation

```cpp
// ContextMenuHandler.cpp

#include <windows.h>
#include <shlobj.h>
#include <shellapi.h>
#include <string>

class CathedralContextMenuHandler :
    public IShellExtInit,
    public IContextMenu
{
private:
    ULONG m_refCount;
    std::wstring m_selectedText;

public:
    // IShellExtInit
    STDMETHODIMP Initialize(
        PCIDLIST_ABSOLUTE pidlFolder,
        IDataObject* pdtobj,
        HKEY hkeyProgID) override
    {
        // Get selected text from clipboard or data object
        if (OpenClipboard(NULL)) {
            HANDLE hData = GetClipboardData(CF_UNICODETEXT);
            if (hData) {
                wchar_t* pText = static_cast<wchar_t*>(GlobalLock(hData));
                if (pText) {
                    m_selectedText = pText;
                    GlobalUnlock(hData);
                }
            }
            CloseClipboard();
        }
        return S_OK;
    }

    // IContextMenu
    STDMETHODIMP QueryContextMenu(
        HMENU hmenu,
        UINT indexMenu,
        UINT idCmdFirst,
        UINT idCmdLast,
        UINT uFlags) override
    {
        if (uFlags & CMF_DEFAULTONLY)
            return MAKE_HRESULT(SEVERITY_SUCCESS, 0, 0);

        // Add separator
        InsertMenu(hmenu, indexMenu++, MF_SEPARATOR | MF_BYPOSITION, 0, NULL);

        // Add "Send to Cathedral" menu item
        InsertMenuW(hmenu, indexMenu++, MF_STRING | MF_BYPOSITION,
            idCmdFirst + 0, L"Send to Cathedral");

        // Add "Store in ScriptureGate" menu item
        InsertMenuW(hmenu, indexMenu++, MF_STRING | MF_BYPOSITION,
            idCmdFirst + 1, L"Store in ScriptureGate");

        return MAKE_HRESULT(SEVERITY_SUCCESS, 0, 2);
    }

    STDMETHODIMP InvokeCommand(CMINVOKECOMMANDINFO* pici) override
    {
        if (HIWORD(pici->lpVerb) != 0)
            return E_INVALIDARG;

        switch (LOWORD(pici->lpVerb)) {
            case 0:  // Send to Cathedral
                OpenCathedralWithText(m_selectedText, false);
                break;
            case 1:  // Store in ScriptureGate
                OpenCathedralWithText(m_selectedText, true);
                break;
        }
        return S_OK;
    }

private:
    void OpenCathedralWithText(const std::wstring& text, bool storeToScripture)
    {
        // URL-encode the text
        std::wstring encoded = UrlEncode(text);

        // Build Cathedral URL with pre-filled context
        std::wstring url = L"http://localhost:8000/?context=";
        url += encoded;

        if (storeToScripture) {
            url += L"&store=scripture";
        }

        // Open in default browser
        ShellExecuteW(NULL, L"open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
    }

    std::wstring UrlEncode(const std::wstring& str)
    {
        // URL encoding implementation
        // ...
    }
};
```

### 3.4 Registration

```batch
@echo off
:: register.bat - Run as Administrator

:: Register the DLL
regsvr32 /s "%~dp0CathedralShellExt.dll"

:: Add context menu handler to registry
reg add "HKEY_CLASSES_ROOT\*\shell\Cathedral" /ve /d "Send to Cathedral" /f
reg add "HKEY_CLASSES_ROOT\*\shell\Cathedral\command" /ve /d "\"%~dp0CathedralShellExt.dll\" \"%%1\"" /f

echo Cathedral Shell Extension registered successfully.
pause
```

### 3.5 Simpler Alternative: PowerShell + Hotkey

If the full COM shell extension is too heavy, a simpler alternative using AutoHotkey or PowerShell:

```powershell
# cathedral-hotkey.ps1
# Run at startup via Task Scheduler

Add-Type -AssemblyName System.Windows.Forms

# Register global hotkey: Win+Shift+C
$form = New-Object System.Windows.Forms.Form
$form.Visible = $false

# Hotkey handler
Register-ObjectEvent -InputObject $form -EventName KeyDown -Action {
    if ($Event.SourceEventArgs.KeyCode -eq 'C' -and
        $Event.SourceEventArgs.Modifiers -eq 'Control, Shift') {

        # Get clipboard text
        $text = Get-Clipboard

        if ($text) {
            # URL encode
            $encoded = [System.Web.HttpUtility]::UrlEncode($text)

            # Open Cathedral
            Start-Process "http://localhost:8000/?context=$encoded"
        }
    }
}

# Keep script running
[System.Windows.Forms.Application]::Run($form)
```

---

## Part 4: WebSocket Hub

### 4.1 Server-Side WebSocket Handler

```python
# In altar/run.py

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import uuid

class BrowserWebSocketHub:
    """Manages WebSocket connections from browser extensions and internal requests."""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}  # connection_id -> WebSocket
        self.extension_connections: Set[str] = set()  # IDs of extension connections
        self.pending_requests: Dict[str, asyncio.Future] = {}  # request_id -> Future

    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new connection."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())[:8]
        self.connections[connection_id] = websocket
        return connection_id

    async def disconnect(self, connection_id: str):
        """Handle disconnection."""
        self.connections.pop(connection_id, None)
        self.extension_connections.discard(connection_id)

    async def handle_message(self, connection_id: str, message: dict):
        """Process incoming WebSocket message."""
        msg_type = message.get('type')

        if msg_type == 'identify':
            # Extension identifying itself
            if message.get('client') == 'chrome-extension':
                self.extension_connections.add(connection_id)
                await self.send(connection_id, {'type': 'ack', 'status': 'registered'})

        elif msg_type == 'page_content':
            # Extension sending page content
            await self._handle_page_content(connection_id, message)

        elif msg_type == 'fetch_response':
            # Response to a fetch request
            request_id = message.get('request_id')
            if request_id in self.pending_requests:
                self.pending_requests[request_id].set_result(message.get('content'))

        elif msg_type == 'selection':
            # User sent selection from extension
            await self._handle_selection(connection_id, message)

    async def request_fetch(self, url: str, timeout: float = 30.0) -> dict:
        """Request extension to fetch a URL."""
        if not self.extension_connections:
            raise RuntimeError("No browser extension connected")

        # Pick first available extension
        connection_id = next(iter(self.extension_connections))
        request_id = str(uuid.uuid4())[:8]

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        try:
            # Send fetch request
            await self.send(connection_id, {
                'type': 'fetch_request',
                'request_id': request_id,
                'url': url
            })

            # Wait for response
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.pending_requests.pop(request_id, None)

    async def send(self, connection_id: str, message: dict):
        """Send message to a connection."""
        ws = self.connections.get(connection_id)
        if ws:
            await ws.send_json(message)

    async def broadcast_to_extensions(self, message: dict):
        """Broadcast to all extension connections."""
        for conn_id in self.extension_connections:
            await self.send(conn_id, message)

    async def _handle_page_content(self, connection_id: str, message: dict):
        """Process page content from extension."""
        content = message.get('content', {})
        store_to_scripture = message.get('store_to_scripture', False)

        # Parse and process
        from cathedral import BrowserGate

        parsed = BrowserGate.parse_html(content['html'], content['url'])

        if store_to_scripture:
            ref = await BrowserGate.store_to_scripture(parsed)
            await self.send(connection_id, {
                'type': 'stored',
                'scripture_ref': ref
            })

        # Emit event
        await emit_event('browsing', f"Received page: {content['title'][:50]}")

    async def _handle_selection(self, connection_id: str, message: dict):
        """Process text selection from extension."""
        selection = message.get('selection', '')
        source_url = message.get('url', '')
        store_to_scripture = message.get('store_to_scripture', False)

        if store_to_scripture:
            from cathedral import ScriptureGate
            # Store as text artifact
            # ...

        await emit_event('browsing', f"Received selection: {selection[:50]}...")


# Global hub instance
browser_ws_hub = BrowserWebSocketHub()


@app.websocket("/ws/browser")
async def websocket_browser(websocket: WebSocket):
    """WebSocket endpoint for browser extension communication."""
    connection_id = await browser_ws_hub.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            await browser_ws_hub.handle_message(connection_id, data)
    except WebSocketDisconnect:
        await browser_ws_hub.disconnect(connection_id)
```

### 4.2 Agent Integration

When the LLM agent needs to browse:

```python
# In cathedral/__init__.py

async def handle_agent_browse_request(url: str) -> str:
    """Handle agent's request to fetch a webpage."""
    from altar.run import browser_ws_hub

    # Check if extension is connected
    if browser_ws_hub.extension_connections:
        # Use extension (preserves cookies, handles JS)
        content = await browser_ws_hub.request_fetch(url)
        return BrowserGate.parse_html(content['html'], url).markdown
    else:
        # Fall back to headless browser
        page = await BrowserGate.fetch(url)
        return page.markdown
```

---

## Part 5: Web UI Integration

### 5.1 Cathedral URL Parameters

The Cathedral web UI should handle URL parameters for pre-filled context:

```javascript
// In altar/static/app.js

document.addEventListener('DOMContentLoaded', () => {
    // Parse URL parameters
    const params = new URLSearchParams(window.location.search);

    const context = params.get('context');
    const store = params.get('store');

    if (context) {
        // Pre-fill the chat input with context
        const decodedContext = decodeURIComponent(context);

        if (store === 'scripture') {
            // Auto-store to ScriptureGate
            sendToScripture(decodedContext);
        } else {
            // Pre-fill chat input
            const input = document.getElementById('chat-input');
            input.value = `Context from external source:\n\n${decodedContext}\n\n`;
            input.focus();
        }

        // Clear URL parameters
        window.history.replaceState({}, '', '/');
    }
});

async function sendToScripture(content) {
    // Show notification
    showToast('Storing to ScriptureGate...');

    // Store via API
    const response = await fetch('/api/scripture/store-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            content: content,
            title: 'External Selection',
            source_type: 'external'
        })
    });

    if (response.ok) {
        const data = await response.json();
        showToast(`Stored as ${data.ref}`);
    }
}
```

---

## Part 6: Configuration

### 6.1 BrowserGate Configuration

```json
// data/config/browser.json

{
    "browser": {
        "headless": true,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "viewport_width": 1280,
        "viewport_height": 720,
        "default_timeout_ms": 30000,
        "screenshot_quality": 80
    },
    "search": {
        "provider": "duckduckgo",       // Default provider (duckduckgo|searxng|brave)
        "default_num_results": 10,
        "default_language": "en",
        "default_safe_search": 1,       // 0=off, 1=moderate, 2=strict
        "timeout_seconds": 10,

        // Provider-specific settings (only needed if using that provider)
        "providers": {
            "searxng": {
                "url": "http://localhost:8080"
            },
            "brave": {
                "api_key": ""           // Get from https://brave.com/search/api/
            }
        }
    },
    "storage": {
        "auto_store_fetched": false,
        "store_screenshots": true,
        "max_content_length": 500000
    },
    "extension": {
        "require_auth": false,
        "allowed_origins": ["chrome-extension://*"]
    }
}
```

### 6.2 Search Provider Comparison

| Provider | API Key Required | Privacy | Rate Limits | Features |
|----------|------------------|---------|-------------|----------|
| **DuckDuckGo** (default) | No | High | Generous | Web, images, news |
| **SearXNG** | No (self-hosted) | Maximum | None (your server) | Aggregates multiple engines |
| **Brave** | Yes (free tier) | High | 2000/month free | Web, with AI summaries |

---

## Part 7: Dependencies

### 7.1 Python Dependencies

```
# requirements-browser.txt

playwright>=1.40.0          # Headless browser
markdownify>=0.11.6         # HTML to Markdown
beautifulsoup4>=4.12.0      # HTML parsing
readability-lxml>=0.8.1     # Content extraction
httpx>=0.25.0               # Async HTTP client
duckduckgo-search>=4.0.0    # DuckDuckGo search (default provider)
websockets>=12.0            # WebSocket support (if not using Starlette's)
```

### 7.2 Playwright Setup

```bash
# Install Playwright browsers
playwright install chromium
```

### 7.3 Search Provider Setup

**DuckDuckGo (Default)** - No setup required. Works out of the box.

**SearXNG (Optional)** - For maximum privacy with self-hosted search:

```yaml
# docker-compose.searxng.yml

version: '3'
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8080:8080"
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/
```

Then update `data/config/browser.json`:
```json
{
    "search": {
        "provider": "searxng",
        "providers": {
            "searxng": { "url": "http://localhost:8080" }
        }
    }
}
```

**Brave Search (Optional)** - Get API key from https://brave.com/search/api/

---

## Part 8: File Summary

### Files to Create

| Path | Purpose |
|------|---------|
| `cathedral/BrowserGate/__init__.py` | Main module interface |
| `cathedral/BrowserGate/models.py` | Pydantic models |
| `cathedral/BrowserGate/fetcher.py` | Playwright page fetcher |
| `cathedral/BrowserGate/parser.py` | HTML→Markdown parser |
| `cathedral/BrowserGate/searcher.py` | Search provider wrapper |
| `cathedral/BrowserGate/config.py` | Configuration management |
| `cathedral/BrowserGate/providers/__init__.py` | Provider registry |
| `cathedral/BrowserGate/providers/base.py` | Abstract provider class |
| `cathedral/BrowserGate/providers/duckduckgo.py` | DuckDuckGo (default) |
| `cathedral/BrowserGate/providers/searxng.py` | SearXNG (self-hosted) |
| `cathedral/BrowserGate/providers/brave.py` | Brave Search (API) |
| `cathedral-extension/manifest.json` | Chrome extension manifest |
| `cathedral-extension/background.js` | Service worker |
| `cathedral-extension/content.js` | Content script |
| `cathedral-extension/popup/*` | Popup UI |
| `cathedral-shell/src/*` | Windows shell extension |
| `data/config/browser.json` | Browser configuration |

### Files to Modify

| Path | Changes |
|------|---------|
| `cathedral/__init__.py` | Add /browse, /search commands |
| `altar/run.py` | Add WebSocket hub, /api/browse/* endpoints |
| `altar/static/app.js` | Handle URL parameters for context |
| `altar/templates/index.html` | Add browser status indicator |
| `requirements.txt` | Add playwright, markdownify, etc. |

---

## Part 9: Implementation Phases

### Phase 1: BrowserGate Core
1. Create module structure
2. Implement models.py
3. Implement fetcher.py (Playwright)
4. Implement parser.py (HTML→MD)
5. Implement providers/ (base, duckduckgo, searxng, brave)
6. Implement searcher.py (provider wrapper)
7. Implement __init__.py

### Phase 2: API & Commands
1. Add WebSocket hub to run.py
2. Add /api/browse/* endpoints
3. Add chat commands (/browse, /search)
4. Test with DuckDuckGo (default provider)

### Phase 3: Chrome Extension
1. Create extension structure
2. Implement background.js with WebSocket
3. Implement context menus
4. Implement popup UI
5. Test extension-server communication

### Phase 4: Windows Shell Extension
1. Set up C++ build environment
2. Implement COM shell extension
3. Create installer/registration scripts
4. Test system-wide integration

### Phase 5: Integration & Polish
1. Update web UI for context parameters
2. Add browsing history UI
3. Add extension status indicator
4. Documentation

---

## Verification Checklist

1. **BrowserGate Core**
   - [ ] `BrowserGate.fetch(url)` returns markdown content
   - [ ] `BrowserGate.search(query)` returns DuckDuckGo results
   - [ ] Provider switching works (DDG → SearXNG → Brave)
   - [ ] Screenshots captured and stored
   - [ ] ScriptureGate integration works

2. **Chat Commands**
   - [ ] `/browse <url>` displays page content
   - [ ] `/search <query>` shows search results
   - [ ] `--store` flag stores to ScriptureGate

3. **Chrome Extension**
   - [ ] Connects to Cathedral via WebSocket
   - [ ] "Send Page to Cathedral" works
   - [ ] "Send to ScriptureGate" works
   - [ ] "Ask Cathedral About This" opens with context

4. **Windows Shell**
   - [ ] Context menu appears on right-click
   - [ ] "Send to Cathedral" opens browser
   - [ ] Selected text is pre-filled

5. **WebSocket Hub**
   - [ ] Extension connections tracked
   - [ ] Agent can request fetches via extension
   - [ ] Fallback to headless browser works
