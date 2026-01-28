"""
Page content fetcher.

Supports simple mode (requests + BeautifulSoup) and headless mode (Playwright).
"""

import time
import asyncio
from typing import Optional, Dict, Any
from .models import PageContent, FetchMode, ContentFormat


class PageFetcher:
    """
    Fetches and processes web page content.

    Supports two modes:
    - SIMPLE: Uses requests + BeautifulSoup (fast, no JS)
    - HEADLESS: Uses Playwright (slower, full JS rendering)
    """

    def __init__(
        self,
        default_mode: FetchMode = FetchMode.SIMPLE,
        default_format: ContentFormat = ContentFormat.MARKDOWN,
        timeout_seconds: int = 30,
        max_content_length: int = 100000,
    ):
        self.default_mode = default_mode
        self.default_format = default_format
        self.timeout_seconds = timeout_seconds
        self.max_content_length = max_content_length

    async def fetch(
        self,
        url: str,
        mode: Optional[FetchMode] = None,
        output_format: Optional[ContentFormat] = None,
        wait_for_selector: Optional[str] = None,
        **kwargs
    ) -> PageContent:
        """
        Fetch page content.

        Args:
            url: URL to fetch
            mode: Fetch mode (default from config)
            output_format: Output format (default from config)
            wait_for_selector: CSS selector to wait for (headless only)
            **kwargs: Additional options

        Returns:
            PageContent with fetched data
        """
        mode = mode or self.default_mode
        output_format = output_format or self.default_format

        start_time = time.time()

        if mode == FetchMode.SIMPLE:
            html, title = await self._fetch_simple(url)
        else:
            html, title = await self._fetch_headless(url, wait_for_selector)

        # Convert content to requested format
        content = self._convert_content(html, output_format)

        # Truncate if needed
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n\n[Content truncated...]"

        fetch_time_ms = int((time.time() - start_time) * 1000)

        return PageContent(
            url=url,
            title=title,
            content=content,
            format=output_format,
            fetch_mode=mode,
            word_count=len(content.split()),
            fetch_time_ms=fetch_time_ms,
        )

    async def _fetch_simple(self, url: str) -> tuple[str, str]:
        """Fetch using requests (no JS rendering)."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "Simple fetch requires: pip install requests beautifulsoup4"
            ) from e

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Cathedral/1.0)"}
            )
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Remove script/style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        return str(soup), title

    async def _fetch_headless(
        self,
        url: str,
        wait_for_selector: Optional[str] = None
    ) -> tuple[str, str]:
        """Fetch using Playwright (full JS rendering)."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                "Headless fetch requires: pip install playwright && playwright install chromium"
            ) from e

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=self.timeout_seconds * 1000)

                # Wait for specific element if requested
                if wait_for_selector:
                    await page.wait_for_selector(
                        wait_for_selector,
                        timeout=self.timeout_seconds * 1000
                    )
                else:
                    # Wait for network idle
                    await page.wait_for_load_state("networkidle")

                title = await page.title()
                html = await page.content()

                return html, title
            finally:
                await browser.close()

    def _convert_content(self, html: str, output_format: ContentFormat) -> str:
        """Convert HTML to requested format."""
        if output_format == ContentFormat.HTML:
            return html

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        if output_format == ContentFormat.TEXT:
            return soup.get_text(separator="\n", strip=True)

        # MARKDOWN format
        return self._html_to_markdown(soup)

    def _html_to_markdown(self, soup) -> str:
        """Convert BeautifulSoup object to markdown."""
        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.ignore_emphasis = False
            h.body_width = 0  # No wrapping
            return h.handle(str(soup))
        except ImportError:
            # Fallback to simple text extraction
            return soup.get_text(separator="\n", strip=True)


# Global fetcher instance
_fetcher: Optional[PageFetcher] = None


def get_fetcher() -> PageFetcher:
    """Get or create global PageFetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = PageFetcher()
    return _fetcher


async def fetch(url: str, **kwargs) -> PageContent:
    """Convenience function to fetch a page."""
    return await get_fetcher().fetch(url, **kwargs)
