"""Tool for fetching and extracting readable content from web pages."""

import httpx
from bs4 import BeautifulSoup

from .base import BaseTool


class ContentFetcherTool(BaseTool):
    """
    Fetch a URL and extract its main text content.

    Strips navigation, ads, and other non-content elements to return
    clean readable text suitable for analysis.
    """

    # CSS selectors for main content (tried in order)
    CONTENT_SELECTORS = ["article", "main", "[role='main']", ".content", "#content", ".post", ".entry"]

    # Elements to remove (noise)
    NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]

    def __init__(self, timeout: int = 30, max_length: int = 10000):
        self.timeout = timeout
        self.max_length = max_length

    @property
    def name(self) -> str:
        return "fetch_content"

    @property
    def description(self) -> str:
        return "Fetch a web page and extract its main text content for reading and analysis."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "extract_links": {
                    "type": "boolean",
                    "description": "Also return links found on the page",
                    "default": False,
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, extract_links: bool = False) -> dict:
        """Fetch URL and return extracted content."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers={"User-Agent": "ResearchBot/1.0 (Content Fetcher)"},
                    timeout=float(self.timeout),
                    follow_redirects=True,
                )
                response.raise_for_status()
                return self._extract_content(response.text, str(response.url), extract_links)

            except httpx.HTTPError as e:
                return {"error": f"Failed to fetch: {e}"}
            except Exception as e:
                return {"error": f"Processing error: {e}"}

    def _extract_content(self, html: str, url: str, extract_links: bool) -> dict:
        """Parse HTML and extract readable content."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup(self.NOISE_TAGS):
            tag.decompose()

        # Get title
        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"

        # Find main content container
        content_elem = None
        for selector in self.CONTENT_SELECTORS:
            content_elem = soup.select_one(selector)
            if content_elem:
                break

        # Fall back to body if no content container found
        if not content_elem:
            content_elem = soup.body

        # Extract text
        text = content_elem.get_text(separator="\n", strip=True) if content_elem else ""

        # Truncate if needed
        if len(text) > self.max_length:
            text = text[:self.max_length] + "\n\n[Content truncated...]"

        result = {"title": title, "url": url, "content": text}

        # Optionally extract links
        if extract_links:
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    links.append({
                        "text": a.get_text(strip=True)[:100],
                        "url": href,
                    })
            result["links"] = links[:50]  # Cap at 50 links

        return result
