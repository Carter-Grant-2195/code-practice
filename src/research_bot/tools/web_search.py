"""Web search tool using DuckDuckGo (no API key required)."""

import re
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from .base import BaseTool


class WebSearchTool(BaseTool):
    """
    Search the web via DuckDuckGo HTML interface.

    Note: For production use, consider a proper search API (Google, Bing, SerpAPI)
    for better reliability and structured results.
    """

    SEARCH_URL = "https://html.duckduckgo.com/html/"

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information. Returns titles, URLs, and snippets."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        """
        Execute web search and parse results from DuckDuckGo HTML.

        Returns list of {title, url, snippet} dicts.
        """
        num_results = min(num_results, self.max_results)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.SEARCH_URL,
                    data={"q": query},
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    timeout=15.0,
                )
                response.raise_for_status()
                return self._parse_results(response.text, num_results)

            except httpx.HTTPError as e:
                return [{"error": f"Search failed: {e}"}]

    def _parse_results(self, html: str, limit: int) -> list[dict[str, str]]:
        """Extract search results from DuckDuckGo HTML response."""
        soup = BeautifulSoup(html, "html.parser")
        results = []

        # DuckDuckGo wraps each result in a div with class "result"
        for result_div in soup.select(".result")[:limit]:
            # Title and URL are in the .result__a link
            link = result_div.select_one(".result__a")
            if not link:
                continue

            title = link.get_text(strip=True)
            url = link.get("href", "")

            # Extract actual URL from DuckDuckGo redirect
            # Format: //duckduckgo.com/l/?uddg=ENCODED_URL&...
            if "uddg=" in url:
                match = re.search(r"uddg=([^&]+)", url)
                if match:
                    from urllib.parse import unquote
                    url = unquote(match.group(1))

            # Snippet is in .result__snippet
            snippet_elem = result_div.select_one(".result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            if title and url.startswith("http"):
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet[:300],  # Truncate long snippets
                })

        return results if results else [{"error": "No results found"}]
