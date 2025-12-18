"""Content fetcher tool for retrieving web page content."""

import httpx
from bs4 import BeautifulSoup

from .base import BaseTool


class ContentFetcherTool(BaseTool):
    """Tool for fetching and extracting content from web pages."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "fetch_content"

    @property
    def description(self) -> str:
        return (
            "Fetch the content of a web page and extract the main text. "
            "Useful for reading articles, documentation, or other web content."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to fetch",
                },
                "extract_links": {
                    "type": "boolean",
                    "description": "Whether to also extract links from the page",
                    "default": False,
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, extract_links: bool = False) -> dict:
        """
        Fetch and extract content from a URL.

        Args:
            url: The URL to fetch
            extract_links: Whether to extract links

        Returns:
            Dictionary with content and optionally links
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers={"User-Agent": "ResearchBot/1.0"},
                    timeout=float(self.timeout),
                    follow_redirects=True,
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                # Extract title
                title = soup.title.string if soup.title else "No title"

                # Extract main content
                # Try common content containers
                main_content = None
                for selector in ["article", "main", ".content", "#content", ".post"]:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break

                if not main_content:
                    main_content = soup.body

                text = main_content.get_text(separator="\n", strip=True) if main_content else ""

                # Limit text length
                max_length = 10000
                if len(text) > max_length:
                    text = text[:max_length] + "... [truncated]"

                result = {
                    "title": title,
                    "url": str(response.url),
                    "content": text,
                }

                if extract_links:
                    links = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        if href.startswith("http"):
                            links.append({
                                "text": a.get_text(strip=True)[:100],
                                "url": href,
                            })
                    result["links"] = links[:50]  # Limit to 50 links

                return result

            except httpx.HTTPError as e:
                return {"error": f"Failed to fetch URL: {str(e)}"}
            except Exception as e:
                return {"error": f"Error processing content: {str(e)}"}
