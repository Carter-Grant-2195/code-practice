"""ArXiv search tool for academic paper research.

ArXiv API documentation: https://info.arxiv.org/help/api/index.html

The arXiv API is free and requires no authentication.
Rate limit: 1 request per 3 seconds (we add delays to be respectful)
"""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

import httpx

from ..base import BaseTool
from .data_models import Paper, Author, SearchResults


class ArxivSearchTool(BaseTool):
    """Tool for searching arXiv papers.

    Features:
    - Search by query (title, abstract, authors)
    - Filter by category (cs.AI, cs.LG, physics, math, etc.)
    - Sort by relevance, date, or citation count
    - Get paper metadata, abstracts, and PDF links
    """

    ARXIV_API_URL = "http://export.arxiv.org/api/query"

    # arXiv category codes for reference
    CATEGORIES = {
        # Computer Science
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Computation and Language (NLP)",
        "cs.CV": "Computer Vision",
        "cs.LG": "Machine Learning",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.RO": "Robotics",
        "cs.SE": "Software Engineering",
        # Statistics
        "stat.ML": "Machine Learning (Statistics)",
        # Math
        "math.OC": "Optimization and Control",
        # Physics
        "quant-ph": "Quantum Physics",
        # Quantitative Biology
        "q-bio.NC": "Neurons and Cognition",
        # Electrical Engineering
        "eess.SP": "Signal Processing",
    }

    def __init__(self, max_results: int = 20, rate_limit_delay: float = 1.0):
        """
        Initialize ArXiv search tool.

        Args:
            max_results: Maximum number of results per search
            rate_limit_delay: Seconds to wait between requests (arXiv asks for 3s)
        """
        self.max_results = max_results
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0

    @property
    def name(self) -> str:
        return "arxiv_search"

    @property
    def description(self) -> str:
        return (
            "Search arXiv for academic papers. Returns paper titles, authors, "
            "abstracts, categories, and PDF links. Supports filtering by "
            "category (e.g., cs.AI, cs.LG, cs.CL) and sorting by relevance or date."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (searches title, abstract, authors)",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional arXiv category to filter by (e.g., 'cs.AI', 'cs.LG', 'cs.CL', "
                        "'stat.ML'). Leave empty to search all categories."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 50)",
                    "default": 10,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort order for results (default: relevance)",
                    "default": "relevance",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["descending", "ascending"],
                    "description": "Sort direction (default: descending)",
                    "default": "descending",
                },
            },
            "required": ["query"],
        }

    async def _wait_for_rate_limit(self):
        """Ensure we don't exceed arXiv rate limits."""
        import time

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _build_query(
        self,
        query: str,
        category: str | None = None,
    ) -> str:
        """Build arXiv API query string.

        arXiv uses a specific query syntax:
        - all: search all fields
        - ti: title
        - au: author
        - abs: abstract
        - cat: category

        Examples:
        - all:transformer
        - cat:cs.AI AND all:attention
        """
        # Search all fields
        search_query = f"all:{query}"

        # Add category filter if specified
        if category:
            search_query = f"cat:{category} AND {search_query}"

        return search_query

    def _parse_entry(self, entry: ET.Element, ns: dict) -> Paper:
        """Parse a single arXiv entry into a Paper object."""

        def get_text(tag: str) -> str:
            elem = entry.find(tag, ns)
            return elem.text.strip() if elem is not None and elem.text else ""

        # Parse authors
        authors = []
        for author_elem in entry.findall("atom:author", ns):
            name_elem = author_elem.find("atom:name", ns)
            affil_elem = author_elem.find("arxiv:affiliation", ns)
            if name_elem is not None and name_elem.text:
                authors.append(Author(
                    name=name_elem.text,
                    affiliation=affil_elem.text if affil_elem is not None else None,
                ))

        # Parse categories
        categories = []
        primary_category = None
        for cat_elem in entry.findall("atom:category", ns):
            term = cat_elem.get("term")
            if term:
                categories.append(term)
        # Primary category from arxiv namespace
        primary_cat_elem = entry.find("arxiv:primary_category", ns)
        if primary_cat_elem is not None:
            primary_category = primary_cat_elem.get("term")

        # Parse dates
        published_str = get_text("atom:published")
        updated_str = get_text("atom:updated")

        published_date = None
        updated_date = None
        try:
            if published_str:
                published_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            if updated_str:
                updated_date = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        # Extract arXiv ID from the entry ID URL
        entry_id = get_text("atom:id")
        arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id

        # Get PDF and abstract URLs
        pdf_url = None
        abstract_url = None
        for link_elem in entry.findall("atom:link", ns):
            link_type = link_elem.get("type", "")
            link_title = link_elem.get("title", "")
            href = link_elem.get("href", "")

            if link_title == "pdf" or link_type == "application/pdf":
                pdf_url = href
            elif link_type == "text/html":
                abstract_url = href

        # Get DOI if available
        doi = None
        doi_elem = entry.find("arxiv:doi", ns)
        if doi_elem is not None and doi_elem.text:
            doi = doi_elem.text

        return Paper(
            title=get_text("atom:title").replace("\n", " "),
            authors=authors,
            abstract=get_text("atom:summary").replace("\n", " "),
            published_date=published_date,
            updated_date=updated_date,
            arxiv_id=arxiv_id,
            doi=doi,
            pdf_url=pdf_url,
            abstract_url=abstract_url or entry_id,
            categories=categories,
            primary_category=primary_category,
            source="arxiv",
        )

    async def execute(
        self,
        query: str,
        category: str | None = None,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> dict[str, Any]:
        """
        Execute arXiv search.

        Args:
            query: Search query
            category: Optional category filter (e.g., "cs.AI")
            max_results: Number of results (max 50)
            sort_by: Sort field ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort direction ("descending", "ascending")

        Returns:
            Dictionary with papers and metadata
        """
        await self._wait_for_rate_limit()

        # Clamp max_results
        max_results = min(max_results, min(self.max_results, 50))

        # Build query parameters
        search_query = self._build_query(query, category)

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        url = f"{self.ARXIV_API_URL}?{urlencode(params)}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers={"User-Agent": "ResearchBot/1.0 (Academic Research)"},
                    timeout=30.0,
                )
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.text)

                # Define namespaces
                ns = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom",
                    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
                }

                # Get total results
                total_elem = root.find("opensearch:totalResults", ns)
                total_results = int(total_elem.text) if total_elem is not None and total_elem.text else 0

                # Parse entries
                papers = []
                for entry in root.findall("atom:entry", ns):
                    try:
                        paper = self._parse_entry(entry, ns)
                        papers.append(paper)
                    except Exception as e:
                        # Skip malformed entries
                        continue

                results = SearchResults(
                    query=query,
                    papers=papers,
                    total_results=total_results,
                    offset=0,
                    limit=max_results,
                    source="arxiv",
                )

                return {
                    "success": True,
                    "query": query,
                    "category_filter": category,
                    "total_results": total_results,
                    "returned_results": len(papers),
                    "papers": [p.to_dict() for p in papers],
                    # Include summary stats for quick overview
                    "categories_found": results.get_category_counts(),
                    "years_distribution": {
                        str(k): len(v) for k, v in results.get_papers_by_year().items()
                    },
                }

            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                    "query": query,
                }
            except ET.ParseError as e:
                return {
                    "success": False,
                    "error": f"XML parse error: {str(e)}",
                    "query": query,
                }


class ArxivPaperFetchTool(BaseTool):
    """Tool for fetching detailed information about a specific arXiv paper."""

    ARXIV_API_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self._last_request_time: float = 0

    @property
    def name(self) -> str:
        return "arxiv_paper_details"

    @property
    def description(self) -> str:
        return (
            "Get detailed information about a specific arXiv paper by its ID. "
            "Returns full metadata including abstract, authors, categories, and links."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": (
                        "The arXiv paper ID (e.g., '2301.07041' or 'cs/0001001' for older papers)"
                    ),
                },
            },
            "required": ["arxiv_id"],
        }

    async def execute(self, arxiv_id: str) -> dict[str, Any]:
        """Fetch details for a specific arXiv paper."""
        # Clean the ID (remove any arxiv: prefix or version suffix if needed)
        clean_id = arxiv_id.replace("arxiv:", "").strip()

        params = {"id_list": clean_id}
        url = f"{self.ARXIV_API_URL}?{urlencode(params)}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers={"User-Agent": "ResearchBot/1.0 (Academic Research)"},
                    timeout=30.0,
                )
                response.raise_for_status()

                root = ET.fromstring(response.text)
                ns = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom",
                }

                entry = root.find("atom:entry", ns)
                if entry is None:
                    return {
                        "success": False,
                        "error": f"Paper not found: {arxiv_id}",
                    }

                # Reuse parsing logic from search tool
                search_tool = ArxivSearchTool()
                paper = search_tool._parse_entry(entry, ns)

                return {
                    "success": True,
                    "paper": paper.to_dict(),
                }

            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                }
