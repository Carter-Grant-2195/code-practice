"""Semantic Scholar API tool for academic paper research.

Semantic Scholar API: https://api.semanticscholar.org/

Free tier: 100 requests per 5 minutes (no API key required)
With API key: higher limits available

Semantic Scholar is excellent for:
- Citation counts and metrics
- Citation networks (who cites whom)
- Finding influential papers
- Author information and h-index
"""

from typing import Any
from datetime import datetime

import httpx

from ..base import BaseTool
from .data_models import Paper, Author, SearchResults


class SemanticScholarTool(BaseTool):
    """Tool for searching papers on Semantic Scholar.

    Semantic Scholar provides rich citation data, making it ideal for:
    - Finding highly-cited papers
    - Building citation networks
    - Understanding research impact
    - Discovering related work
    """

    API_URL = "https://api.semanticscholar.org/graph/v1"

    # Fields to request from the API
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "citationCount",
        "referenceCount",
        "influentialCitationCount",
        "publicationDate",
        "authors",
        "externalIds",
        "url",
        "openAccessPdf",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
    ]

    def __init__(self, api_key: str | None = None, max_results: int = 20):
        """
        Initialize Semantic Scholar tool.

        Args:
            api_key: Optional API key for higher rate limits
            max_results: Maximum results per search
        """
        self.api_key = api_key
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "semantic_scholar_search"

    @property
    def description(self) -> str:
        return (
            "Search Semantic Scholar for academic papers. Excellent for finding "
            "highly-cited papers and understanding citation networks. Returns "
            "papers with citation counts, abstracts, and links to full text."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding papers",
                },
                "year": {
                    "type": "string",
                    "description": (
                        "Optional: filter by year or year range. "
                        "Examples: '2023', '2020-2023', '2020-' (2020 onwards)"
                    ),
                },
                "fields_of_study": {
                    "type": "string",
                    "description": (
                        "Optional: filter by field. Options: 'Computer Science', "
                        "'Mathematics', 'Physics', 'Biology', 'Medicine', etc."
                    ),
                },
                "min_citations": {
                    "type": "integer",
                    "description": "Optional: minimum citation count filter",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default: 10, max: 100)",
                    "default": 10,
                },
                "open_access_only": {
                    "type": "boolean",
                    "description": "Only return papers with open access PDFs",
                    "default": False,
                },
            },
            "required": ["query"],
        }

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {"User-Agent": "ResearchBot/1.0 (Academic Research)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _parse_paper(self, data: dict) -> Paper:
        """Parse Semantic Scholar API response into Paper object."""

        # Parse authors
        authors = []
        for author_data in data.get("authors", []):
            authors.append(Author(
                name=author_data.get("name", ""),
                author_id=author_data.get("authorId"),
            ))

        # Parse publication date
        published_date = None
        pub_date_str = data.get("publicationDate")
        if pub_date_str:
            try:
                published_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
            except ValueError:
                pass
        elif data.get("year"):
            try:
                published_date = datetime(int(data["year"]), 1, 1)
            except (ValueError, TypeError):
                pass

        # Get external IDs
        external_ids = data.get("externalIds", {}) or {}

        # Get fields of study as categories
        categories = []
        for field in data.get("s2FieldsOfStudy", []):
            if isinstance(field, dict):
                categories.append(field.get("category", ""))
            elif isinstance(field, str):
                categories.append(field)
        # Also add the high-level fields
        for field in data.get("fieldsOfStudy", []):
            if field and field not in categories:
                categories.append(field)

        # Get PDF URL
        pdf_url = None
        open_access = data.get("openAccessPdf")
        if open_access and isinstance(open_access, dict):
            pdf_url = open_access.get("url")

        return Paper(
            title=data.get("title", ""),
            authors=authors,
            abstract=data.get("abstract", "") or "",
            published_date=published_date,
            semantic_scholar_id=data.get("paperId"),
            arxiv_id=external_ids.get("ArXiv"),
            doi=external_ids.get("DOI"),
            pdf_url=pdf_url,
            abstract_url=data.get("url"),
            categories=categories,
            citation_count=data.get("citationCount"),
            reference_count=data.get("referenceCount"),
            influential_citation_count=data.get("influentialCitationCount"),
            source="semantic_scholar",
        )

    async def execute(
        self,
        query: str,
        year: str | None = None,
        fields_of_study: str | None = None,
        min_citations: int | None = None,
        max_results: int = 10,
        open_access_only: bool = False,
    ) -> dict[str, Any]:
        """
        Search Semantic Scholar for papers.

        Args:
            query: Search query
            year: Year or year range filter
            fields_of_study: Field filter
            min_citations: Minimum citations filter
            max_results: Number of results
            open_access_only: Only open access papers

        Returns:
            Dictionary with papers and metadata
        """
        max_results = min(max_results, min(self.max_results, 100))

        # Build query parameters
        params = {
            "query": query,
            "limit": max_results,
            "fields": ",".join(self.PAPER_FIELDS),
        }

        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        if open_access_only:
            params["openAccessPdf"] = ""  # Filter for papers with open access PDF

        url = f"{self.API_URL}/paper/search"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()

                papers = []
                for item in data.get("data", []):
                    try:
                        paper = self._parse_paper(item)

                        # Apply min_citations filter (API doesn't support this directly)
                        if min_citations and (paper.citation_count or 0) < min_citations:
                            continue

                        papers.append(paper)
                    except Exception:
                        continue

                # Compute summary statistics
                results = SearchResults(
                    query=query,
                    papers=papers,
                    total_results=data.get("total", len(papers)),
                    source="semantic_scholar",
                )

                # Citation statistics
                citation_counts = [p.citation_count or 0 for p in papers]
                avg_citations = sum(citation_counts) / len(citation_counts) if citation_counts else 0

                return {
                    "success": True,
                    "query": query,
                    "filters": {
                        "year": year,
                        "fields_of_study": fields_of_study,
                        "min_citations": min_citations,
                        "open_access_only": open_access_only,
                    },
                    "total_results": data.get("total", len(papers)),
                    "returned_results": len(papers),
                    "papers": [p.to_dict() for p in papers],
                    # Statistics for visualization
                    "citation_stats": {
                        "total_citations": sum(citation_counts),
                        "avg_citations": round(avg_citations, 1),
                        "max_citations": max(citation_counts) if citation_counts else 0,
                    },
                    "years_distribution": {
                        str(k): len(v) for k, v in results.get_papers_by_year().items()
                    },
                    "fields_distribution": results.get_category_counts(),
                }

            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                    "query": query,
                }


class SemanticScholarCitationsTool(BaseTool):
    """Tool for getting citations and references of a specific paper."""

    API_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "semantic_scholar_citations"

    @property
    def description(self) -> str:
        return (
            "Get papers that cite a specific paper, or papers referenced by it. "
            "Useful for understanding the impact of a paper and finding related work. "
            "Provide a paper ID (Semantic Scholar ID, DOI, or arXiv ID)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": (
                        "Paper identifier. Can be: Semantic Scholar ID, DOI (prefix with 'DOI:'), "
                        "arXiv ID (prefix with 'ARXIV:'), or URL to paper"
                    ),
                },
                "type": {
                    "type": "string",
                    "enum": ["citations", "references"],
                    "description": "Get papers that cite this paper ('citations') or papers this paper cites ('references')",
                    "default": "citations",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default: 20, max: 100)",
                    "default": 20,
                },
            },
            "required": ["paper_id"],
        }

    def _get_headers(self) -> dict:
        headers = {"User-Agent": "ResearchBot/1.0 (Academic Research)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def execute(
        self,
        paper_id: str,
        type: str = "citations",
        max_results: int = 20,
    ) -> dict[str, Any]:
        """Get citations or references for a paper."""

        max_results = min(max_results, 100)

        # Build endpoint
        endpoint = "citations" if type == "citations" else "references"
        url = f"{self.API_URL}/paper/{paper_id}/{endpoint}"

        params = {
            "limit": max_results,
            "fields": "paperId,title,abstract,year,citationCount,authors,url",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()

                papers = []
                for item in data.get("data", []):
                    # The actual paper is nested under "citingPaper" or "citedPaper"
                    paper_data = item.get("citingPaper" if type == "citations" else "citedPaper", {})
                    if paper_data:
                        papers.append({
                            "paper_id": paper_data.get("paperId"),
                            "title": paper_data.get("title"),
                            "abstract": (paper_data.get("abstract") or "")[:500],
                            "year": paper_data.get("year"),
                            "citation_count": paper_data.get("citationCount"),
                            "authors": [a.get("name") for a in paper_data.get("authors", [])],
                            "url": paper_data.get("url"),
                        })

                return {
                    "success": True,
                    "paper_id": paper_id,
                    "type": type,
                    "returned_results": len(papers),
                    "papers": papers,
                }

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return {
                        "success": False,
                        "error": f"Paper not found: {paper_id}",
                    }
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                }
            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                }


class SemanticScholarAuthorTool(BaseTool):
    """Tool for getting information about an author and their papers."""

    API_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "semantic_scholar_author"

    @property
    def description(self) -> str:
        return (
            "Get information about a researcher/author including their papers, "
            "citation count, h-index, and affiliations. Useful for understanding "
            "a researcher's body of work and impact."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "author_id": {
                    "type": "string",
                    "description": "Semantic Scholar author ID",
                },
            },
            "required": ["author_id"],
        }

    def _get_headers(self) -> dict:
        headers = {"User-Agent": "ResearchBot/1.0 (Academic Research)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def execute(self, author_id: str) -> dict[str, Any]:
        """Get author information and papers."""

        url = f"{self.API_URL}/author/{author_id}"
        params = {
            "fields": "name,affiliations,homepage,paperCount,citationCount,hIndex,papers.title,papers.year,papers.citationCount",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()

                # Get top papers
                papers = data.get("papers", [])
                top_papers = sorted(
                    papers,
                    key=lambda x: x.get("citationCount") or 0,
                    reverse=True
                )[:10]

                return {
                    "success": True,
                    "author_id": author_id,
                    "name": data.get("name"),
                    "affiliations": data.get("affiliations", []),
                    "homepage": data.get("homepage"),
                    "paper_count": data.get("paperCount"),
                    "citation_count": data.get("citationCount"),
                    "h_index": data.get("hIndex"),
                    "top_papers": [
                        {
                            "title": p.get("title"),
                            "year": p.get("year"),
                            "citations": p.get("citationCount"),
                        }
                        for p in top_papers
                    ],
                }

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return {
                        "success": False,
                        "error": f"Author not found: {author_id}",
                    }
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                }
            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                }


class SemanticScholarRecommendationsTool(BaseTool):
    """Tool for getting paper recommendations based on a paper."""

    API_URL = "https://api.semanticscholar.org/recommendations/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "semantic_scholar_recommendations"

    @property
    def description(self) -> str:
        return (
            "Get paper recommendations based on a seed paper. Returns papers that are "
            "similar or related to the given paper. Great for literature review and "
            "finding related work."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper ID to get recommendations for",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of recommendations (default: 10)",
                    "default": 10,
                },
            },
            "required": ["paper_id"],
        }

    def _get_headers(self) -> dict:
        headers = {"User-Agent": "ResearchBot/1.0 (Academic Research)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def execute(
        self,
        paper_id: str,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Get paper recommendations."""

        url = f"{self.API_URL}/papers/forpaper/{paper_id}"
        params = {
            "limit": min(max_results, 100),
            "fields": "paperId,title,abstract,year,citationCount,authors,url",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()

                papers = []
                for item in data.get("recommendedPapers", []):
                    papers.append({
                        "paper_id": item.get("paperId"),
                        "title": item.get("title"),
                        "abstract": (item.get("abstract") or "")[:500],
                        "year": item.get("year"),
                        "citation_count": item.get("citationCount"),
                        "authors": [a.get("name") for a in item.get("authors", [])],
                        "url": item.get("url"),
                    })

                return {
                    "success": True,
                    "seed_paper_id": paper_id,
                    "returned_results": len(papers),
                    "recommended_papers": papers,
                }

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return {
                        "success": False,
                        "error": f"Paper not found: {paper_id}",
                    }
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                }
            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                }
