"""Data models for academic research results.

These models provide structured data that can be easily used for:
- Visualization (charts, graphs, timelines)
- PDF report generation
- Data analysis and comparison
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Author:
    """Represents a paper/dataset author."""

    name: str
    affiliation: str | None = None
    author_id: str | None = None  # Platform-specific ID (arXiv, Semantic Scholar, etc.)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "affiliation": self.affiliation,
            "author_id": self.author_id,
        }


@dataclass
class Paper:
    """Represents an academic paper with structured metadata.

    This structure is designed to support:
    - Citation network visualization
    - Timeline charts (publication dates)
    - Category/topic analysis
    - Author collaboration graphs
    """

    title: str
    authors: list[Author]
    abstract: str
    published_date: datetime | None = None
    updated_date: datetime | None = None

    # Identifiers
    arxiv_id: str | None = None
    doi: str | None = None
    semantic_scholar_id: str | None = None

    # URLs
    pdf_url: str | None = None
    abstract_url: str | None = None

    # Classification
    categories: list[str] = field(default_factory=list)  # e.g., ["cs.AI", "cs.LG"]
    primary_category: str | None = None

    # Metrics (useful for visualization)
    citation_count: int | None = None
    reference_count: int | None = None
    influential_citation_count: int | None = None

    # Related papers (for network graphs)
    references: list[str] = field(default_factory=list)  # List of paper IDs
    citations: list[str] = field(default_factory=list)   # Papers that cite this one

    # Source tracking
    source: str = ""  # "arxiv", "semantic_scholar", etc.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "authors": [a.to_dict() for a in self.authors],
            "abstract": self.abstract,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "updated_date": self.updated_date.isoformat() if self.updated_date else None,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "semantic_scholar_id": self.semantic_scholar_id,
            "pdf_url": self.pdf_url,
            "abstract_url": self.abstract_url,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "influential_citation_count": self.influential_citation_count,
            "source": self.source,
        }

    def get_year(self) -> int | None:
        """Extract publication year for timeline visualizations."""
        if self.published_date:
            return self.published_date.year
        return None

    def get_author_names(self) -> list[str]:
        """Get list of author names for display."""
        return [a.name for a in self.authors]


@dataclass
class Dataset:
    """Represents a dataset from HuggingFace or similar platforms.

    This structure supports:
    - Size comparison charts
    - Task/domain categorization
    - Download/popularity metrics
    """

    name: str
    description: str

    # Identifiers
    dataset_id: str  # e.g., "squad", "glue", "imdb"

    # URLs
    url: str | None = None
    homepage: str | None = None

    # Metadata
    author: str | None = None
    license: str | None = None
    language: list[str] = field(default_factory=list)

    # Size metrics (for comparison visualizations)
    size_bytes: int | None = None
    num_rows: dict[str, int] = field(default_factory=dict)  # {"train": 10000, "test": 1000}
    num_features: int | None = None

    # Classification
    task_categories: list[str] = field(default_factory=list)  # ["text-classification", "qa"]
    task_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Popularity metrics
    downloads: int | None = None
    likes: int | None = None

    # Features schema (useful for understanding dataset structure)
    features: dict[str, Any] = field(default_factory=dict)

    # Splits available
    splits: list[str] = field(default_factory=list)  # ["train", "test", "validation"]

    # Source
    source: str = "huggingface"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "dataset_id": self.dataset_id,
            "url": self.url,
            "homepage": self.homepage,
            "author": self.author,
            "license": self.license,
            "language": self.language,
            "size_bytes": self.size_bytes,
            "num_rows": self.num_rows,
            "num_features": self.num_features,
            "task_categories": self.task_categories,
            "task_ids": self.task_ids,
            "tags": self.tags,
            "downloads": self.downloads,
            "likes": self.likes,
            "features": self.features,
            "splits": self.splits,
            "source": self.source,
        }

    def get_total_rows(self) -> int:
        """Get total number of rows across all splits."""
        return sum(self.num_rows.values())

    def get_size_mb(self) -> float | None:
        """Get size in megabytes for display."""
        if self.size_bytes:
            return self.size_bytes / (1024 * 1024)
        return None


@dataclass
class SearchResults:
    """Container for search results with metadata.

    Useful for:
    - Pagination handling
    - Result aggregation across sources
    - Search analytics
    """

    query: str
    papers: list[Paper] = field(default_factory=list)
    datasets: list[Dataset] = field(default_factory=list)

    # Pagination
    total_results: int = 0
    offset: int = 0
    limit: int = 10

    # Metadata
    source: str = ""
    search_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "papers": [p.to_dict() for p in self.papers],
            "datasets": [d.to_dict() for d in self.datasets],
            "total_results": self.total_results,
            "offset": self.offset,
            "limit": self.limit,
            "source": self.source,
            "search_time_ms": self.search_time_ms,
        }

    def get_papers_by_year(self) -> dict[int, list[Paper]]:
        """Group papers by publication year for timeline visualization."""
        by_year: dict[int, list[Paper]] = {}
        for paper in self.papers:
            year = paper.get_year()
            if year:
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(paper)
        return dict(sorted(by_year.items()))

    def get_category_counts(self) -> dict[str, int]:
        """Count papers by category for bar chart visualization."""
        counts: dict[str, int] = {}
        for paper in self.papers:
            for cat in paper.categories:
                counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def get_top_authors(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most prolific authors for visualization."""
        author_counts: dict[str, int] = {}
        for paper in self.papers:
            for author in paper.authors:
                author_counts[author.name] = author_counts.get(author.name, 0) + 1
        sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_authors[:limit]
