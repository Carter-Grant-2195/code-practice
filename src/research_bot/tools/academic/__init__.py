"""Academic and scientific research tools."""

from .arxiv_search import ArxivSearchTool, ArxivPaperFetchTool
from .huggingface_datasets import (
    HuggingFaceDatasetsTool,
    HuggingFaceDatasetDetailsTool,
    HuggingFaceModelsTool,
)
from .semantic_scholar import (
    SemanticScholarTool,
    SemanticScholarCitationsTool,
    SemanticScholarAuthorTool,
    SemanticScholarRecommendationsTool,
)
from .data_models import Paper, Dataset, Author, SearchResults

__all__ = [
    # arXiv
    "ArxivSearchTool",
    "ArxivPaperFetchTool",
    # HuggingFace
    "HuggingFaceDatasetsTool",
    "HuggingFaceDatasetDetailsTool",
    "HuggingFaceModelsTool",
    # Semantic Scholar
    "SemanticScholarTool",
    "SemanticScholarCitationsTool",
    "SemanticScholarAuthorTool",
    "SemanticScholarRecommendationsTool",
    # Data models
    "Paper",
    "Dataset",
    "Author",
    "SearchResults",
]
