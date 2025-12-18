"""HuggingFace Datasets Hub search tool.

HuggingFace Hub API: https://huggingface.co/docs/hub/api

The API is free and requires no authentication for public datasets.
"""

from typing import Any

import httpx

from ..base import BaseTool
from .data_models import Dataset, SearchResults


class HuggingFaceDatasetsTool(BaseTool):
    """Tool for searching and exploring HuggingFace datasets.

    Features:
    - Search datasets by keyword
    - Filter by task, language, size
    - Get detailed dataset information
    - View dataset structure and splits
    """

    HF_API_URL = "https://huggingface.co/api"

    # Common task categories for reference
    TASK_CATEGORIES = [
        "text-classification",
        "token-classification",
        "question-answering",
        "summarization",
        "translation",
        "text-generation",
        "text2text-generation",
        "fill-mask",
        "sentence-similarity",
        "image-classification",
        "object-detection",
        "image-segmentation",
        "audio-classification",
        "automatic-speech-recognition",
        "reinforcement-learning",
        "tabular-classification",
        "tabular-regression",
    ]

    def __init__(self, max_results: int = 20):
        """
        Initialize HuggingFace datasets tool.

        Args:
            max_results: Maximum number of results per search
        """
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "huggingface_datasets"

    @property
    def description(self) -> str:
        return (
            "Search HuggingFace Hub for datasets. Returns dataset names, descriptions, "
            "sizes, tasks they support, and download statistics. Useful for finding "
            "training data for machine learning projects."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (searches dataset names and descriptions)",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Optional: filter by ML task (e.g., 'text-classification', "
                        "'question-answering', 'summarization', 'translation', "
                        "'image-classification', 'object-detection')"
                    ),
                },
                "language": {
                    "type": "string",
                    "description": "Optional: filter by language code (e.g., 'en', 'zh', 'es')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10, max: 50)",
                    "default": 10,
                },
                "sort": {
                    "type": "string",
                    "enum": ["downloads", "likes", "created", "modified"],
                    "description": "Sort by field (default: downloads)",
                    "default": "downloads",
                },
            },
            "required": ["query"],
        }

    def _parse_dataset(self, data: dict) -> Dataset:
        """Parse HuggingFace API response into Dataset object."""

        # Extract card data if available
        card_data = data.get("cardData", {}) or {}

        # Get size info
        num_rows = {}
        if "splits" in card_data:
            for split_name, split_info in card_data.get("splits", {}).items():
                if isinstance(split_info, dict) and "num_examples" in split_info:
                    num_rows[split_name] = split_info["num_examples"]

        # Get language
        language = card_data.get("language", [])
        if isinstance(language, str):
            language = [language]

        return Dataset(
            name=data.get("id", "").split("/")[-1],  # Get name without org prefix
            description=card_data.get("description", data.get("description", ""))[:500],
            dataset_id=data.get("id", ""),
            url=f"https://huggingface.co/datasets/{data.get('id', '')}",
            author=data.get("author", ""),
            license=card_data.get("license", ""),
            language=language,
            num_rows=num_rows,
            task_categories=card_data.get("task_categories", []),
            task_ids=card_data.get("task_ids", []),
            tags=data.get("tags", []),
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            splits=list(num_rows.keys()) if num_rows else [],
            source="huggingface",
        )

    async def execute(
        self,
        query: str,
        task: str | None = None,
        language: str | None = None,
        max_results: int = 10,
        sort: str = "downloads",
    ) -> dict[str, Any]:
        """
        Search HuggingFace datasets.

        Args:
            query: Search query
            task: Optional task filter
            language: Optional language filter
            max_results: Number of results
            sort: Sort field

        Returns:
            Dictionary with datasets and metadata
        """
        max_results = min(max_results, min(self.max_results, 50))

        # Build query parameters
        params = {
            "search": query,
            "limit": max_results,
            "sort": sort,
            "direction": -1,  # Descending
            "full": "true",   # Get full dataset info
        }

        if task:
            params["task_categories"] = task
        if language:
            params["language"] = language

        url = f"{self.HF_API_URL}/datasets"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers={"User-Agent": "ResearchBot/1.0 (Academic Research)"},
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()

                datasets = []
                for item in data:
                    try:
                        dataset = self._parse_dataset(item)
                        datasets.append(dataset)
                    except Exception:
                        continue

                # Compute summary stats
                task_counts: dict[str, int] = {}
                total_downloads = 0
                for ds in datasets:
                    total_downloads += ds.downloads or 0
                    for task_cat in ds.task_categories:
                        task_counts[task_cat] = task_counts.get(task_cat, 0) + 1

                return {
                    "success": True,
                    "query": query,
                    "filters": {
                        "task": task,
                        "language": language,
                    },
                    "returned_results": len(datasets),
                    "datasets": [d.to_dict() for d in datasets],
                    # Summary stats for visualization
                    "task_distribution": dict(sorted(
                        task_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )),
                    "total_downloads": total_downloads,
                }

            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                    "query": query,
                }


class HuggingFaceDatasetDetailsTool(BaseTool):
    """Tool for getting detailed information about a specific HuggingFace dataset."""

    HF_API_URL = "https://huggingface.co/api"

    @property
    def name(self) -> str:
        return "huggingface_dataset_details"

    @property
    def description(self) -> str:
        return (
            "Get detailed information about a specific HuggingFace dataset, including "
            "its structure, features, splits, and example data. Use this after finding "
            "a dataset with huggingface_datasets to get more details."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": (
                        "The dataset ID (e.g., 'squad', 'glue', 'openai/gsm8k', 'allenai/c4')"
                    ),
                },
            },
            "required": ["dataset_id"],
        }

    async def execute(self, dataset_id: str) -> dict[str, Any]:
        """Get detailed information about a dataset."""

        async with httpx.AsyncClient() as client:
            try:
                # Get dataset info
                info_url = f"{self.HF_API_URL}/datasets/{dataset_id}"
                info_response = await client.get(
                    info_url,
                    headers={"User-Agent": "ResearchBot/1.0 (Academic Research)"},
                    timeout=30.0,
                )
                info_response.raise_for_status()
                info_data = info_response.json()

                # Try to get parquet info for structure details
                parquet_url = f"{self.HF_API_URL}/datasets/{dataset_id}/parquet"
                parquet_data = None
                try:
                    parquet_response = await client.get(
                        parquet_url,
                        headers={"User-Agent": "ResearchBot/1.0"},
                        timeout=10.0,
                    )
                    if parquet_response.status_code == 200:
                        parquet_data = parquet_response.json()
                except Exception:
                    pass

                # Parse dataset info
                card_data = info_data.get("cardData", {}) or {}

                # Get splits info
                splits_info = {}
                if parquet_data and isinstance(parquet_data, dict):
                    for config_name, config_splits in parquet_data.items():
                        if isinstance(config_splits, dict):
                            for split_name, files in config_splits.items():
                                if isinstance(files, list):
                                    splits_info[f"{config_name}/{split_name}"] = {
                                        "num_files": len(files),
                                    }

                # Try to get README content
                readme_content = None
                try:
                    readme_url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
                    readme_response = await client.get(
                        readme_url,
                        headers={"User-Agent": "ResearchBot/1.0"},
                        timeout=10.0,
                    )
                    if readme_response.status_code == 200:
                        # Get first 2000 chars of README
                        readme_content = readme_response.text[:2000]
                except Exception:
                    pass

                return {
                    "success": True,
                    "dataset_id": dataset_id,
                    "name": info_data.get("id", "").split("/")[-1],
                    "author": info_data.get("author", ""),
                    "description": card_data.get("description", ""),
                    "license": card_data.get("license", ""),
                    "language": card_data.get("language", []),
                    "task_categories": card_data.get("task_categories", []),
                    "tags": info_data.get("tags", []),
                    "downloads": info_data.get("downloads", 0),
                    "likes": info_data.get("likes", 0),
                    "created_at": info_data.get("createdAt", ""),
                    "last_modified": info_data.get("lastModified", ""),
                    "splits": splits_info,
                    "dataset_url": f"https://huggingface.co/datasets/{dataset_id}",
                    "viewer_url": f"https://huggingface.co/datasets/{dataset_id}/viewer",
                    "readme_excerpt": readme_content,
                    # Citation info if available
                    "citation": card_data.get("citation", ""),
                }

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return {
                        "success": False,
                        "error": f"Dataset not found: {dataset_id}",
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


class HuggingFaceModelsTool(BaseTool):
    """Tool for searching HuggingFace models (useful for finding pretrained models)."""

    HF_API_URL = "https://huggingface.co/api"

    @property
    def name(self) -> str:
        return "huggingface_models"

    @property
    def description(self) -> str:
        return (
            "Search HuggingFace Hub for pretrained models. Returns model names, "
            "architectures, tasks, and download statistics. Useful for finding "
            "models to use or fine-tune for ML projects."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (searches model names and descriptions)",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Optional: filter by ML task (e.g., 'text-classification', "
                        "'text-generation', 'image-classification')"
                    ),
                },
                "library": {
                    "type": "string",
                    "description": (
                        "Optional: filter by ML library (e.g., 'transformers', 'pytorch', "
                        "'tensorflow', 'jax')"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        task: str | None = None,
        library: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Search HuggingFace models."""

        params = {
            "search": query,
            "limit": min(max_results, 50),
            "sort": "downloads",
            "direction": -1,
        }

        if task:
            params["pipeline_tag"] = task
        if library:
            params["library"] = library

        url = f"{self.HF_API_URL}/models"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers={"User-Agent": "ResearchBot/1.0 (Academic Research)"},
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()

                models = []
                for item in data:
                    models.append({
                        "model_id": item.get("id", ""),
                        "author": item.get("author", ""),
                        "pipeline_tag": item.get("pipeline_tag", ""),
                        "tags": item.get("tags", []),
                        "downloads": item.get("downloads", 0),
                        "likes": item.get("likes", 0),
                        "library_name": item.get("library_name", ""),
                        "url": f"https://huggingface.co/{item.get('id', '')}",
                    })

                return {
                    "success": True,
                    "query": query,
                    "filters": {"task": task, "library": library},
                    "returned_results": len(models),
                    "models": models,
                }

            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                    "query": query,
                }
