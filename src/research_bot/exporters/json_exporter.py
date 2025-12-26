"""
JSON exporter for research results.

LEARNING NOTE - JSON Export:
JSON (JavaScript Object Notation) is a universal data format.
It's great for:
- Storing data for later processing
- Sending data to APIs
- Importing into other tools
- Machine-readable output
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING

from .base import BaseExporter, ExportFormat

if TYPE_CHECKING:
    from ..agents.research_agent import ResearchResult


class JSONExporter(BaseExporter):
    """
    Export research results as JSON.

    Produces structured, machine-readable output perfect for:
    - Integration with other tools
    - Data analysis pipelines
    - Programmatic access to results
    """

    format = ExportFormat.JSON
    extension = "json"

    def __init__(self, pretty: bool = True, include_raw: bool = False):
        """
        Initialize JSON exporter.

        Args:
            pretty: If True, format JSON with indentation (human-readable)
            include_raw: If True, include raw_findings (can be large)
        """
        self.pretty = pretty
        self.include_raw = include_raw

    def _format_content(self, result: "ResearchResult") -> str:
        """Convert research result to JSON string."""
        data = result.to_dict()

        # Add export metadata
        data["export_metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "format": "json",
            "version": "1.0",
        }

        # Optionally remove large raw findings
        if not self.include_raw:
            data.pop("raw_findings", None)
        else:
            # Include raw findings if requested
            data["raw_findings"] = result.raw_findings

        # Format JSON
        if self.pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(data, ensure_ascii=False)
