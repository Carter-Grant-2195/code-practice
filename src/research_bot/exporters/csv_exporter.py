"""
CSV exporter for research results.

LEARNING NOTE - CSV Export:
CSV (Comma-Separated Values) is ideal for:
- Opening in Excel/Google Sheets
- Importing into databases
- Data analysis with pandas
- Simple, universal format
"""

import csv
import io
from datetime import datetime
from typing import TYPE_CHECKING

from .base import BaseExporter, ExportFormat

if TYPE_CHECKING:
    from ..agents.research_agent import ResearchResult


class CSVExporter(BaseExporter):
    """
    Export research results as CSV.

    Produces tabular data perfect for:
    - Spreadsheet analysis
    - Importing into databases
    - Data processing pipelines
    - Comparison across multiple research runs

    Note: CSV is inherently flat, so we export multiple "views":
    - Main summary row
    - Sources table
    - Extracted specifications table
    - Statistics table
    """

    format = ExportFormat.CSV
    extension = "csv"

    def __init__(self, view: str = "summary"):
        """
        Initialize CSV exporter.

        Args:
            view: Which view to export:
                - "summary": Single row with main research info
                - "sources": All sources as rows
                - "specifications": Extracted specs as rows
                - "statistics": Extracted stats as rows
                - "all": Combined export with section headers
        """
        self.view = view

    def _format_content(self, result: "ResearchResult") -> str:
        """Convert research result to CSV string."""
        output = io.StringIO()

        if self.view == "summary":
            self._write_summary(output, result)
        elif self.view == "sources":
            self._write_sources(output, result)
        elif self.view == "specifications":
            self._write_specifications(output, result)
        elif self.view == "statistics":
            self._write_statistics(output, result)
        elif self.view == "all":
            self._write_all(output, result)
        else:
            self._write_summary(output, result)

        return output.getvalue()

    def _write_summary(self, output: io.StringIO, result: "ResearchResult") -> None:
        """Write summary view."""
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Query", "Status", "Iterations", "Sources",
            "Model", "Input Tokens", "Output Tokens", "Cost USD",
            "Exported At", "Summary"
        ])

        # Data row
        writer.writerow([
            result.query,
            "Complete" if result.completed else "Partial",
            result.iterations,
            len(result.sources),
            result.usage.model,
            result.usage.input_tokens,
            result.usage.output_tokens,
            f"{result.usage.cost_usd:.4f}",
            datetime.now().isoformat(),
            result.summary[:500] + "..." if len(result.summary) > 500 else result.summary
        ])

    def _write_sources(self, output: io.StringIO, result: "ResearchResult") -> None:
        """Write sources view."""
        writer = csv.writer(output)

        # Header
        writer.writerow(["Index", "Title", "URL", "Query"])

        # Data rows
        for i, source in enumerate(result.sources, 1):
            writer.writerow([
                i,
                source.get("title", "Untitled"),
                source.get("url", ""),
                result.query
            ])

    def _write_specifications(self, output: io.StringIO, result: "ResearchResult") -> None:
        """Write specifications view."""
        writer = csv.writer(output)

        # Header
        writer.writerow(["Specification", "Value", "Query"])

        # Data rows
        for key, value in result.extracted_data.specifications.items():
            writer.writerow([key, value, result.query])

    def _write_statistics(self, output: io.StringIO, result: "ResearchResult") -> None:
        """Write statistics view."""
        writer = csv.writer(output)

        # Header
        writer.writerow(["Statistic", "Context", "Query"])

        # Data rows
        for stat, context in result.extracted_data.statistics.items():
            writer.writerow([stat, context, result.query])

    def _write_all(self, output: io.StringIO, result: "ResearchResult") -> None:
        """Write all data with section markers."""
        writer = csv.writer(output)
        timestamp = datetime.now().isoformat()

        # === Summary Section ===
        writer.writerow(["=== SUMMARY ==="])
        writer.writerow(["Property", "Value"])
        writer.writerow(["Query", result.query])
        writer.writerow(["Status", "Complete" if result.completed else "Partial"])
        writer.writerow(["Iterations", result.iterations])
        writer.writerow(["Sources Count", len(result.sources)])
        writer.writerow(["Model", result.usage.model])
        writer.writerow(["Input Tokens", result.usage.input_tokens])
        writer.writerow(["Output Tokens", result.usage.output_tokens])
        writer.writerow(["Total Tokens", result.usage.total_tokens])
        writer.writerow(["Cost USD", f"${result.usage.cost_usd:.4f}"])
        writer.writerow(["Exported At", timestamp])
        writer.writerow([])

        # === Key Findings Section ===
        if result.key_findings:
            writer.writerow(["=== KEY FINDINGS ==="])
            writer.writerow(["Finding"])
            for finding in result.key_findings:
                writer.writerow([finding])
            writer.writerow([])

        # === Sources Section ===
        writer.writerow(["=== SOURCES ==="])
        writer.writerow(["Index", "Title", "URL"])
        for i, source in enumerate(result.sources, 1):
            writer.writerow([
                i,
                source.get("title", "Untitled"),
                source.get("url", "")
            ])
        writer.writerow([])

        # === Specifications Section ===
        if result.extracted_data.specifications:
            writer.writerow(["=== SPECIFICATIONS ==="])
            writer.writerow(["Specification", "Value"])
            for key, value in result.extracted_data.specifications.items():
                writer.writerow([key, value])
            writer.writerow([])

        # === Statistics Section ===
        if result.extracted_data.statistics:
            writer.writerow(["=== STATISTICS ==="])
            writer.writerow(["Statistic", "Context"])
            for stat, context in list(result.extracted_data.statistics.items())[:20]:
                writer.writerow([stat, context])
            writer.writerow([])

        # === Prices Section ===
        if result.extracted_data.prices:
            writer.writerow(["=== PRICING ==="])
            writer.writerow(["Item", "Price"])
            for item, price in result.extracted_data.prices.items():
                writer.writerow([item, price])
            writer.writerow([])

        # === Versions Section ===
        if result.extracted_data.versions:
            writer.writerow(["=== VERSIONS ==="])
            writer.writerow(["Version"])
            for version in result.extracted_data.versions[:15]:
                writer.writerow([version])
            writer.writerow([])

        # === Dates Section ===
        if result.extracted_data.dates:
            writer.writerow(["=== DATES ==="])
            writer.writerow(["Date"])
            for date in result.extracted_data.dates[:15]:
                writer.writerow([date])
