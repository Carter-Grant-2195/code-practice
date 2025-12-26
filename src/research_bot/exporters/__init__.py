"""Export utilities for research results."""

from .base import BaseExporter, ExportFormat
from .json_exporter import JSONExporter
from .markdown_exporter import MarkdownExporter
from .html_exporter import HTMLExporter
from .csv_exporter import CSVExporter

__all__ = [
    "BaseExporter",
    "ExportFormat",
    "JSONExporter",
    "MarkdownExporter",
    "HTMLExporter",
    "CSVExporter",
]


def get_exporter(format: str) -> BaseExporter:
    """
    Factory function to get the appropriate exporter.

    LEARNING NOTE - Factory Pattern:
    A factory function creates objects based on input.
    This lets you switch exporters without changing calling code.

    Usage:
        exporter = get_exporter("json")
        exporter.export(result, "output.json")
    """
    exporters = {
        "json": JSONExporter,
        "markdown": MarkdownExporter,
        "md": MarkdownExporter,
        "html": HTMLExporter,
        "csv": CSVExporter,
    }

    format_lower = format.lower()
    if format_lower not in exporters:
        valid = ", ".join(sorted(set(exporters.keys())))
        raise ValueError(f"Unknown format '{format}'. Valid formats: {valid}")

    return exporters[format_lower]()
