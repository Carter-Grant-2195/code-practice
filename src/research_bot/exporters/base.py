"""
Base exporter class for research results.

LEARNING NOTE - Abstract Base Class:
This defines the interface all exporters must follow.
Each exporter (JSON, Markdown, etc.) implements this interface.
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

# LEARNING NOTE - TYPE_CHECKING:
# This import only runs during type checking (mypy), not at runtime.
# It prevents circular imports while still allowing type hints.
if TYPE_CHECKING:
    from ..agents.research_agent import ResearchResult


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


class BaseExporter(ABC):
    """
    Abstract base class for all exporters.

    LEARNING NOTE - Template Method Pattern:
    The export() method is the template - it handles common logic.
    Subclasses implement _format_content() for format-specific logic.
    """

    # Each subclass sets its format
    format: ExportFormat

    # File extension for this format
    extension: str

    @abstractmethod
    def _format_content(self, result: "ResearchResult") -> str:
        """
        Format the research result as a string.

        This is the main method subclasses must implement.
        Returns the formatted content ready to write to file.
        """
        pass

    def export(self, result: "ResearchResult", filepath: str | Path) -> Path:
        """
        Export research result to a file.

        LEARNING NOTE - Template Method:
        This method defines the export workflow:
        1. Format the content (subclass-specific)
        2. Write to file (same for all formats)
        3. Return the path

        Subclasses only need to implement _format_content().
        """
        filepath = Path(filepath)

        # Add extension if not present
        if not filepath.suffix:
            filepath = filepath.with_suffix(f".{self.extension}")

        # Format and write
        content = self._format_content(result)
        filepath.write_text(content, encoding="utf-8")

        return filepath

    def export_string(self, result: "ResearchResult") -> str:
        """
        Export research result as a string (no file).

        Useful for displaying in GUI or sending over network.
        """
        return self._format_content(result)
