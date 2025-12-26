"""
Data extraction module for parsing structured information from research content.

LEARNING NOTE: This module extracts data WITHOUT additional API calls.
All extraction uses regex patterns and text parsing - keeping costs at $0.

Key patterns extracted:
- Statistics and numbers (percentages, prices, measurements)
- Dates and timelines
- Code snippets
- Technical specifications
- Key-value pairs (e.g., "RAM: 16GB")
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedData:
    """
    Container for all extracted data from research content.

    LEARNING NOTE - Why a dataclass?
    Dataclasses give us a clean way to group related data.
    Each field represents a different type of extracted information.
    """

    # Statistics: numbers with context
    # Example: {"85%": "user satisfaction rate", "1.5M": "downloads"}
    statistics: dict[str, str] = field(default_factory=dict)

    # Dates mentioned in the content
    # Example: ["2024-01-15", "March 2024", "Q2 2025"]
    dates: list[str] = field(default_factory=list)

    # Code snippets found in the content
    # Example: [{"language": "python", "code": "print('hello')"}]
    code_snippets: list[dict[str, str]] = field(default_factory=list)

    # Technical specifications (key-value pairs)
    # Example: {"RAM": "16GB", "CPU": "M3 Pro", "Storage": "512GB SSD"}
    specifications: dict[str, str] = field(default_factory=dict)

    # URLs found in the content
    urls: list[str] = field(default_factory=list)

    # Version numbers found
    # Example: ["v2.0.0", "Python 3.12", "Node 20.x"]
    versions: list[str] = field(default_factory=list)

    # Prices found
    # Example: {"Pro Plan": "$19.99/month", "Enterprise": "$99/user"}
    prices: dict[str, str] = field(default_factory=dict)

    # Companies/products mentioned frequently
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "statistics": self.statistics,
            "dates": self.dates,
            "code_snippets": self.code_snippets,
            "specifications": self.specifications,
            "urls": self.urls,
            "versions": self.versions,
            "prices": self.prices,
            "entities": self.entities,
        }

    def is_empty(self) -> bool:
        """Check if any data was extracted."""
        return not any([
            self.statistics, self.dates, self.code_snippets,
            self.specifications, self.urls, self.versions,
            self.prices, self.entities
        ])


class DataExtractor:
    """
    Extracts structured data from text using pattern matching.

    LEARNING NOTE - No API calls!
    This class uses regex patterns to find and extract data.
    It's fast, free, and runs locally.

    Usage:
        extractor = DataExtractor()
        data = extractor.extract_all(text)
        print(data.statistics)  # {"85%": "satisfaction rate"}
    """

    # LEARNING NOTE - Compiled regex patterns:
    # We compile patterns once at class level for efficiency.
    # re.IGNORECASE makes matching case-insensitive.

    # Match percentages like "85%", "99.9%"
    PERCENT_PATTERN = re.compile(r'(\d+(?:\.\d+)?%)')

    # Match prices like "$19.99", "$1,000", "€50"
    PRICE_PATTERN = re.compile(
        r'([\$€£¥][\d,]+(?:\.\d{2})?(?:/\w+)?)'
        r'|(\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|dollars?))',
        re.IGNORECASE
    )

    # Match dates in various formats
    DATE_PATTERN = re.compile(
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'  # 2024-01-15
        r'|(\d{1,2}[-/]\d{1,2}[-/]\d{4})'  # 01/15/2024
        r'|((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})'  # January 15, 2024
        r'|((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})'  # January 2024
        r'|(Q[1-4]\s+\d{4})'  # Q1 2024
        r'|(\d{4})',  # Just year
        re.IGNORECASE
    )

    # Match version numbers like "v2.0.0", "Python 3.12", "version 4.5.1"
    VERSION_PATTERN = re.compile(
        r'(v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?)'
        r'|(version\s+\d+(?:\.\d+)*)',
        re.IGNORECASE
    )

    # Match URLs
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+'
    )

    # Match code blocks (markdown style)
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w*)\n(.*?)```',
        re.DOTALL
    )

    # Match inline code
    INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')

    # Match specifications like "RAM: 16GB", "Storage: 512GB"
    SPEC_PATTERN = re.compile(
        r'([A-Z][a-zA-Z\s]{2,20}):\s*(\d+(?:\.\d+)?\s*(?:GB|MB|TB|GHz|MHz|W|mAh|MP|fps|ms|nm|x|"|\'|inch(?:es)?)?)',
        re.IGNORECASE
    )

    # Match measurements like "16GB", "3.5GHz", "1TB"
    MEASUREMENT_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?\s*(?:GB|MB|TB|KB|GHz|MHz|GiB|MiB|W|mAh|MP|fps|ms|nm))',
        re.IGNORECASE
    )

    def extract_all(self, text: str) -> ExtractedData:
        """
        Extract all structured data from text.

        This is the main entry point - it runs all extraction methods
        and returns a complete ExtractedData object.
        """
        return ExtractedData(
            statistics=self._extract_statistics(text),
            dates=self._extract_dates(text),
            code_snippets=self._extract_code(text),
            specifications=self._extract_specifications(text),
            urls=self._extract_urls(text),
            versions=self._extract_versions(text),
            prices=self._extract_prices(text),
            entities=self._extract_entities(text),
        )

    def _extract_statistics(self, text: str) -> dict[str, str]:
        """
        Extract statistics (percentages and numbers with context).

        Tries to capture surrounding context for each number.
        """
        stats = {}

        # Find percentages with context
        for match in self.PERCENT_PATTERN.finditer(text):
            percent = match.group(1)
            # Get surrounding context (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            # Clean up context
            context = ' '.join(context.split())  # Normalize whitespace
            if percent not in stats:  # Don't overwrite
                stats[percent] = context

        return stats

    def _extract_dates(self, text: str) -> list[str]:
        """Extract dates in various formats."""
        dates = []
        for match in self.DATE_PATTERN.finditer(text):
            # Get the matched group that isn't None
            date = next((g for g in match.groups() if g), None)
            if date and date not in dates:
                # Filter out standalone years unless they're meaningful (2020+)
                if len(date) == 4 and date.isdigit():
                    year = int(date)
                    if year >= 2020:  # Only recent years
                        dates.append(date)
                else:
                    dates.append(date)

        return dates[:20]  # Limit to 20 dates

    def _extract_code(self, text: str) -> list[dict[str, str]]:
        """Extract code snippets from markdown code blocks."""
        snippets = []

        # Extract fenced code blocks
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            if code:  # Only add non-empty snippets
                snippets.append({
                    "language": language,
                    "code": code[:1000]  # Limit size
                })

        return snippets[:10]  # Limit to 10 snippets

    def _extract_specifications(self, text: str) -> dict[str, str]:
        """Extract technical specifications (key-value pairs)."""
        specs = {}

        for match in self.SPEC_PATTERN.finditer(text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            # Normalize key
            key = key.title()
            if key not in specs:
                specs[key] = value

        return specs

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        urls = []
        for match in self.URL_PATTERN.finditer(text):
            url = match.group(0)
            # Clean up trailing punctuation
            url = url.rstrip('.,;:!?)')
            if url not in urls:
                urls.append(url)

        return urls[:50]  # Limit to 50 URLs

    def _extract_versions(self, text: str) -> list[str]:
        """Extract version numbers."""
        versions = []
        for match in self.VERSION_PATTERN.finditer(text):
            version = next((g for g in match.groups() if g), None)
            if version and version not in versions:
                versions.append(version)

        return versions[:20]  # Limit to 20 versions

    def _extract_prices(self, text: str) -> dict[str, str]:
        """Extract prices with context."""
        prices = {}

        for match in self.PRICE_PATTERN.finditer(text):
            price = next((g for g in match.groups() if g), None)
            if price:
                # Get context (words before the price)
                start = max(0, match.start() - 40)
                context = text[start:match.start()].strip()
                # Get last few words as label
                words = context.split()[-4:]  # Last 4 words
                label = ' '.join(words) if words else "Price"
                label = label.strip(':- ')
                if label and price not in prices.values():
                    prices[label] = price

        return prices

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract frequently mentioned entities (companies, products, tech).

        LEARNING NOTE - Simple approach:
        We look for capitalized multi-word phrases that appear multiple times.
        This is a heuristic, not NER (Named Entity Recognition).
        """
        # Find capitalized words/phrases
        entity_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')

        # Count occurrences
        entity_counts: dict[str, int] = {}
        for match in entity_pattern.finditer(text):
            entity = match.group(1)
            # Filter out common words
            if entity.lower() not in {'the', 'this', 'that', 'these', 'those',
                                       'january', 'february', 'march', 'april',
                                       'may', 'june', 'july', 'august', 'september',
                                       'october', 'november', 'december',
                                       'monday', 'tuesday', 'wednesday', 'thursday',
                                       'friday', 'saturday', 'sunday'}:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1

        # Return entities mentioned 2+ times, sorted by frequency
        entities = [
            entity for entity, count in sorted(
                entity_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if count >= 2
        ]

        return entities[:15]  # Top 15 entities
