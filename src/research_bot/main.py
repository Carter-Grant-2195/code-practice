"""CLI entry point for the research bot."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import Config
from .agents.research_agent import ResearchAgent, ResearchResult
from .exporters import get_exporter


def save_result(result: ResearchResult, output_dir: str, format: str = "json") -> Path:
    """
    Save research results to file in the specified format.

    Security: Uses Path.resolve() to prevent path traversal attacks.
    Validates that the output file is within the intended directory.

    Args:
        result: The research result to save
        output_dir: Directory to save the file
        format: Export format (json, markdown, html, csv)
    """
    # Resolve to absolute path to prevent directory traversal
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Create filename from query and timestamp
    # Security: Strip all non-alphanumeric chars to prevent injection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c.isspace() else "_" for c in result.query[:50])
    safe_query = safe_query.strip().replace(" ", "_")

    # Ensure filename is not empty
    if not safe_query:
        safe_query = "research"

    # Get the exporter and its extension
    exporter = get_exporter(format)
    extension = exporter.extension

    filename = f"research_{safe_query}_{timestamp}.{extension}"
    filepath = (output_path / filename).resolve()

    # Security: Verify the resolved path is still within output directory
    # This prevents path traversal attacks like "../../../etc/passwd"
    if not str(filepath).startswith(str(output_path)):
        raise ValueError(f"Security: Output path '{filepath}' is outside output directory")

    # Export using the appropriate exporter
    exporter.export(result, filepath)

    return filepath


def print_result(result: ResearchResult) -> None:
    """Display research results to console."""
    print("\n" + "=" * 60)
    print("RESEARCH COMPLETE")
    print("=" * 60)
    print(f"Iterations: {result.iterations}")
    print(f"Sources: {len(result.sources)}")
    print(f"Key Findings: {len(result.key_findings)}")
    print(f"Cost: ${result.usage.cost_usd:.4f}")

    if result.sources:
        print("\nSources:")
        for src in result.sources:
            print(f"  - {src.get('title', 'Untitled')}")
            print(f"    {src.get('url', '')}")

    # Display key findings if any
    if result.key_findings:
        print("\n" + "-" * 60)
        print("KEY FINDINGS:")
        print("-" * 60)
        for i, finding in enumerate(result.key_findings, 1):
            print(f"  {i}. {finding}")

    print("\n" + "-" * 60)
    print("SUMMARY:")
    print("-" * 60)
    print(result.summary)

    # Show extracted data summary
    data = result.extracted_data
    if not data.is_empty():
        print("\n" + "-" * 60)
        print("EXTRACTED DATA:")
        print("-" * 60)
        if data.specifications:
            print(f"  Specifications: {len(data.specifications)} found")
        if data.statistics:
            print(f"  Statistics: {len(data.statistics)} found")
        if data.prices:
            print(f"  Prices: {len(data.prices)} found")
        if data.versions:
            print(f"  Versions: {', '.join(data.versions[:5])}")
        if data.code_snippets:
            print(f"  Code snippets: {len(data.code_snippets)} found")
        print("  (See exported file for full details)")


def main() -> int:
    """Run the research bot CLI."""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Bot - Research any topic using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  research-bot "What are the latest developments in quantum computing?"
  research-bot --max-iterations 10 "Climate change solutions"
  research-bot --format markdown "Compare React vs Vue"
  research-bot --format html --output reports "AI trends 2024"
  research-bot --no-save "Quick question about Python"

Export Formats:
  json     - Machine-readable, good for automation/APIs
  markdown - Great for GitHub, documentation, note-taking
  html     - Shareable reports with styling
  csv      - Spreadsheet-compatible tabular data
        """,
    )
    parser.add_argument("query", help="The research query or topic")
    parser.add_argument("-o", "--output", default="research_output", help="Output directory")
    parser.add_argument("-m", "--max-iterations", type=int, default=5, help="Max research iterations")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "md", "html", "csv"],
        default="json",
        help="Export format (default: json)"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    # Build config from env + CLI args
    config = Config.from_env()
    config.output_dir = args.output
    config.max_iterations = args.max_iterations
    config.model = args.model

    try:
        config.validate()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Normalize format
    export_format = "markdown" if args.format == "md" else args.format

    # Show what we're doing
    print(f"Researching: {args.query}")
    print(f"Model: {config.model} | Max iterations: {config.max_iterations} | Format: {export_format}")
    print("-" * 60)

    # Run research
    agent = ResearchAgent(config)
    result = asyncio.run(agent.research(args.query))

    # Display results
    print_result(result)

    # Save if requested
    if not args.no_save:
        filepath = save_result(result, config.output_dir, export_format)
        print(f"\nExported to: {filepath}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
