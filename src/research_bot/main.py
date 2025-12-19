"""CLI entry point for the research bot."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import Config
from .agents.research_agent import ResearchAgent, ResearchResult


def save_result(result: ResearchResult, output_dir: str) -> Path:
    """Save research results to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create filename from query and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in result.query[:50])
    filepath = output_path / f"research_{safe_query}_{timestamp}.json"

    filepath.write_text(
        json.dumps(
            {
                "query": result.query,
                "summary": result.summary,
                "sources": result.sources,
                "iterations": result.iterations,
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return filepath


def print_result(result: ResearchResult) -> None:
    """Display research results to console."""
    print("\n" + "=" * 60)
    print("RESEARCH COMPLETE")
    print("=" * 60)
    print(f"Iterations: {result.iterations}")
    print(f"Sources: {len(result.sources)}")

    if result.sources:
        print("\nSources:")
        for src in result.sources:
            print(f"  - {src.get('title', 'Untitled')}")
            print(f"    {src.get('url', '')}")

    print("\n" + "-" * 60)
    print("SUMMARY:")
    print("-" * 60)
    print(result.summary)


def main() -> int:
    """Run the research bot CLI."""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Bot - Research any topic using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  research-bot "What are the latest developments in quantum computing?"
  research-bot --max-iterations 10 "Climate change solutions"
  research-bot --no-save "Quick question about Python"
        """,
    )
    parser.add_argument("query", help="The research query or topic")
    parser.add_argument("-o", "--output", default="research_output", help="Output directory")
    parser.add_argument("-m", "--max-iterations", type=int, default=5, help="Max research iterations")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
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

    # Show what we're doing
    print(f"Researching: {args.query}")
    print(f"Model: {config.model} | Max iterations: {config.max_iterations}")
    print("-" * 60)

    # Run research
    agent = ResearchAgent(config)
    result = asyncio.run(agent.research(args.query))

    # Display results
    print_result(result)

    # Save if requested
    if not args.no_save:
        filepath = save_result(result, config.output_dir)
        print(f"\nSaved to: {filepath}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
