"""Main entry point for the research bot."""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from .config import Config
from .agents.research_agent import ResearchAgent, ResearchResult


def save_result(result: ResearchResult, output_dir: str) -> Path:
    """Save research result to a file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in result.query[:50])
    filename = f"research_{safe_query}_{timestamp}.json"

    filepath = output_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "query": result.query,
                "summary": result.summary,
                "sources": result.sources,
                "iterations": result.iterations,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return filepath


async def run_research(query: str, config: Config) -> ResearchResult:
    """Run a research task."""
    agent = ResearchAgent(config)
    return await agent.research(query)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Bot - Research any topic using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  research-bot "What are the latest developments in quantum computing?"
  research-bot --output ./my_research "History of the internet"
  research-bot --max-iterations 10 "Climate change solutions"
        """,
    )

    parser.add_argument(
        "query",
        type=str,
        help="The research query or topic to investigate",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="research_output",
        help="Directory to save research results (default: research_output)",
    )

    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=5,
        help="Maximum research iterations (default: 5)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file, only print to console",
    )

    args = parser.parse_args()

    # Build config
    config = Config.from_env()
    config.output_dir = args.output
    config.max_iterations = args.max_iterations
    config.model = args.model

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    print(f"🔍 Researching: {args.query}")
    print(f"   Model: {config.model}")
    print(f"   Max iterations: {config.max_iterations}")
    print("-" * 50)

    # Run research
    result = asyncio.run(run_research(args.query, config))

    # Display results
    print("\n" + "=" * 50)
    print("RESEARCH COMPLETE")
    print("=" * 50)
    print(f"\nIterations: {result.iterations}")
    print(f"Sources found: {len(result.sources)}")

    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  - {source.get('title', 'Untitled')}")
            print(f"    {source.get('url', '')}")

    print("\n" + "-" * 50)
    print("SUMMARY:")
    print("-" * 50)
    print(result.summary)

    # Save results
    if not args.no_save:
        filepath = save_result(result, config.output_dir)
        print(f"\n📄 Results saved to: {filepath}")

    return 0


if __name__ == "__main__":
    exit(main())
