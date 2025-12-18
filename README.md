# Research Bot

An autonomous research agent powered by Claude that can research any topic by searching the web and analyzing content.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key'
   ```

3. Run a research task:
   ```bash
   research-bot "What are the latest developments in quantum computing?"
   ```

## Features

- Autonomous research loop with configurable iterations
- Web search integration
- Content extraction from web pages
- Source tracking and citation
- JSON output for research results

## Configuration

Environment variables:
- `ANTHROPIC_API_KEY` - Required. Your Anthropic API key
- `RESEARCH_BOT_MODEL` - Claude model to use (default: claude-sonnet-4-20250514)
- `RESEARCH_BOT_MAX_ITERATIONS` - Max research iterations (default: 5)
