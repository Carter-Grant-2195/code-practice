# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an autonomous research bot that uses Claude to research topics by searching the web and analyzing content. The bot runs in an agentic loop: search → fetch → analyze → decide if more research is needed.

## Commands

### Development Setup
```bash
pip install -e ".[dev]"    # Install with dev dependencies
```

### Running the Bot
```bash
# Set API key first
export ANTHROPIC_API_KEY='your-key'

# Run research
research-bot "your research query"
research-bot --max-iterations 10 "query"
research-bot --no-save "query"  # Don't save to file
```

### Testing
```bash
pytest                      # Run all tests
pytest tests/test_config.py # Run specific test file
pytest -v                   # Verbose output
```

### Linting and Type Checking
```bash
ruff check src/             # Lint
ruff format src/            # Format
mypy src/                   # Type check
```

## Architecture

```
src/research_bot/
├── main.py           # CLI entry point
├── config.py         # Configuration management (env vars)
├── agents/
│   └── research_agent.py  # Main agent with agentic loop
└── tools/
    ├── base.py            # Abstract BaseTool class
    ├── web_search.py      # Web search tool
    └── content_fetcher.py # URL content extraction
```

### Key Components

**ResearchAgent** (`agents/research_agent.py`): The core agent that orchestrates research. Uses Claude's tool_use to call tools in a loop until research is complete (marked by `<research_complete>` tags) or max iterations reached.

**Tools**: Inherit from `BaseTool` and implement:
- `name`, `description`, `parameters` properties
- `async execute(**kwargs)` method
- `to_claude_tool()` converts to Claude's tool format

**Config**: Loaded from environment variables via `Config.from_env()`. Requires `ANTHROPIC_API_KEY`.

## Adding New Tools

1. Create a new file in `src/research_bot/tools/`
2. Inherit from `BaseTool`
3. Implement required properties and `execute` method
4. Add to `tools/__init__.py`
5. Register in `ResearchAgent.__init__` if it should be available by default
