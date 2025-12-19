"""Configuration management for the research bot."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    Bot configuration loaded from environment variables.

    All settings have sensible defaults but can be overridden via env vars
    prefixed with RESEARCH_BOT_ (except ANTHROPIC_API_KEY).
    """

    # API settings
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096

    # Research behavior
    max_search_results: int = 10
    max_iterations: int = 5
    timeout_seconds: int = 30

    # Output
    output_dir: str = "research_output"
    save_intermediate: bool = True

    def validate(self) -> None:
        """Raise ValueError if required config is missing."""
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Set it with: export ANTHROPIC_API_KEY='your-key'"
            )

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables with defaults."""
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("RESEARCH_BOT_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=int(os.getenv("RESEARCH_BOT_MAX_TOKENS", "4096")),
            max_search_results=int(os.getenv("RESEARCH_BOT_MAX_RESULTS", "10")),
            max_iterations=int(os.getenv("RESEARCH_BOT_MAX_ITERATIONS", "5")),
            timeout_seconds=int(os.getenv("RESEARCH_BOT_TIMEOUT", "30")),
            output_dir=os.getenv("RESEARCH_BOT_OUTPUT_DIR", "research_output"),
            save_intermediate=os.getenv("RESEARCH_BOT_SAVE_INTERMEDIATE", "true").lower() == "true",
        )
