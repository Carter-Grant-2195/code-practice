"""Tests for configuration module."""

import os
import pytest
from unittest.mock import patch

from research_bot.config import Config


class TestConfig:
    """Tests for the Config class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            config = Config(anthropic_api_key="test-key")

            assert config.model == "claude-sonnet-4-20250514"
            assert config.max_tokens == 4096
            assert config.max_search_results == 10
            assert config.max_iterations == 5

    def test_validate_missing_api_key(self):
        """Test that validation fails without API key."""
        config = Config(anthropic_api_key="")

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            config.validate()

    def test_validate_with_api_key(self):
        """Test that validation passes with API key."""
        config = Config(anthropic_api_key="test-key")
        config.validate()  # Should not raise

    def test_from_env(self):
        """Test loading config from environment."""
        env_vars = {
            "ANTHROPIC_API_KEY": "env-test-key",
            "RESEARCH_BOT_MODEL": "claude-3-opus-20240229",
            "RESEARCH_BOT_MAX_TOKENS": "8192",
            "RESEARCH_BOT_MAX_RESULTS": "20",
            "RESEARCH_BOT_MAX_ITERATIONS": "10",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config.from_env()

            assert config.anthropic_api_key == "env-test-key"
            assert config.model == "claude-3-opus-20240229"
            assert config.max_tokens == 8192
            assert config.max_search_results == 20
            assert config.max_iterations == 10
