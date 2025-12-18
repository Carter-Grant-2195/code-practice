"""Tests for research tools."""

import pytest

from research_bot.tools.base import BaseTool
from research_bot.tools.web_search import WebSearchTool
from research_bot.tools.content_fetcher import ContentFetcherTool


class TestWebSearchTool:
    """Tests for the WebSearchTool."""

    def test_tool_properties(self):
        """Test that tool properties are correct."""
        tool = WebSearchTool()

        assert tool.name == "web_search"
        assert "search" in tool.description.lower()
        assert "query" in tool.parameters["properties"]
        assert "query" in tool.parameters["required"]

    def test_to_claude_tool(self):
        """Test conversion to Claude tool format."""
        tool = WebSearchTool()
        claude_tool = tool.to_claude_tool()

        assert claude_tool["name"] == "web_search"
        assert "description" in claude_tool
        assert "input_schema" in claude_tool


class TestContentFetcherTool:
    """Tests for the ContentFetcherTool."""

    def test_tool_properties(self):
        """Test that tool properties are correct."""
        tool = ContentFetcherTool()

        assert tool.name == "fetch_content"
        assert "fetch" in tool.description.lower()
        assert "url" in tool.parameters["properties"]
        assert "url" in tool.parameters["required"]

    def test_to_claude_tool(self):
        """Test conversion to Claude tool format."""
        tool = ContentFetcherTool()
        claude_tool = tool.to_claude_tool()

        assert claude_tool["name"] == "fetch_content"
        assert "description" in claude_tool
        assert "input_schema" in claude_tool

    def test_custom_timeout(self):
        """Test that custom timeout is stored."""
        tool = ContentFetcherTool(timeout=60)
        assert tool.timeout == 60
