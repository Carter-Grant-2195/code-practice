"""Main research agent that orchestrates autonomous research tasks."""

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from ..config import Config
from ..tools.base import BaseTool
from ..tools.web_search import WebSearchTool
from ..tools.content_fetcher import ContentFetcherTool


@dataclass
class ResearchResult:
    """Result of a research task."""

    query: str
    summary: str
    sources: list[dict[str, str]] = field(default_factory=list)
    raw_findings: list[str] = field(default_factory=list)
    iterations: int = 0


class ResearchAgent:
    """
    Autonomous research agent that uses Claude to research topics.

    The agent uses a loop of:
    1. Think about what information is needed
    2. Use tools to gather information
    3. Analyze findings
    4. Decide if more research is needed or if done
    """

    SYSTEM_PROMPT = """You are an autonomous research agent. Your goal is to thoroughly research topics and provide comprehensive, accurate information.

When researching:
1. Break down complex topics into specific questions
2. Use web_search to find relevant sources
3. Use fetch_content to read promising sources in detail
4. Synthesize information from multiple sources
5. Identify gaps in your research and fill them
6. Cite your sources

You have access to these tools:
- web_search: Search the web for information
- fetch_content: Fetch and read the content of a specific URL

When you have gathered enough information to provide a comprehensive answer, respond with your findings in this format:
<research_complete>
Your comprehensive research summary here, with citations.
</research_complete>

If you need more information, continue using tools to gather it."""

    def __init__(self, config: Config | None = None, tools: list[BaseTool] | None = None):
        """
        Initialize the research agent.

        Args:
            config: Configuration object. If None, loads from environment.
            tools: List of tools available to the agent. If None, uses defaults.
        """
        self.config = config or Config.from_env()
        self.config.validate()

        self.client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

        # Set up tools
        if tools is None:
            self.tools = [
                WebSearchTool(max_results=self.config.max_search_results),
                ContentFetcherTool(timeout=self.config.timeout_seconds),
            ]
        else:
            self.tools = tools

        self.tool_map = {tool.name: tool for tool in self.tools}

    async def research(self, query: str) -> ResearchResult:
        """
        Perform autonomous research on a topic.

        Args:
            query: The research query or topic

        Returns:
            ResearchResult with findings and sources
        """
        messages = [
            {
                "role": "user",
                "content": f"Please research the following topic thoroughly: {query}",
            }
        ]

        sources = []
        raw_findings = []
        iterations = 0

        while iterations < self.config.max_iterations:
            iterations += 1

            # Call Claude with tools
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.SYSTEM_PROMPT,
                tools=[tool.to_claude_tool() for tool in self.tools],
                messages=messages,
            )

            # Process response
            assistant_content = []
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})

                    # Check if research is complete
                    if "<research_complete>" in block.text:
                        # Extract the summary
                        start = block.text.find("<research_complete>") + len("<research_complete>")
                        end = block.text.find("</research_complete>")
                        if end > start:
                            summary = block.text[start:end].strip()
                        else:
                            summary = block.text[start:].strip()

                        return ResearchResult(
                            query=query,
                            summary=summary,
                            sources=sources,
                            raw_findings=raw_findings,
                            iterations=iterations,
                        )

                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

                    # Execute the tool
                    tool = self.tool_map.get(block.name)
                    if tool:
                        result = await tool.execute(**block.input)

                        # Track sources and findings
                        if block.name == "fetch_content" and "content" in result:
                            sources.append({
                                "url": block.input.get("url", ""),
                                "title": result.get("title", ""),
                            })
                            raw_findings.append(result.get("content", "")[:2000])

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: Unknown tool '{block.name}'",
                            "is_error": True,
                        })

            # Add assistant message
            messages.append({"role": "assistant", "content": assistant_content})

            # Add tool results if any
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Check stop reason
            if response.stop_reason == "end_turn" and not tool_results:
                # Agent finished without marking complete
                last_text = ""
                for block in response.content:
                    if block.type == "text":
                        last_text = block.text
                        break

                return ResearchResult(
                    query=query,
                    summary=last_text,
                    sources=sources,
                    raw_findings=raw_findings,
                    iterations=iterations,
                )

        # Max iterations reached
        return ResearchResult(
            query=query,
            summary="Research incomplete - maximum iterations reached.",
            sources=sources,
            raw_findings=raw_findings,
            iterations=iterations,
        )

    def research_sync(self, query: str) -> ResearchResult:
        """Synchronous wrapper for research."""
        import asyncio
        return asyncio.run(self.research(query))
