"""
Research agent that autonomously gathers and synthesizes information.

The agent operates in a loop: search → fetch → analyze → decide if done.
Claude controls the research process using available tools.
"""

import json
from dataclasses import dataclass, field

import anthropic

from ..config import Config
from ..tools.base import BaseTool
from ..tools.web_search import WebSearchTool
from ..tools.content_fetcher import ContentFetcherTool


# Pricing per million tokens (as of 2024)
MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
    "default": {"input": 3.00, "output": 15.00},
}


@dataclass
class TokenUsage:
    """Tracks token usage and calculates cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["default"])
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


@dataclass
class ResearchResult:
    """Output from a completed research task."""

    query: str
    summary: str
    sources: list[dict[str, str]] = field(default_factory=list)
    raw_findings: list[str] = field(default_factory=list)
    iterations: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    completed: bool = True  # False if hit max iterations


COMPLETION_TAG = "<research_complete>"
COMPLETION_END_TAG = "</research_complete>"

# More efficient system prompt - encourages faster completion
SYSTEM_PROMPT = """You are a fast, efficient research agent. Your goal is to quickly find accurate information and provide a comprehensive answer.

IMPORTANT: Be efficient with API calls. Don't over-research.
- 1-2 searches are usually enough for simple topics
- Only fetch content from the most relevant 2-3 sources
- Stop as soon as you have enough information to answer well

Research strategy:
1. Start with one focused web search
2. Fetch content from the 1-3 most relevant results
3. If you have enough info, provide your answer immediately
4. Only do additional searches if the first results were insufficient

Available tools:
- web_search: Search the web (use focused, specific queries)
- fetch_content: Read a URL's content (only fetch what you need)

When ready (aim for 2-4 iterations), wrap your answer in:
<research_complete>
Your answer here with citations.
</research_complete>

Be concise but thorough. Cite sources with URLs."""

# Prompt used when forcing a summary after max iterations
SUMMARIZE_PROMPT = """Based on the research conducted so far, provide a comprehensive summary of what was found.

Even if the research feels incomplete, synthesize the available information into a useful answer. Include:
1. Key findings from the sources reviewed
2. Direct answers to the original query where possible
3. Citations to sources used
4. Any important caveats about what wasn't fully researched

Wrap your response in:
<research_complete>
Your summary here
</research_complete>"""


class ResearchAgent:
    """
    Autonomous research agent powered by Claude.

    Runs an agentic loop where Claude decides which tools to use,
    what to search for, and when research is complete.
    """

    def __init__(self, config: Config | None = None, tools: list[BaseTool] | None = None):
        self.config = config or Config.from_env()
        self.config.validate()

        self.client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

        self.tools = tools or [
            WebSearchTool(max_results=self.config.max_search_results),
            ContentFetcherTool(timeout=self.config.timeout_seconds),
        ]
        self.tool_map = {tool.name: tool for tool in self.tools}

    async def research(self, query: str) -> ResearchResult:
        """
        Research a topic autonomously.

        Always returns a useful result, even if max iterations reached.
        """
        messages = [{"role": "user", "content": f"Research this topic efficiently: {query}"}]
        sources: list[dict[str, str]] = []
        raw_findings: list[str] = []
        usage = TokenUsage(model=self.config.model)

        for iteration in range(1, self.config.max_iterations + 1):
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[t.to_claude_tool() for t in self.tools],
                messages=messages,
            )

            usage.add(response.usage.input_tokens, response.usage.output_tokens)

            assistant_content = []
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})

                    # Check for completion
                    if COMPLETION_TAG in block.text:
                        summary = self._extract_summary(block.text)
                        return ResearchResult(
                            query=query,
                            summary=summary,
                            sources=sources,
                            raw_findings=raw_findings,
                            iterations=iteration,
                            usage=usage,
                            completed=True,
                        )

                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

                    result = await self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

                    if block.name == "fetch_content" and "content" in result:
                        sources.append({
                            "url": block.input.get("url", ""),
                            "title": result.get("title", ""),
                        })
                        raw_findings.append(result.get("content", "")[:2000])

            messages.append({"role": "assistant", "content": assistant_content})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # If Claude finished without tools or completion tag
            if response.stop_reason == "end_turn" and not tool_results:
                last_text = self._get_last_text(response.content)
                return ResearchResult(
                    query=query,
                    summary=last_text,
                    sources=sources,
                    raw_findings=raw_findings,
                    iterations=iteration,
                    usage=usage,
                    completed=True,
                )

        # Max iterations reached - force a summary of what we found
        return await self._force_summary(
            query, messages, sources, raw_findings, usage
        )

    async def _force_summary(
        self,
        query: str,
        messages: list,
        sources: list[dict[str, str]],
        raw_findings: list[str],
        usage: TokenUsage,
    ) -> ResearchResult:
        """Force Claude to summarize findings when max iterations reached."""

        # Add a message asking for summary
        messages.append({
            "role": "user",
            "content": SUMMARIZE_PROMPT,
        })

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        usage.add(response.usage.input_tokens, response.usage.output_tokens)

        # Extract the summary
        summary = ""
        for block in response.content:
            if block.type == "text":
                if COMPLETION_TAG in block.text:
                    summary = self._extract_summary(block.text)
                else:
                    summary = block.text
                break

        # If still no summary, create one from raw findings
        if not summary and raw_findings:
            summary = self._create_fallback_summary(query, sources, raw_findings)
        elif not summary:
            summary = f"Research on '{query}' was attempted but no results were gathered. Try a more specific query or increase max iterations."

        return ResearchResult(
            query=query,
            summary=summary,
            sources=sources,
            raw_findings=raw_findings,
            iterations=self.config.max_iterations,
            usage=usage,
            completed=False,  # Mark as incomplete so user knows
        )

    def _create_fallback_summary(
        self,
        query: str,
        sources: list[dict[str, str]],
        raw_findings: list[str],
    ) -> str:
        """Create a basic summary from raw findings if Claude fails to summarize."""
        summary = f"Research Summary: {query}\n\n"
        summary += "Note: Maximum iterations reached. Here's what was found:\n\n"

        for i, (source, finding) in enumerate(zip(sources, raw_findings), 1):
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            # Truncate finding for readability
            excerpt = finding[:500] + "..." if len(finding) > 500 else finding
            summary += f"From {title}:\n{excerpt}\n\nSource: {url}\n\n"

        return summary

    async def _execute_tool(self, name: str, inputs: dict) -> dict:
        tool = self.tool_map.get(name)
        if tool:
            return await tool.execute(**inputs)
        return {"error": f"Unknown tool: {name}"}

    def _extract_summary(self, text: str) -> str:
        start = text.find(COMPLETION_TAG) + len(COMPLETION_TAG)
        end = text.find(COMPLETION_END_TAG)
        if end > start:
            return text[start:end].strip()
        return text[start:].strip()

    def _get_last_text(self, content: list) -> str:
        for block in reversed(content):
            if hasattr(block, "type") and block.type == "text":
                return block.text
        return ""

    def research_sync(self, query: str) -> ResearchResult:
        import asyncio
        return asyncio.run(self.research(query))
