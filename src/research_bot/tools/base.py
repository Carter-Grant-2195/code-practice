"""Abstract base class for research tools."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Base class for all tools available to the research agent.

    To create a new tool:
    1. Inherit from BaseTool
    2. Implement name, description, parameters properties
    3. Implement async execute() method
    4. Register in tools/__init__.py
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier used by Claude to call this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Explains to Claude when and how to use this tool."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema defining the tool's input parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Run the tool with given parameters and return results."""
        ...

    def to_claude_tool(self) -> dict:
        """Convert to Claude API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
