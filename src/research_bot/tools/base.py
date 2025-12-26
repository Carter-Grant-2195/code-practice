"""Abstract base class for research tools.

LEARNING NOTE: This file demonstrates several important Python concepts:
1. Abstract Base Classes (ABC) - enforce that child classes implement certain methods
2. Properties - make methods look like attributes using @property decorator
3. Type hints - specify what types functions expect and return
4. Async programming - functions that can run concurrently
"""

# ABC = Abstract Base Class - lets us create "template" classes
# Child classes MUST implement methods marked with @abstractmethod
from abc import ABC, abstractmethod

# 'Any' type hint means "any data type is allowed"
from typing import Any


class BaseTool(ABC):
    """
    Base class for all tools available to the research agent.

    WHAT IS AN ABSTRACT BASE CLASS?
    It's like a blueprint or contract. Any class that inherits from BaseTool
    MUST implement the methods marked as @abstractmethod. This ensures all
    tools have a consistent interface.

    WHY USE THIS?
    - Ensures all tools have the same structure (name, description, execute, etc.)
    - Makes it easy to add new tools without changing the agent code
    - Python will raise an error if you forget to implement required methods

    To create a new tool:
    1. Inherit from BaseTool: class MyTool(BaseTool):
    2. Implement name, description, parameters properties
    3. Implement async execute() method
    4. Register in tools/__init__.py
    """

    @property  # Makes this method accessible like an attribute: tool.name instead of tool.name()
    @abstractmethod  # Child classes MUST implement this
    def name(self) -> str:
        """
        Unique tool identifier used by Claude to call this tool.

        LEARNING NOTE - The @property decorator:
        This makes a method act like a variable. Instead of calling tool.name(),
        you just use tool.name (no parentheses).
        """
        ...  # The '...' is a placeholder meaning "implement this in child class"

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Explains to Claude when and how to use this tool.

        This description helps Claude understand when to use this tool.
        Be clear and specific so Claude knows when it's appropriate.
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        JSON schema defining the tool's input parameters.

        LEARNING NOTE - JSON Schema:
        This is a standard format for describing data structures. It tells Claude
        what parameters the tool expects, their types, and which are required.

        Example:
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"}
            },
            "required": ["url"]
        }
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """
        Run the tool with given parameters and return results.

        LEARNING NOTE - Async/Await:
        The 'async' keyword means this function can pause and resume, allowing
        other code to run while waiting (like for network requests).
        You call it with: result = await tool.execute(...)

        LEARNING NOTE - **kwargs:
        This means "accept any number of keyword arguments as a dictionary".
        If you call execute(url="http://...", timeout=30), kwargs will be:
        {"url": "http://...", "timeout": 30}
        """
        ...

    def to_claude_tool(self) -> dict:
        """
        Convert to Claude API tool format.

        LEARNING NOTE:
        This method takes our BaseTool structure and converts it to the format
        that the Claude API expects. It's like translating our internal format
        to Claude's language.

        Not marked as @abstractmethod because all tools use the same conversion.
        Child classes inherit this method automatically.
        """
        return {
            "name": self.name,  # Access the property we defined above
            "description": self.description,
            "input_schema": self.parameters,  # Claude calls it 'input_schema'
        }
