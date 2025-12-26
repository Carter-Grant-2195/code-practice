# Python Concepts Learning Guide

This guide explains all the Python concepts used in the research bot, organized from basic to advanced.

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Intermediate Concepts](#intermediate-concepts)
3. [Advanced Concepts](#advanced-concepts)
4. [Architecture Patterns](#architecture-patterns)

---

## Basic Concepts

### Type Hints
Type hints tell Python (and developers) what type of data a variable should contain.

```python
# Basic type hints
name: str = "Alice"          # String
age: int = 25                # Integer
is_student: bool = True      # Boolean

# Function with type hints
def greet(name: str) -> str:  # Takes str, returns str
    return f"Hello, {name}!"
```

**Where we use it:** Throughout the codebase, especially in function definitions.

**Why it's useful:**
- Makes code self-documenting
- Catches errors early
- Enables better IDE autocomplete

---

### Environment Variables
Environment variables store configuration outside your code.

```python
import os

# Read environment variable, use default if not set
api_key = os.getenv("API_KEY", "default_value")
```

**Where we use it:** `config.py` - loading API keys and settings.

**Why it's useful:**
- Keep secrets out of source code
- Different settings for dev/production
- Easy to change without modifying code

---

### Dictionaries
Key-value pairs for storing related data.

```python
# Creating a dictionary
person = {
    "name": "Alice",
    "age": 25,
    "city": "NYC"
}

# Accessing values
print(person["name"])        # "Alice"
print(person.get("age", 0))  # 25, or 0 if not found

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}
```

**Where we use it:**
- `research_agent.py` - tool_map for quick tool lookup
- Everywhere - for structured data

**Why it's useful:** Fast lookup, organized data structure

---

## Intermediate Concepts

### Dataclasses
Automatically generate common methods for data-holding classes.

```python
from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int
    hobbies: list = field(default_factory=list)  # New list for each instance

# Automatically creates:
# - __init__(name, age, hobbies)
# - __repr__() for printing
# - __eq__() for comparison
```

**Where we use it:**
- `config.py` - Config class
- `research_agent.py` - TokenUsage, ResearchResult

**Why use field(default_factory=list)?**
```python
# WRONG - All instances share the SAME list!
class Wrong:
    items: list = []

# RIGHT - Each instance gets its OWN list
class Right:
    items: list = field(default_factory=list)
```

---

### Properties
Make methods look like attributes.

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):  # Access as circle.area, not circle.area()
        return 3.14 * self.radius ** 2

circle = Circle(5)
print(circle.area)  # 78.5 - looks like an attribute!
```

**Where we use it:**
- `base.py` - tool name, description, parameters
- `research_agent.py` - TokenUsage cost calculation

**Why it's useful:** Cleaner syntax, can compute values on-demand

---

### Class Methods
Methods that work with the class, not instances.

```python
class Config:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @classmethod
    def from_env(cls):  # cls = the class itself
        key = os.getenv("API_KEY")
        return cls(api_key=key)  # Create instance

# Two ways to create Config:
config1 = Config(api_key="abc123")        # Regular way
config2 = Config.from_env()               # Using class method
```

**Where we use it:** `config.py` - Config.from_env()

**Why it's useful:** Alternative constructors, factory patterns

---

### Exception Handling
Handle errors gracefully without crashing.

```python
try:
    value = int(input("Enter number: "))
    result = 10 / value
except ValueError:
    print("That's not a number!")
except ZeroDivisionError:
    print("Can't divide by zero!")
except Exception as e:
    print(f"Unexpected error: {e}")
```

**Where we use it:**
- `config.py` - parsing environment variables
- `tools/*.py` - handling network errors

**Why it's useful:** Graceful error handling, better user experience

---

## Advanced Concepts

### Abstract Base Classes (ABC)
Create "blueprints" that force child classes to implement certain methods.

```python
from abc import ABC, abstractmethod

class Animal(ABC):  # Can't instantiate Animal directly
    @abstractmethod
    def make_sound(self):  # Child classes MUST implement this
        pass

class Dog(Animal):
    def make_sound(self):  # Must implement or Python raises error
        return "Woof!"

# dog = Dog()  # Works
# animal = Animal()  # ERROR - can't instantiate abstract class
```

**Where we use it:** `base.py` - BaseTool class

**Why it's useful:**
- Ensures consistent interface across child classes
- Catches missing implementations at runtime
- Documents required methods

---

### Async/Await
Run code concurrently without blocking.

```python
import asyncio
import httpx

async def fetch_url(url):  # async function
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # Wait without blocking
        return response.text

# Run multiple fetches concurrently
async def fetch_multiple():
    urls = ["http://example.com", "http://example.org"]
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)  # Run in parallel
    return results

# Execute async function
asyncio.run(fetch_multiple())
```

**Where we use it:**
- `tools/*.py` - web requests
- `research_agent.py` - research method

**Why it's useful:**
- Don't wait idle during network requests
- Handle multiple operations efficiently
- Better performance for I/O-bound tasks

**Normal vs Async:**
```python
# Normal (blocking): Total time = 3 seconds
def normal():
    time.sleep(1)  # Wait 1 second
    time.sleep(1)  # Wait 1 second
    time.sleep(1)  # Wait 1 second

# Async (concurrent): Total time = 1 second
async def async_version():
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1)
    )  # All sleep at the same time!
```

---

### Context Managers
Automatically set up and tear down resources.

```python
# The 'with' statement
async with httpx.AsyncClient() as client:
    response = await client.get(url)
    # Client automatically closed when done

# Equivalent to:
client = httpx.AsyncClient()
try:
    response = await client.get(url)
finally:
    await client.aclose()  # Always close, even if error
```

**Where we use it:** All HTTP requests in `tools/*.py`

**Why it's useful:**
- Ensures cleanup (closing files, network connections)
- Prevents resource leaks
- Cleaner, more readable code

---

### List Comprehensions
Create lists in a single line.

```python
# Traditional way
squares = []
for x in range(10):
    squares.append(x**2)

# List comprehension - same result
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]

# Dictionary comprehension
tools_map = {tool.name: tool for tool in tools}
```

**Where we use it:**
- `research_agent.py` - creating tool_map
- Throughout - whenever we transform lists

**Why it's useful:** More concise, often faster

---

### **kwargs
Accept arbitrary keyword arguments.

```python
def flexible_function(**kwargs):
    # kwargs is a dictionary of all keyword arguments
    for key, value in kwargs.items():
        print(f"{key} = {value}")

flexible_function(name="Alice", age=25, city="NYC")
# Output:
# name = Alice
# age = 25
# city = NYC
```

**Where we use it:** `base.py` - execute(**kwargs)

**Why it's useful:** Flexible function signatures

---

## Architecture Patterns

### The Agentic Loop
The core pattern of our research bot.

```
1. Start with a goal (query)
2. Agent decides: "What should I do next?"
3. Agent calls a tool (search, fetch)
4. Agent receives results
5. Agent analyzes: "Do I have enough information?"
   - If YES: Return summary
   - If NO: Go to step 2
6. Repeat until done or max iterations
```

**Code flow:**
```python
messages = [initial_query]
for iteration in range(max_iterations):
    # 1. Send messages to Claude
    response = client.messages.create(messages=messages)

    # 2. Claude decides to use tools or finish
    for block in response.content:
        if block.type == "tool_use":
            # 3. Execute tool
            result = await tool.execute(**block.input)
            # 4. Add result to messages
            messages.append(tool_result)

    # 5. Check if Claude signaled completion
    if "<research_complete>" in response:
        return summary

    # 6. Loop continues with updated messages
```

**Why this pattern:**
- AI has agency - it decides what to do
- Stateful - maintains conversation history
- Flexible - can adapt strategy based on results

---

### Tool Use Pattern
How Claude interacts with external tools.

```python
# 1. Define tool interface
class BaseTool(ABC):
    @abstractmethod
    async def execute(self, **kwargs): pass

    def to_claude_tool(self):
        return {"name": self.name, "description": ...}

# 2. Register tools with Claude
tools = [WebSearchTool(), ContentFetcherTool()]
response = client.messages.create(
    tools=[t.to_claude_tool() for t in tools]
)

# 3. Claude decides to use a tool
# Response: {"type": "tool_use", "name": "web_search", "input": {...}}

# 4. Execute the tool
result = await tool_map[name].execute(**input)

# 5. Return result to Claude
messages.append({"type": "tool_result", "content": result})
```

**Benefits:**
- Separation of concerns
- Easy to add new tools
- Claude chooses when and how to use tools

---

### Configuration Pattern
Centralized settings management.

```python
# 1. Define config with defaults
@dataclass
class Config:
    api_key: str = ""
    max_tokens: int = 4096

# 2. Load from environment
@classmethod
def from_env(cls):
    return cls(
        api_key=os.getenv("API_KEY"),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096"))
    )

# 3. Validate
def validate(self):
    if not self.api_key:
        raise ValueError("API key required")

# 4. Use throughout app
config = Config.from_env()
config.validate()
agent = ResearchAgent(config)
```

**Benefits:**
- Single source of truth
- Easy to test (pass different configs)
- Clear documentation of all settings

---

## Common Python Idioms

### The "or" trick for defaults
```python
value = user_input or "default"
# If user_input is truthy, use it
# If user_input is falsy (None, "", 0, []), use "default"
```

### Ternary operator
```python
# Instead of:
if condition:
    x = "yes"
else:
    x = "no"

# Write:
x = "yes" if condition else "no"
```

### F-strings
```python
name = "Alice"
age = 25

# Old way
print("Hello, " + name + "! You are " + str(age))

# New way (Python 3.6+)
print(f"Hello, {name}! You are {age}")

# With expressions
print(f"Next year you'll be {age + 1}")
```

### Unpacking
```python
# Lists
first, *rest, last = [1, 2, 3, 4, 5]
# first = 1, rest = [2, 3, 4], last = 5

# Dictionaries
person = {"name": "Alice", "age": 25}
print(f"{person['name']}")  # Alice

# Better
name = person.get("name", "Unknown")
```

---

## Learning Path

If you're new to these concepts, study them in this order:

1. **Start here:** Type hints, dictionaries, environment variables
2. **Then:** Dataclasses, properties, exception handling
3. **Next:** Class methods, abstract base classes
4. **Advanced:** Async/await, context managers
5. **Patterns:** Agentic loop, tool use pattern

Each concept builds on the previous ones!

---

## Further Learning

- **Official Python Tutorial:** https://docs.python.org/3/tutorial/
- **Type Hints:** https://docs.python.org/3/library/typing.html
- **Async/Await:** https://docs.python.org/3/library/asyncio.html
- **Dataclasses:** https://docs.python.org/3/library/dataclasses.html
- **Claude API:** https://docs.anthropic.com/

---

## Practice Exercises

1. **Create a new tool:** Inherit from BaseTool and implement a calculator
2. **Modify the agent:** Add a token budget limit
3. **Extend config:** Add a new setting with validation
4. **Error handling:** Add better error messages to tools

Good luck with your learning journey! 🚀
