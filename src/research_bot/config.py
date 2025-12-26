"""Configuration management for the research bot.

LEARNING NOTE: This file demonstrates:
1. Dataclasses - automatic creation of __init__, __repr__, etc.
2. Environment variables - reading config from the system environment
3. Validation - checking that configuration is valid and safe
4. Class methods - methods that create instances of a class
"""

import os  # For reading environment variables
from dataclasses import dataclass  # Simplifies creating data-holding classes


@dataclass  # Automatically creates __init__, __repr__, __eq__, etc.
class Config:
    """
    Bot configuration loaded from environment variables.

    LEARNING NOTE - What is a dataclass?
    The @dataclass decorator automatically generates common methods:
    - __init__(): Constructor to initialize the object
    - __repr__(): String representation for debugging
    - __eq__(): Equality comparison
    This saves you from writing boilerplate code!

    LEARNING NOTE - Environment variables:
    These are system-level settings stored outside your code. Benefits:
    - Keep secrets (like API keys) out of source code
    - Different settings for development vs production
    - Easy to change without modifying code

    All settings have sensible defaults but can be overridden via env vars
    prefixed with RESEARCH_BOT_ (except ANTHROPIC_API_KEY).
    """

    # LEARNING NOTE - Type hints with defaults:
    # The format "variable_name: type = default_value" tells Python:
    # 1. What type of data this should be (str, int, bool, etc.)
    # 2. What the default value is if not specified

    # API settings
    anthropic_api_key: str = ""  # Your API key from Anthropic (kept secret!)
    model: str = "claude-sonnet-4-20250514"  # Which Claude model to use
    max_tokens: int = 4096  # Maximum tokens Claude can generate per response

    # Research behavior
    max_search_results: int = 10  # Max search results to return per query
    max_iterations: int = 5  # Max research loop iterations before stopping
    timeout_seconds: int = 30  # How long to wait for web requests

    # Output
    output_dir: str = "research_output"  # Where to save research results
    save_intermediate: bool = True  # Whether to save intermediate steps

    def validate(self) -> None:
        """
        Validate configuration values for security and correctness.

        Raises ValueError if configuration is invalid.
        Security: Validates all inputs to prevent misuse.
        """
        # Check API key exists (but don't log it!)
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Set it with: export ANTHROPIC_API_KEY='your-key'"
            )

        # Validate API key format (basic check without exposing the key)
        if len(self.anthropic_api_key) < 20:
            raise ValueError("ANTHROPIC_API_KEY appears to be invalid (too short)")

        # Validate numeric ranges
        if self.max_tokens < 1 or self.max_tokens > 200000:
            raise ValueError(f"max_tokens must be between 1 and 200000, got {self.max_tokens}")

        if self.max_search_results < 1 or self.max_search_results > 50:
            raise ValueError(f"max_search_results must be between 1 and 50, got {self.max_search_results}")

        if self.max_iterations < 1 or self.max_iterations > 50:
            raise ValueError(f"max_iterations must be between 1 and 50, got {self.max_iterations}")

        if self.timeout_seconds < 1 or self.timeout_seconds > 300:
            raise ValueError(f"timeout_seconds must be between 1 and 300, got {self.timeout_seconds}")

        # Validate model name format
        if not self.model or len(self.model) < 5:
            raise ValueError(f"Invalid model name: {self.model}")

    @classmethod  # Decorator that makes this a class method, not instance method
    def from_env(cls) -> "Config":
        """
        Create config from environment variables with defaults.

        LEARNING NOTE - What is @classmethod?
        A class method receives the class (cls) as first parameter, not an instance (self).
        This lets you create instances in alternative ways:
        - Regular: Config(api_key="...", model="...")
        - From env: Config.from_env()  # Much cleaner!

        LEARNING NOTE - Why return type is "Config" in quotes?
        The quotes are needed because we're inside the Config class definition.
        Python doesn't know what Config is yet, so we use a "forward reference".

        Security: Safely parses integer values with error handling.
        """
        def safe_int(env_var: str, default: int) -> int:
            """
            Safely parse integer from environment variable.

            LEARNING NOTE - Nested function:
            This function is defined inside another function. It can only be
            used within from_env(). Useful for helper functions that aren't
            needed elsewhere.

            LEARNING NOTE - Try/Except blocks:
            This is error handling. If converting to int fails, we catch the
            ValueError and raise a more helpful error message.
            """
            try:
                value = os.getenv(env_var)  # Get environment variable
                return int(value) if value else default  # Convert to int if exists
            except ValueError:
                # If conversion fails, raise an error with clear message
                raise ValueError(
                    f"Invalid integer value for {env_var}: {os.getenv(env_var)}"
                )

        # LEARNING NOTE - Using cls() to create an instance:
        # cls() is like calling Config(), but works even if you subclass Config
        # cls refers to whatever class this method was called on
        return cls(
            # os.getenv("NAME", "default") reads environment variable or uses default
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("RESEARCH_BOT_MODEL", "claude-sonnet-4-20250514"),
            # Use safe_int helper to convert string env vars to integers safely
            max_tokens=safe_int("RESEARCH_BOT_MAX_TOKENS", 4096),
            max_search_results=safe_int("RESEARCH_BOT_MAX_RESULTS", 10),
            max_iterations=safe_int("RESEARCH_BOT_MAX_ITERATIONS", 5),
            timeout_seconds=safe_int("RESEARCH_BOT_TIMEOUT", 30),
            output_dir=os.getenv("RESEARCH_BOT_OUTPUT_DIR", "research_output"),
            # Parse boolean: get env var, convert to lowercase, check if it equals "true"
            save_intermediate=os.getenv("RESEARCH_BOT_SAVE_INTERMEDIATE", "true").lower() == "true",
        )
