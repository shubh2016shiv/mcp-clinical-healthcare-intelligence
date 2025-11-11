"""Centralized prompt and instruction management for MCP healthcare tools.

This module provides a centralized system for managing tool prompts, descriptions,
and system instructions. All content is stored in separate JSON files and loaded
dynamically, allowing for easy maintenance and updates without modifying the server
registration code.

Key Benefits:
    - Centralized prompt and instruction management
    - Easy content updates without server code changes
    - Consistent formatting and structure
    - Version control for all prompt changes
    - Separation of concerns between logic and presentation

Features:
    - Tool prompts: Individual JSON files for each tool's description and usage
    - System instructions: Centralized server-level instructions and guidelines
    - Instruction categories: Organized behavioral guidelines and tool categories
    - Caching: Performance optimization with reload capabilities for development

Usage:
    # Tool prompts
    from . import get_tool_prompt
    prompt = get_tool_prompt("search_patients")

    # System instructions
    from . import get_system_instructions, get_instruction_category
    instructions = get_system_instructions()
    categories = get_instruction_category("tool_categories")
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Directory containing prompt files
PROMPTS_DIR = Path(__file__).parent


class ToolPrompt:
    """Structured representation of a tool prompt."""

    def __init__(self, name: str, description: str, usage: str, examples: list[str]):
        self.name = name
        self.description = description
        self.usage = usage
        self.examples = examples

    def to_docstring(self) -> str:
        """Convert the prompt to a properly formatted docstring."""
        examples_text = "\n".join(f"- {example}" for example in self.examples)
        return f"""{self.description}

{self.usage}

Examples:
{examples_text}"""

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ToolPrompt":
        """Create a ToolPrompt from dictionary data."""
        return cls(
            name=name,
            description=data["description"],
            usage=data["usage"],
            examples=data["examples"],
        )


# Cache for loaded prompts
_prompts_cache: dict[str, ToolPrompt] = {}
# Cache for system instructions
_system_instructions_cache: dict[str, Any] | None = None
# Redis prompt cache
_redis_prompt_cache = None


def _initialize_redis_prompt_cache():
    """Initialize Redis prompt cache if available."""
    global _redis_prompt_cache
    if _redis_prompt_cache is not None:
        return

    try:
        from src.mcp_server.cache import get_cache_manager

        cache_manager = get_cache_manager()
        if cache_manager.is_available():
            _redis_prompt_cache = cache_manager.prompt_cache
            logger.debug("Redis prompt cache initialized")
    except Exception as e:
        logger.debug(f"Redis prompt cache not available: {e}")


def get_tool_prompt(tool_name: str) -> str | None:
    """Get the formatted prompt/docstring for a tool.

    Uses Redis for distributed caching across server instances while maintaining
    local cache for performance.

    Args:
        tool_name: Name of the tool (e.g., "search_patients")

    Returns:
        Formatted docstring for the tool, or None if not found
    """
    # Initialize Redis cache if not already done
    _initialize_redis_prompt_cache()

    # Check Redis cache first (before local cache for consistency)
    if _redis_prompt_cache:
        try:
            cached_prompt = _redis_prompt_cache.get_prompt(tool_name)
            if cached_prompt:
                logger.debug(f"Using cached prompt for {tool_name}")
                # Update local cache for next lookup
                if tool_name not in _prompts_cache:
                    _prompts_cache[tool_name] = ToolPrompt(
                        name=tool_name, description="", usage="", examples=[]
                    )
                return cached_prompt
        except Exception as e:
            logger.debug(f"Failed to get prompt from Redis: {e}")

    # Check local cache
    if tool_name in _prompts_cache:
        return _prompts_cache[tool_name].to_docstring()

    # Load prompt from file
    prompt_file = PROMPTS_DIR / f"{tool_name}.json"
    if not prompt_file.exists():
        logger.warning(f"Prompt file not found: {prompt_file}")
        return None

    try:
        with open(prompt_file, encoding="utf-8") as f:
            data = json.load(f)

        prompt = ToolPrompt.from_dict(tool_name, data)
        prompt_docstring = prompt.to_docstring()

        # Cache in local cache
        _prompts_cache[tool_name] = prompt

        # Cache in Redis for multi-instance consistency
        if _redis_prompt_cache:
            try:
                _redis_prompt_cache.set_prompt(tool_name, prompt_docstring)
            except Exception as e:
                logger.debug(f"Failed to cache prompt in Redis: {e}")

        return prompt_docstring

    except Exception as e:
        logger.error(f"Error loading prompt for {tool_name}: {e}")
        return None


def reload_prompts() -> None:
    """Reload all prompts from disk. Useful for development."""
    global _prompts_cache
    _prompts_cache.clear()
    logger.info("Prompts cache cleared")


def list_available_prompts() -> list[str]:
    """List all available tool names with prompts."""
    return [
        f.stem
        for f in PROMPTS_DIR.glob("*.json")
        if f.is_file() and f.name != "__init__.py" and f.name != "system_instructions.json"
    ]


def get_system_instructions() -> str | None:
    """Get the system instructions for the MCP server.

    Returns:
        System instructions string, or None if not found
    """
    global _system_instructions_cache

    if _system_instructions_cache is not None:
        return _system_instructions_cache.get("server_instructions")

    # Load system instructions from file
    instructions_file = PROMPTS_DIR / "system_instructions.json"
    if not instructions_file.exists():
        logger.warning(f"System instructions file not found: {instructions_file}")
        return None

    try:
        with open(instructions_file, encoding="utf-8") as f:
            _system_instructions_cache = json.load(f)

        return _system_instructions_cache.get("server_instructions")

    except Exception as e:
        logger.error(f"Error loading system instructions: {e}")
        return None


def get_instruction_category(category: str) -> dict[str, Any] | None:
    """Get instruction data for a specific category.

    Args:
        category: Category name (e.g., "tool_categories", "behavior_guidelines")

    Returns:
        Dictionary containing category instructions, or None if not found
    """
    global _system_instructions_cache

    if _system_instructions_cache is None:
        get_system_instructions()  # Load cache if not already loaded

    if _system_instructions_cache and category in _system_instructions_cache:
        return _system_instructions_cache[category]

    return None


def reload_system_instructions() -> None:
    """Reload system instructions from disk. Useful for development."""
    global _system_instructions_cache
    _system_instructions_cache = None
    logger.info("System instructions cache cleared")
