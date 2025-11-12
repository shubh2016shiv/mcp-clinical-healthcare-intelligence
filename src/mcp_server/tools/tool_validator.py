"""Tool validation utilities for ensuring MCP tool reliability.

This module provides utilities to validate tool registration, return types,
and ensure consistent behavior across all MCP tools.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any, get_type_hints

from .models import ErrorResponse

logger = logging.getLogger(__name__)


class ToolValidator:
    """Validates tool registration and execution reliability."""

    @staticmethod
    def validate_tool_signature(tool_func: Callable) -> tuple[bool, str]:
        """Validate that a tool function has proper signature.

        Args:
            tool_func: The tool function to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if function is async
            if not inspect.iscoroutinefunction(tool_func):
                return False, f"Tool {tool_func.__name__} must be async (use async def)"

            # Check if function has type hints
            try:
                hints = get_type_hints(tool_func)
            except Exception as e:
                return False, f"Tool {tool_func.__name__} has invalid type hints: {e}"

            # Check return type hint
            if "return" not in hints:
                return False, f"Tool {tool_func.__name__} missing return type hint"

            return True, "Tool signature is valid"

        except Exception as e:
            return False, f"Error validating tool signature: {e}"

    @staticmethod
    def validate_tool_return_value(
        tool_name: str, return_value: Any, expected_type: type = None
    ) -> tuple[bool, str]:
        """Validate that a tool's return value is of the expected type.

        Args:
            tool_name: Name of the tool
            return_value: The value returned by the tool
            expected_type: Expected type (if None, accepts dict or ErrorResponse)

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for ErrorResponse (always valid)
            if isinstance(return_value, ErrorResponse):
                return True, "ErrorResponse is valid return type"

            # If no expected type, check for dict
            if expected_type is None:
                if isinstance(return_value, dict):
                    return True, "Dict return value is valid"
                elif isinstance(return_value, list):
                    return True, "List return value is valid"
                else:
                    return (
                        False,
                        f"Tool {tool_name} returned unexpected type: {type(return_value).__name__}",
                    )

            # Check against expected type
            if not isinstance(return_value, expected_type):
                return (
                    False,
                    f"Tool {tool_name} returned {type(return_value).__name__}, expected {expected_type.__name__}",
                )

            return True, "Return value matches expected type"

        except Exception as e:
            return False, f"Error validating return value: {e}"

    @staticmethod
    def validate_tool_registration(tools: dict[str, Callable]) -> dict[str, list[str]]:
        """Validate all registered tools.

        Args:
            tools: Dictionary of tool name to tool function

        Returns:
            Dictionary of tool_name to list of validation errors
        """
        validation_errors = {}

        for tool_name, tool_func in tools.items():
            errors = []

            # Validate signature
            is_valid, message = ToolValidator.validate_tool_signature(tool_func)
            if not is_valid:
                errors.append(message)

            # Check docstring
            if not tool_func.__doc__:
                errors.append(f"Tool {tool_name} missing docstring")

            # Store errors if any
            if errors:
                validation_errors[tool_name] = errors
                logger.warning(f"Tool {tool_name} has validation errors: {errors}")

        return validation_errors

    @staticmethod
    def create_error_response_from_exception(operation: str, exception: Exception) -> ErrorResponse:
        """Create a standardized ErrorResponse from an exception.

        Args:
            operation: The operation that failed
            exception: The exception that was raised

        Returns:
            ErrorResponse object
        """
        return ErrorResponse(
            error=f"{type(exception).__name__} in {operation}",
            details=str(exception),
            operation=operation,
        )

    @staticmethod
    async def safe_tool_execution(tool_func: Callable, *args, **kwargs) -> Any:
        """Execute a tool with comprehensive error handling.

        This wrapper ensures consistent error handling and return type validation.

        Args:
            tool_func: The tool function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool result or ErrorResponse
        """
        tool_name = tool_func.__name__

        try:
            # Execute the tool
            result = await tool_func(*args, **kwargs)

            # Validate return value
            is_valid, message = ToolValidator.validate_tool_return_value(tool_name, result)
            if not is_valid:
                logger.error(f"Tool return value validation failed: {message}")
                return ErrorResponse(
                    error="Invalid tool return value",
                    details=message,
                    operation=tool_name,
                )

            return result

        except Exception as e:
            logger.error(
                f"Error executing tool {tool_name}: {type(e).__name__}: {e}", exc_info=True
            )
            return ToolValidator.create_error_response_from_exception(tool_name, e)


def validate_tool_decorator(func: Callable) -> Callable:
    """Decorator to add automatic validation to tool functions.

    This decorator wraps tool functions with validation logic to ensure
    consistent error handling and return type validation.

    Usage:
        @validate_tool_decorator
        async def my_tool(...) -> dict:
            # Tool implementation
            return result
    """

    async def wrapper(*args, **kwargs):
        return await ToolValidator.safe_tool_execution(func, *args, **kwargs)

    # Preserve function metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__

    return wrapper
