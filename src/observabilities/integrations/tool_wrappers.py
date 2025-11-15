"""Tool wrapper utilities for automatic tool tracing.

This module provides utilities for wrapping tools to automatically
capture execution metrics without modifying tool code.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from ..agent_execution.execution_tracer import AgentExecutionTracer, get_tracer

logger = logging.getLogger(__name__)


class TracedToolWrapper:
    """Wrapper for tools to capture execution metrics automatically.

    Wraps a tool function to automatically log:
    - Tool invocation with arguments
    - Execution time
    - Results and errors

    Example:
        wrapper = TracedToolWrapper(
            tool_name="search_patients",
            tool_fn=search_patients,
            tracer=tracer,
            trace_id=trace_id
        )
        result = await wrapper(condition="diabetes")
    """

    def __init__(
        self,
        tool_name: str,
        tool_fn: Callable,
        tracer: AgentExecutionTracer,
        trace_id: str,
    ):
        """Initialize traced tool wrapper.

        Args:
            tool_name: Name of the tool
            tool_fn: Original tool function
            tracer: AgentExecutionTracer instance
            trace_id: Current trace ID
        """
        self.tool_name = tool_name
        self.tool_fn = tool_fn
        self.tracer = tracer
        self.trace_id = trace_id

    async def __call__(self, *args, **kwargs) -> Any:
        """Execute tool with automatic tracing.

        Args:
            *args: Positional arguments for tool
            **kwargs: Keyword arguments for tool

        Returns:
            Tool result
        """
        # Log tool call
        call_id = self.tracer.log_tool_call(
            trace_id=self.trace_id,
            tool_name=self.tool_name,
            arguments=kwargs,
        )

        start_time = time.time()
        error = None
        result = None

        try:
            # Execute tool
            if asyncio.iscoroutinefunction(self.tool_fn):
                result = await self.tool_fn(*args, **kwargs)
            else:
                result = self.tool_fn(*args, **kwargs)

            return result

        except Exception as e:
            error = str(e)
            raise

        finally:
            # Log result
            execution_time_ms = (time.time() - start_time) * 1000
            self.tracer.log_tool_result(
                trace_id=self.trace_id,
                call_id=call_id,
                result=result,
                execution_time_ms=execution_time_ms,
                error=error,
            )


def create_traced_tool_wrapper(
    tool_name: str,
    tool_fn: Callable,
    trace_id: str,
    tracer: AgentExecutionTracer | None = None,
) -> TracedToolWrapper:
    """Factory function to create traced tool wrapper.

    Args:
        tool_name: Name of the tool
        tool_fn: Original tool function
        trace_id: Current trace ID
        tracer: AgentExecutionTracer instance (uses global if not provided)

    Returns:
        TracedToolWrapper instance

    Example:
        wrapper = create_traced_tool_wrapper(
            "search_patients",
            search_patients_fn,
            trace_id
        )
    """
    if tracer is None:
        tracer = get_tracer()

    return TracedToolWrapper(tool_name, tool_fn, tracer, trace_id)


def wrap_tools_for_tracing(
    tools: list[Any],
    trace_id: str,
    tracer: AgentExecutionTracer | None = None,
) -> list[Any]:
    """Wrap multiple tools for automatic tracing.

    Args:
        tools: List of tools to wrap
        trace_id: Current trace ID
        tracer: AgentExecutionTracer instance (uses global if not provided)

    Returns:
        List of wrapped tools

    Example:
        wrapped_tools = wrap_tools_for_tracing(
            tools=[tool1, tool2, tool3],
            trace_id=trace_id
        )
    """
    if tracer is None:
        tracer = get_tracer()

    wrapped_tools = []

    for tool in tools:
        # Extract tool name
        tool_name = "unknown"
        if hasattr(tool, "metadata") and hasattr(tool.metadata, "name"):
            tool_name = tool.metadata.name
        elif hasattr(tool, "name"):
            tool_name = tool.name

        # Extract function
        tool_fn = None
        if hasattr(tool, "fn"):
            tool_fn = tool.fn
        elif callable(tool):
            tool_fn = tool

        if tool_fn:
            # Create wrapper
            wrapper = create_traced_tool_wrapper(tool_name, tool_fn, trace_id, tracer)

            # Replace tool function with wrapper
            if hasattr(tool, "fn"):
                tool.fn = wrapper
            wrapped_tools.append(tool)
        else:
            wrapped_tools.append(tool)

    return wrapped_tools
