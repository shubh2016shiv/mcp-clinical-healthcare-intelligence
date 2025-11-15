"""Tool call tracer for efficient tool execution logging.

This module provides utilities for tracing individual tool calls
with minimal overhead.
"""

import logging
from typing import Any

from ..agent_execution.execution_tracer import AgentExecutionTracer, get_tracer

logger = logging.getLogger(__name__)


class ToolCallTracer:
    """Efficient tool call tracing with minimal overhead.

    Designed to be called from within tool execution to log:
    - Tool invocation
    - Execution time
    - Results and errors

    All operations are O(1) and non-blocking.

    Example:
        tracer = ToolCallTracer()
        tracer.log_tool_execution(
            tool_name="search_patients",
            arguments={"condition": "diabetes"},
            result={"count": 5},
            execution_time_ms=150.5
        )
    """

    def __init__(self, tracer: AgentExecutionTracer | None = None):
        """Initialize tool call tracer.

        Args:
            tracer: AgentExecutionTracer instance (uses global if not provided)
        """
        self.tracer = tracer or get_tracer()

    def log_tool_execution(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        execution_time_ms: float,
        error: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Log a complete tool execution (O(1) operation).

        This is the most efficient way to log tool calls - single operation
        instead of separate log_tool_call and log_tool_result.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            execution_time_ms: Execution time in milliseconds
            error: Error message if failed (optional)
            trace_id: Trace ID (uses current if not provided)

        Example:
            tracer.log_tool_execution(
                tool_name="search_patients",
                arguments={"condition": "diabetes"},
                result={"count": 5},
                execution_time_ms=150.5
            )
        """
        if trace_id is None:
            # Try to get from current task context
            from .context_managers import ExecutionContext

            trace_id = ExecutionContext.get_current_trace_id()

            if trace_id is None:
                logger.debug("No active trace for tool execution")
                return

        # Log tool call and result in single operation
        call_id = self.tracer.log_tool_call(
            trace_id,
            tool_name,
            arguments,
        )

        self.tracer.log_tool_result(
            trace_id,
            call_id,
            result,
            execution_time_ms,
            error,
        )
