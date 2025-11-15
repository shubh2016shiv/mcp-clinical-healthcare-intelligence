"""Context managers for observability tracing.

This module provides context managers that simplify tracing operations.
Context managers ensure proper setup/teardown and make code more readable.

Example:
    with TraceContext(session_id, query) as trace:
        # All operations within this block are automatically traced
        result = await some_operation()
        trace.log_reasoning("Processing result...")
"""

import asyncio
import logging
import time
from typing import Any

from ..agent_execution.execution_tracer import AgentExecutionTracer, get_tracer
from ..agent_execution.trace_models import AgentExecutionTrace

logger = logging.getLogger(__name__)


class TraceContext:
    """Context manager for managing execution trace lifecycle.

    Automatically starts and finalizes a trace. All operations within
    the context are automatically associated with the trace.

    Example:
        with TraceContext("session_123", "Find patients") as trace:
            tracer.log_reasoning(trace.trace_id, "Starting search...")
            result = await search_patients()
            tracer.finalize_trace(trace.trace_id, str(result))
    """

    def __init__(
        self,
        session_id: str,
        query: str,
        tracer: AgentExecutionTracer | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize trace context.

        Args:
            session_id: Session identifier
            query: Query or operation description
            tracer: AgentExecutionTracer instance (uses global if not provided)
            metadata: Additional metadata for the trace
        """
        self.session_id = session_id
        self.query = query
        self.tracer = tracer or get_tracer()
        self.metadata = metadata or {}
        self.trace: AgentExecutionTrace | None = None
        self.trace_id: str | None = None
        self.start_time: float | None = None

    def __enter__(self) -> "TraceContext":
        """Enter trace context."""
        self.trace = self.tracer.start_trace(
            session_id=self.session_id,
            query=self.query,
            metadata=self.metadata,
        )
        self.trace_id = self.trace.trace_id
        self.start_time = time.time()
        logger.debug(f"Started trace context: {self.trace_id[:8]}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit trace context and finalize trace."""
        if self.trace_id is None:
            return

        execution_time_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0.0

        # Log error if exception occurred
        if exc_type is not None:
            error_msg = f"{exc_type.__name__}: {str(exc_val)}" if exc_val else exc_type.__name__
            self.tracer.log_error(self.trace_id, error_msg)

        # Finalize with appropriate response
        final_response = "Completed successfully" if exc_type is None else "Failed with error"
        self.tracer.finalize_trace(self.trace_id, final_response)

        logger.debug(
            f"Finalized trace context: {self.trace_id[:8]} " f"({execution_time_ms:.2f}ms)"
        )

    @property
    def id(self) -> str | None:
        """Get current trace ID."""
        return self.trace_id

    def log_reasoning(self, reasoning_text: str, metadata: dict[str, Any] | None = None) -> None:
        """Log reasoning step in the trace.

        Args:
            reasoning_text: Reasoning text
            metadata: Additional metadata
        """
        if self.trace_id:
            self.tracer.log_reasoning(self.trace_id, reasoning_text, metadata)

    def log_error(self, error_message: str, metadata: dict[str, Any] | None = None) -> None:
        """Log error in the trace.

        Args:
            error_message: Error message
            metadata: Additional metadata
        """
        if self.trace_id:
            self.tracer.log_error(self.trace_id, error_message, metadata)


class ToolCallContext:
    """Context manager for tracing individual tool calls.

    Automatically logs tool invocation, execution time, and results.

    Example:
        with ToolCallContext("search_patients", {"condition": "diabetes"}, trace_id) as tool_call:
            result = await search_patients(condition="diabetes")
            tool_call.set_result(result)
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        trace_id: str,
        tracer: AgentExecutionTracer | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize tool call context.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            trace_id: Current trace ID
            tracer: AgentExecutionTracer instance (uses global if not provided)
            metadata: Additional metadata
        """
        self.tool_name = tool_name
        self.arguments = arguments
        self.trace_id = trace_id
        self.tracer = tracer or get_tracer()
        self.metadata = metadata or {}
        self.call_id: str | None = None
        self.start_time: float | None = None
        self.result: Any = None
        self.error: str | None = None

    def __enter__(self) -> "ToolCallContext":
        """Enter tool call context."""
        self.call_id = self.tracer.log_tool_call(
            trace_id=self.trace_id,
            tool_name=self.tool_name,
            arguments=self.arguments,
            metadata=self.metadata,
        )
        self.start_time = time.time()
        logger.debug(f"Started tool call: {self.tool_name} (call_id: {self.call_id[:8]})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit tool call context and log result."""
        if self.call_id is None:
            return

        execution_time_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0.0

        # Log error if exception occurred
        if exc_type is not None:
            self.error = f"{exc_type.__name__}: {str(exc_val)}" if exc_val else exc_type.__name__

        # Log result
        self.tracer.log_tool_result(
            trace_id=self.trace_id,
            call_id=self.call_id,
            result=self.result,
            execution_time_ms=execution_time_ms,
            error=self.error,
        )

        logger.debug(
            f"Completed tool call: {self.tool_name} "
            f"({execution_time_ms:.2f}ms, {'error' if self.error else 'success'})"
        )

    def set_result(self, result: Any) -> None:
        """Set tool execution result.

        Args:
            result: Tool execution result
        """
        self.result = result


class ExecutionContext:
    """Context manager for managing current execution trace ID.

    Stores trace ID in task-local storage for nested tracing contexts.
    Allows automatic trace ID discovery in nested operations.

    Example:
        with TraceContext("session_123", "Query") as trace:
            with ExecutionContext(trace.id):
                # Nested operations can discover trace_id automatically
                result = await some_operation()
    """

    def __init__(self, trace_id: str):
        """Initialize execution context.

        Args:
            trace_id: Trace ID for this context
        """
        self.trace_id = trace_id

    def __enter__(self) -> "ExecutionContext":
        """Enter execution context."""
        task = asyncio.current_task()
        if task is None:
            logger.warning("No current task for execution context")
            return self

        if not hasattr(task, "_execution_context_stack"):
            task._execution_context_stack = []

        task._execution_context_stack.append(self.trace_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit execution context."""
        task = asyncio.current_task()
        if task and hasattr(task, "_execution_context_stack"):
            if task._execution_context_stack:
                task._execution_context_stack.pop()

    @staticmethod
    def get_current_trace_id() -> str | None:
        """Get current trace ID from context.

        Returns:
            Current trace ID or None if not in context
        """
        task = asyncio.current_task()
        if task and hasattr(task, "_execution_context_stack"):
            if task._execution_context_stack:
                return task._execution_context_stack[-1]
        return None


class SessionContext:
    """Context manager for managing session-level tracing.

    Provides session-level utilities and ensures all traces in a session
    are properly associated.

    Example:
        with SessionContext("user_123") as session:
            # All traces created in this context are associated with the session
            with TraceContext(session.id, "Query 1") as trace:
                ...
    """

    def __init__(self, session_id: str):
        """Initialize session context.

        Args:
            session_id: Session identifier
        """
        self.session_id = session_id

    def __enter__(self) -> "SessionContext":
        """Enter session context."""
        logger.debug(f"Started session context: {self.session_id[:8]}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit session context."""
        logger.debug(f"Ended session context: {self.session_id[:8]}")

    @property
    def id(self) -> str:
        """Get session ID."""
        return self.session_id

    def get_summary(self, tracer: AgentExecutionTracer | None = None) -> dict[str, Any]:
        """Get session summary.

        Args:
            tracer: AgentExecutionTracer instance (uses global if not provided)

        Returns:
            Session summary
        """
        if tracer is None:
            tracer = get_tracer()
        return tracer.get_session_summary(self.session_id)
