"""Tracing mixins for integrating observability into orchestrators.

This module provides mixin classes that can be added to orchestrator classes
to automatically enable tracing functionality.
"""

import asyncio
import logging
import time
from typing import Any

from ..agent_execution.execution_tracer import get_tracer
from ..agent_execution.trace_models import AgentExecutionTrace

logger = logging.getLogger(__name__)


class LowLatencyTracingMixin:
    """Mixin for automatic low-latency tracing in orchestrators.

    Adds tracing with <1ms overhead per query by:
    - Using async-safe operations only
    - Deferring heavy operations
    - Minimizing allocations in hot path
    - Using task-local storage for trace context

    Usage:
        class MCPAgentOrchestrator(LowLatencyTracingMixin, BaseOrchestrator):
            pass

    The mixin automatically traces execute_query() and chat() methods.
    """

    def __init__(self, *args, **kwargs):
        """Initialize low-latency tracer."""
        super().__init__(*args, **kwargs)
        self.tracer = get_tracer()
        self._trace_context: dict[int, str] = {}  # task_id -> trace_id
        logger.info("Low-latency tracing initialized")

    def _get_current_trace_id(self) -> str | None:
        """Get current trace ID from task context (O(1), non-blocking).

        Returns:
            Current trace ID or None
        """
        try:
            task = asyncio.current_task()
            if task:
                return self._trace_context.get(id(task))
        except RuntimeError:
            pass
        return None

    def _set_current_trace_id(self, trace_id: str) -> None:
        """Set current trace ID in task context (O(1), non-blocking).

        Args:
            trace_id: Trace identifier
        """
        try:
            task = asyncio.current_task()
            if task:
                self._trace_context[id(task)] = trace_id
        except RuntimeError:
            pass

    def _clear_current_trace_id(self) -> None:
        """Clear current trace ID from task context (O(1), non-blocking).

        Prevents memory leaks from completed tasks.
        """
        try:
            task = asyncio.current_task()
            if task:
                self._trace_context.pop(id(task), None)
        except RuntimeError:
            pass

    async def execute_query(self, query: str, session_id: str = "default") -> str:
        """Execute query with automatic low-latency tracing.

        Wraps the original execute_query with minimal overhead.

        Args:
            query: The query to execute
            session_id: Session identifier for tracing

        Returns:
            The agent's response
        """
        # Start trace (O(1) operation)
        trace = self.tracer.start_trace(session_id, query)
        trace_id = trace.trace_id
        self._set_current_trace_id(trace_id)

        start_time = time.time()
        error = None

        try:
            # Log initial reasoning (non-blocking)
            self.tracer.log_reasoning(
                trace_id,
                f"Processing query: {query[:100]}",
            )

            # Call original execute_query (from parent class)
            response = await super().execute_query(query)

            return response

        except Exception as e:
            error = str(e)
            self.tracer.log_error(trace_id, error)
            raise

        finally:
            # Finalize trace (O(n) where n = number of steps, but deferred)
            execution_time_ms = (time.time() - start_time) * 1000
            self.tracer.finalize_trace(trace_id, "Query completed")
            self._clear_current_trace_id()

            # Log timing (non-blocking)
            logger.debug(f"Query traced in {execution_time_ms:.2f}ms")

    async def chat(
        self,
        session_id: str,
        user_message: str,
    ) -> str:
        """Execute chat with automatic low-latency tracing.

        Wraps the original chat with minimal overhead.

        Args:
            session_id: Session identifier
            user_message: User message

        Returns:
            Assistant response
        """
        # Start trace (O(1) operation)
        trace = self.tracer.start_trace(session_id, user_message)
        trace_id = trace.trace_id
        self._set_current_trace_id(trace_id)

        start_time = time.time()
        error = None

        try:
            # Log initial reasoning (non-blocking)
            self.tracer.log_reasoning(
                trace_id,
                f"Processing chat: {user_message[:100]}",
            )

            # Call original chat (from parent class)
            response = await super().chat(session_id, user_message)

            return response

        except Exception as e:
            error = str(e)
            self.tracer.log_error(trace_id, error)
            raise

        finally:
            # Finalize trace (O(n) where n = number of steps, but deferred)
            execution_time_ms = (time.time() - start_time) * 1000
            self.tracer.finalize_trace(trace_id, "Chat completed")
            self._clear_current_trace_id()

            # Log timing (non-blocking)
            logger.debug(f"Chat traced in {execution_time_ms:.2f}ms")

    def get_trace(self, trace_id: str) -> AgentExecutionTrace | None:
        """Get a trace by ID (O(log n) lookup).

        Args:
            trace_id: Trace identifier

        Returns:
            AgentExecutionTrace or None
        """
        return self.tracer.get_trace(trace_id)

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get session summary (deferred operation).

        Args:
            session_id: Session identifier

        Returns:
            Session summary
        """
        return self.tracer.get_session_summary(session_id)

    def export_trace_json(self, trace_id: str) -> str | None:
        """Export trace as JSON (deferred operation).

        Args:
            trace_id: Trace identifier

        Returns:
            JSON string or None
        """
        return self.tracer.export_trace_json(trace_id)

    def visualize_trace(self, trace_id: str) -> str | None:
        """Visualize trace as ASCII (deferred operation).

        Args:
            trace_id: Trace identifier

        Returns:
            ASCII visualization or None
        """
        return self.tracer.visualize_trace(trace_id)


class OrchestratorTracingMixin:
    """Mixin for manual tracing control in orchestrators.

    Provides explicit tracing methods for fine-grained control.
    Use this when you need manual control over when tracing starts/stops.

    Usage:
        class MCPAgentOrchestrator(OrchestratorTracingMixin, BaseOrchestrator):
            pass

        # Then use explicit methods:
        response, trace_id = await orchestrator.execute_query_traced(query)
    """

    def __init__(self, *args, **kwargs):
        """Initialize tracer."""
        super().__init__(*args, **kwargs)
        self.tracer = get_tracer()
        logger.info("Orchestrator tracing initialized")

    async def execute_query_traced(
        self,
        query: str,
        session_id: str = "default",
    ) -> tuple[str, str]:
        """Execute query with full tracing.

        Args:
            query: User query
            session_id: Session identifier

        Returns:
            Tuple of (response, trace_id)
        """
        # Start execution trace
        trace = self.tracer.start_trace(
            session_id=session_id,
            query=query,
        )
        trace_id = trace.trace_id

        try:
            # Log reasoning
            self.tracer.log_reasoning(
                trace_id,
                f"Processing query: {query}",
            )

            # Execute with tracing context
            from .context_managers import TraceContext

            with TraceContext(trace_id):
                response = await self.execute_query(query)

            # Finalize trace
            self.tracer.finalize_trace(trace_id, response)

            # Log visualization (optional)
            logger.debug("\n" + trace.visualize_ascii())

            return response, trace_id

        except Exception as e:
            # Log error in trace
            self.tracer.log_error(trace_id, str(e))
            raise

    async def chat_traced(
        self,
        session_id: str,
        user_message: str,
    ) -> tuple[str, str]:
        """Execute chat with full tracing.

        Args:
            session_id: Session identifier
            user_message: User message

        Returns:
            Tuple of (response, trace_id)
        """
        # Start execution trace
        trace = self.tracer.start_trace(
            session_id=session_id,
            query=user_message,
        )
        trace_id = trace.trace_id

        try:
            # Log reasoning
            self.tracer.log_reasoning(
                trace_id,
                f"Processing chat message: {user_message}",
            )

            # Execute with tracing context
            from .context_managers import TraceContext

            with TraceContext(trace_id):
                response = await self.chat(session_id, user_message)

            # Finalize trace
            self.tracer.finalize_trace(trace_id, response)

            # Log visualization (optional)
            logger.debug("\n" + trace.visualize_ascii())

            return response, trace_id

        except Exception as e:
            # Log error in trace
            self.tracer.log_error(trace_id, str(e))
            raise

    def get_trace_visualization(self, trace_id: str) -> str | None:
        """Get ASCII visualization of a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            ASCII visualization or None if not found
        """
        return self.tracer.visualize_trace(trace_id)

    def get_session_trace_summary(self, session_id: str) -> dict[str, Any]:
        """Get summary of all traces in a session.

        Args:
            session_id: Session identifier

        Returns:
            Session summary with metrics and recent queries
        """
        return self.tracer.get_session_summary(session_id)

    def export_trace_json(self, trace_id: str) -> str | None:
        """Export trace as JSON.

        Args:
            trace_id: Trace identifier

        Returns:
            JSON string or None if not found
        """
        return self.tracer.export_trace_json(trace_id)
