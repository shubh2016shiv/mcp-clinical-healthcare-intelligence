"""Agent execution tracer for observabilities.

This module provides the main tracer class for capturing agent execution:
- Session-based trace storage
- Tool call tracking with timing
- Structured execution logs
- Query audit trail
- Performance profiling per step

Enterprise Pattern:
- Execution trace per session
- Tool call graph visualization
- Structured execution logs
- Query audit trail
- Performance profiling per step
"""

import asyncio
import json
import logging
from typing import Any

from .trace_models import (
    AgentExecutionTrace,
    ExecutionStep,
    ExecutionStepType,
    ToolCall,
)

logger = logging.getLogger(__name__)


class AgentExecutionTracer:
    """Tracer for agent execution flow.

    This class captures the complete decision-making process of the agent:
    - What query was asked
    - How the agent reasoned about the query
    - Which tools it decided to call
    - What arguments it passed to each tool
    - What results it got back
    - How it synthesized the final response

    Enterprise Pattern:
    - Session-based trace storage
    - Structured execution logs
    - Query audit trail
    - Performance profiling
    - Exportable traces (JSON, visualization)

    Example:
        >>> tracer = AgentExecutionTracer()
        >>> trace = tracer.start_trace(
        ...     session_id="user123",
        ...     query="Find patients with diabetes"
        ... )
        >>> tracer.log_reasoning(trace.trace_id, "I need to search patients...")
        >>> tracer.log_tool_call(trace.trace_id, "search_patients", {"condition": "diabetes"})
        >>> tracer.log_tool_result(trace.trace_id, call_id, {"count": 5}, 150.5)
        >>> tracer.finalize_trace(trace.trace_id, "Found 5 patients...")
        >>> print(trace.visualize_ascii())
    """

    def __init__(self, max_traces_per_session: int = 100):
        """Initialize the tracer.

        Args:
            max_traces_per_session: Maximum traces to keep per session (LRU eviction)
        """
        self.max_traces_per_session = max_traces_per_session

        # Storage: session_id -> list of traces
        self._session_traces: dict[str, list[AgentExecutionTrace]] = {}

        # Active traces: trace_id -> trace
        self._active_traces: dict[str, AgentExecutionTrace] = {}

        # Current step counters per trace
        self._step_counters: dict[str, int] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"Initialized AgentExecutionTracer (max_traces_per_session={max_traces_per_session})"
        )

    def start_trace(
        self,
        session_id: str,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentExecutionTrace:
        """Start a new execution trace.

        Args:
            session_id: Session identifier (e.g., user ID, conversation ID)
            query: User query or request
            metadata: Additional metadata (optional)

        Returns:
            AgentExecutionTrace instance for this query
        """
        trace = AgentExecutionTrace(
            session_id=session_id,
            query=query,
        )

        self._active_traces[trace.trace_id] = trace
        self._step_counters[trace.trace_id] = 0

        logger.info(
            f"Started execution trace {trace.trace_id[:8]} "
            f"for session {session_id[:8]}: {query[:50]}..."
        )

        return trace

    def _get_next_step_number(self, trace_id: str) -> int:
        """Get next step number for a trace (internal).

        Args:
            trace_id: Trace identifier

        Returns:
            Next sequential step number
        """
        self._step_counters[trace_id] += 1
        return self._step_counters[trace_id]

    def log_reasoning(
        self,
        trace_id: str,
        reasoning_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log agent reasoning step.

        Captures the agent's thinking process, planning, or decision-making.

        Args:
            trace_id: Trace identifier
            reasoning_text: Agent's reasoning/thought process
            metadata: Additional metadata (optional)
        """
        if trace_id not in self._active_traces:
            logger.warning(f"Trace {trace_id} not found (already finalized?)")
            return

        trace = self._active_traces[trace_id]
        step = ExecutionStep(
            step_number=self._get_next_step_number(trace_id),
            step_type=ExecutionStepType.REASONING,
            reasoning_text=reasoning_text,
            metadata=metadata or {},
        )

        trace.add_step(step)
        logger.debug(f"[{trace_id[:8]}] Reasoning: {reasoning_text[:100]}...")

    def log_tool_call(
        self,
        trace_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Log tool call initiation.

        Called when the agent decides to invoke a tool.

        Args:
            trace_id: Trace identifier
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            metadata: Additional metadata (optional)

        Returns:
            Call ID for matching with result (use in log_tool_result)
        """
        if trace_id not in self._active_traces:
            logger.warning(f"Trace {trace_id} not found")
            return ""

        trace = self._active_traces[trace_id]

        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
        )

        step = ExecutionStep(
            step_number=self._get_next_step_number(trace_id),
            step_type=ExecutionStepType.TOOL_CALL,
            tool_call=tool_call,
            metadata=metadata or {},
        )

        trace.add_step(step)
        logger.info(
            f"[{trace_id[:8]}] Tool call: {tool_name} "
            f"with args: {json.dumps(arguments, default=str)[:100]}..."
        )

        return tool_call.call_id

    def log_tool_result(
        self,
        trace_id: str,
        call_id: str,
        result: Any,
        execution_time_ms: float,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log tool execution result.

        Called after a tool completes execution.

        Args:
            trace_id: Trace identifier
            call_id: Call ID from log_tool_call
            result: Tool execution result
            execution_time_ms: Execution time in milliseconds
            error: Error message if tool failed (optional)
            metadata: Additional metadata (optional)
        """
        if trace_id not in self._active_traces:
            logger.warning(f"Trace {trace_id} not found")
            return

        trace = self._active_traces[trace_id]

        # Find the corresponding tool call step and update it
        for step in reversed(trace.steps):
            if step.tool_call and step.tool_call.call_id == call_id:
                step.tool_call.result = result
                step.tool_call.execution_time_ms = execution_time_ms
                step.tool_call.error = error
                step.duration_ms = execution_time_ms
                break

        status = "failed" if error else "succeeded"
        logger.info(
            f"[{trace_id[:8]}] Tool call {call_id} {status} " f"in {execution_time_ms:.2f}ms"
        )

    def log_error(
        self,
        trace_id: str,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an error during execution.

        Args:
            trace_id: Trace identifier
            error_message: Error message
            metadata: Additional metadata (optional)
        """
        if trace_id not in self._active_traces:
            logger.warning(f"Trace {trace_id} not found")
            return

        trace = self._active_traces[trace_id]

        step = ExecutionStep(
            step_number=self._get_next_step_number(trace_id),
            step_type=ExecutionStepType.ERROR,
            error_message=error_message,
            metadata=metadata or {},
        )

        trace.add_step(step)
        logger.error(f"[{trace_id[:8]}] Error: {error_message}")

    def finalize_trace(
        self,
        trace_id: str,
        final_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentExecutionTrace | None:
        """Finalize an execution trace.

        Called when the agent completes processing and returns a response.

        Args:
            trace_id: Trace identifier
            final_response: Agent's final response
            metadata: Additional metadata (optional)

        Returns:
            Finalized AgentExecutionTrace, or None if not found
        """
        if trace_id not in self._active_traces:
            logger.warning(f"Trace {trace_id} not found")
            return None

        trace = self._active_traces[trace_id]

        # Add completion step
        step = ExecutionStep(
            step_number=self._get_next_step_number(trace_id),
            step_type=ExecutionStepType.COMPLETION,
            final_response=final_response,
            metadata=metadata or {},
        )
        trace.add_step(step)

        # Finalize trace (calculate totals)
        trace.finalize()

        # Move to session history
        if trace.session_id not in self._session_traces:
            self._session_traces[trace.session_id] = []

        self._session_traces[trace.session_id].append(trace)

        # Enforce max traces per session (LRU eviction)
        if len(self._session_traces[trace.session_id]) > self.max_traces_per_session:
            self._session_traces[trace.session_id].pop(0)

        # Remove from active traces
        del self._active_traces[trace_id]
        del self._step_counters[trace_id]

        logger.info(
            f"Finalized trace {trace_id[:8]} "
            f"({trace.total_tool_calls} tool calls, {trace.total_execution_time_ms:.2f}ms)"
        )

        return trace

    def get_trace(self, trace_id: str) -> AgentExecutionTrace | None:
        """Get a trace by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            AgentExecutionTrace or None if not found
        """
        # Check active traces first
        if trace_id in self._active_traces:
            return self._active_traces[trace_id]

        # Search in session history
        for traces in self._session_traces.values():
            for trace in traces:
                if trace.trace_id == trace_id:
                    return trace

        return None

    def get_session_traces(self, session_id: str) -> list[AgentExecutionTrace]:
        """Get all traces for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of traces (most recent last)
        """
        return self._session_traces.get(session_id, [])

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get summary statistics for a session.

        Aggregates metrics across all queries in a session.

        Args:
            session_id: Session identifier

        Returns:
            Summary statistics including total queries, tool calls, and recent queries
        """
        traces = self.get_session_traces(session_id)

        if not traces:
            return {
                "session_id": session_id,
                "total_queries": 0,
                "message": "No traces found for this session",
            }

        total_tool_calls = sum(t.total_tool_calls for t in traces)
        all_tools_used = set()
        for trace in traces:
            all_tools_used.update(trace.unique_tools_used)

        avg_execution_time = sum(
            t.total_execution_time_ms for t in traces if t.total_execution_time_ms
        ) / len(traces)

        return {
            "session_id": session_id,
            "total_queries": len(traces),
            "total_tool_calls": total_tool_calls,
            "unique_tools_used": list(all_tools_used),
            "avg_execution_time_ms": f"{avg_execution_time:.2f}",
            "recent_queries": [
                {
                    "trace_id": t.trace_id,
                    "query": t.query,
                    "tool_sequence": t.get_tool_sequence(),
                    "execution_time_ms": f"{t.total_execution_time_ms:.2f}"
                    if t.total_execution_time_ms
                    else None,
                }
                for t in traces[-5:]  # Last 5 queries
            ],
        }

    def export_trace_json(self, trace_id: str) -> str | None:
        """Export trace as JSON.

        Args:
            trace_id: Trace identifier

        Returns:
            JSON string or None if not found
        """
        trace = self.get_trace(trace_id)
        return trace.to_json() if trace else None

    def visualize_trace(self, trace_id: str) -> str | None:
        """Visualize trace as ASCII art.

        Args:
            trace_id: Trace identifier

        Returns:
            ASCII visualization or None if not found
        """
        trace = self.get_trace(trace_id)
        return trace.visualize_ascii() if trace else None

    def clear_session(self, session_id: str) -> None:
        """Clear all traces for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._session_traces:
            del self._session_traces[session_id]
            logger.info(f"Cleared traces for session {session_id[:8]}")

    def get_all_sessions(self) -> list[str]:
        """Get all session IDs with traces.

        Returns:
            List of session IDs
        """
        return list(self._session_traces.keys())


# Global tracer instance
_tracer: AgentExecutionTracer | None = None


def get_tracer() -> AgentExecutionTracer:
    """Get or create global tracer instance.

    Returns:
        Singleton AgentExecutionTracer instance
    """
    global _tracer
    if _tracer is None:
        _tracer = AgentExecutionTracer()
    return _tracer
