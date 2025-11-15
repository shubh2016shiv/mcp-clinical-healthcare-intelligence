"""Analyzer for agent execution traces."""

import logging
from typing import Any

from .execution_tracer import AgentExecutionTracer, get_tracer
from .trace_models import ExecutionStepType

logger = logging.getLogger(__name__)


class TraceAnalyzer:
    """Analyzer for agent execution traces.

    Provides methods to analyze and extract insights from traces.
    """

    def __init__(self, tracer: AgentExecutionTracer | None = None):
        """Initialize trace analyzer.

        Args:
            tracer: AgentExecutionTracer instance (uses global if not provided)
        """
        self.tracer = tracer or get_tracer()

    def get_tool_execution_stats(self, trace_id: str) -> dict[str, Any]:
        """Get execution statistics for tools in a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            Dictionary with tool execution stats (name, count, total time, avg time)
        """
        trace = self.tracer.get_trace(trace_id)
        if not trace:
            return {}

        tool_stats = {}

        for step in trace.steps:
            if step.tool_call:
                tool_name = step.tool_call.tool_name
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "name": tool_name,
                        "call_count": 0,
                        "total_time_ms": 0.0,
                        "min_time_ms": float("inf"),
                        "max_time_ms": 0.0,
                        "errors": 0,
                    }

                stats = tool_stats[tool_name]
                stats["call_count"] += 1

                if step.tool_call.execution_time_ms:
                    stats["total_time_ms"] += step.tool_call.execution_time_ms
                    stats["min_time_ms"] = min(
                        stats["min_time_ms"], step.tool_call.execution_time_ms
                    )
                    stats["max_time_ms"] = max(
                        stats["max_time_ms"], step.tool_call.execution_time_ms
                    )

                if step.tool_call.error:
                    stats["errors"] += 1

        # Calculate averages
        for stats in tool_stats.values():
            if stats["call_count"] > 0:
                stats["avg_time_ms"] = stats["total_time_ms"] / stats["call_count"]
            if stats["min_time_ms"] == float("inf"):
                stats["min_time_ms"] = 0.0

        return tool_stats

    def get_reasoning_steps(self, trace_id: str) -> list[str]:
        """Get all reasoning steps from a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            List of reasoning texts
        """
        trace = self.tracer.get_trace(trace_id)
        if not trace:
            return []

        return [
            step.reasoning_text
            for step in trace.steps
            if step.step_type == ExecutionStepType.REASONING and step.reasoning_text
        ]

    def get_errors(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all errors from a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            List of error information
        """
        trace = self.tracer.get_trace(trace_id)
        if not trace:
            return []

        errors = []

        for step in trace.steps:
            if step.step_type == ExecutionStepType.ERROR:
                errors.append(
                    {
                        "step": step.step_number,
                        "message": step.error_message,
                        "timestamp": step.timestamp,
                    }
                )
            elif step.tool_call and step.tool_call.error:
                errors.append(
                    {
                        "step": step.step_number,
                        "tool": step.tool_call.tool_name,
                        "message": step.tool_call.error,
                        "timestamp": step.tool_call.timestamp,
                    }
                )

        return errors

    def get_session_performance_report(self, session_id: str) -> dict[str, Any]:
        """Get performance report for a session.

        Args:
            session_id: Session identifier

        Returns:
            Performance metrics and statistics
        """
        traces = self.tracer.get_session_traces(session_id)
        if not traces:
            return {"session_id": session_id, "message": "No traces found"}

        # Calculate statistics
        execution_times = [t.total_execution_time_ms for t in traces if t.total_execution_time_ms]

        if not execution_times:
            return {"session_id": session_id, "message": "No execution times recorded"}

        total_tool_calls = sum(t.total_tool_calls for t in traces)
        all_tools = set()
        for trace in traces:
            all_tools.update(trace.unique_tools_used)

        return {
            "session_id": session_id,
            "total_queries": len(traces),
            "total_tool_calls": total_tool_calls,
            "unique_tools": len(all_tools),
            "tools_used": list(all_tools),
            "execution_time": {
                "min_ms": f"{min(execution_times):.2f}",
                "max_ms": f"{max(execution_times):.2f}",
                "avg_ms": f"{sum(execution_times) / len(execution_times):.2f}",
                "total_ms": f"{sum(execution_times):.2f}",
            },
        }
