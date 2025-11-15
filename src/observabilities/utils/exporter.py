"""Exporter for agent execution traces."""

import json
import logging
from typing import Any

from ..agent_execution.execution_tracer import AgentExecutionTracer, get_tracer

logger = logging.getLogger(__name__)


class TraceExporter:
    """Exporter for agent execution traces.

    Provides methods to export traces to various formats.
    """

    def __init__(self, tracer: AgentExecutionTracer | None = None):
        """Initialize trace exporter.

        Args:
            tracer: AgentExecutionTracer instance (uses global if not provided)
        """
        self.tracer = tracer or get_tracer()

    def export_trace_json(self, trace_id: str, pretty: bool = True) -> str | None:
        """Export trace as JSON.

        Args:
            trace_id: Trace identifier
            pretty: Whether to pretty-print JSON

        Returns:
            JSON string or None if trace not found
        """
        trace = self.tracer.get_trace(trace_id)
        if not trace:
            return None

        if pretty:
            return trace.to_json()
        else:
            return json.dumps(trace.to_dict())

    def export_trace_csv(self, trace_id: str) -> str | None:
        """Export trace as CSV (tool calls only).

        Args:
            trace_id: Trace identifier

        Returns:
            CSV string or None if trace not found
        """
        trace = self.tracer.get_trace(trace_id)
        if not trace:
            return None

        lines = ["tool_name,call_id,execution_time_ms,error"]

        for step in trace.steps:
            if step.tool_call:
                tc = step.tool_call
                error = tc.error or ""
                lines.append(
                    f'"{tc.tool_name}","{tc.call_id}",{tc.execution_time_ms or 0},"{error}"'
                )

        return "\n".join(lines)

    def export_session_json(self, session_id: str, pretty: bool = True) -> str | None:
        """Export all traces in a session as JSON.

        Args:
            session_id: Session identifier
            pretty: Whether to pretty-print JSON

        Returns:
            JSON string or None if session not found
        """
        traces = self.tracer.get_session_traces(session_id)
        if not traces:
            return None

        data = {
            "session_id": session_id,
            "trace_count": len(traces),
            "traces": [trace.to_dict() for trace in traces],
        }

        if pretty:
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data)

    def export_trace_markdown(self, trace_id: str) -> str | None:
        """Export trace as Markdown.

        Args:
            trace_id: Trace identifier

        Returns:
            Markdown string or None if trace not found
        """
        trace = self.tracer.get_trace(trace_id)
        if not trace:
            return None

        lines = [
            f"# Execution Trace: {trace.trace_id}",
            "",
            "## Summary",
            f"- **Session**: {trace.session_id}",
            f"- **Query**: {trace.query}",
            f"- **Total Time**: {trace.total_execution_time_ms:.2f}ms"
            if trace.total_execution_time_ms
            else "- **Total Time**: In Progress",
            f"- **Tool Calls**: {trace.total_tool_calls}",
            f"- **Tools Used**: {', '.join(trace.get_tool_sequence())}",
            "",
            "## Execution Steps",
        ]

        for step in trace.steps:
            lines.append(f"### Step {step.step_number}: {step.step_type.value}")

            if step.reasoning_text:
                lines.append(f"**Reasoning**: {step.reasoning_text}")

            if step.tool_call:
                tc = step.tool_call
                lines.append(f"**Tool**: `{tc.tool_name}`")
                lines.append("**Arguments**: ```json")
                lines.append(json.dumps(tc.arguments, indent=2))
                lines.append("```")
                if tc.execution_time_ms:
                    lines.append(f"**Execution Time**: {tc.execution_time_ms:.2f}ms")
                if tc.result:
                    lines.append(f"**Result**: {str(tc.result)[:200]}")
                if tc.error:
                    lines.append(f"**Error**: {tc.error}")

            if step.final_response:
                lines.append(f"**Response**: {step.final_response[:500]}")

            if step.error_message:
                lines.append(f"**Error**: {step.error_message}")

            lines.append("")

        return "\n".join(lines)


def get_trace_summary(trace_id: str, tracer: AgentExecutionTracer | None = None) -> dict[str, Any]:
    """Get summary of a trace.

    Args:
        trace_id: Trace identifier
        tracer: AgentExecutionTracer instance (uses global if not provided)

    Returns:
        Summary dictionary
    """
    if tracer is None:
        tracer = get_tracer()

    trace = tracer.get_trace(trace_id)
    if not trace:
        return {}

    return trace.get_summary()


def get_session_summary(
    session_id: str, tracer: AgentExecutionTracer | None = None
) -> dict[str, Any]:
    """Get summary of a session.

    Args:
        session_id: Session identifier
        tracer: AgentExecutionTracer instance (uses global if not provided)

    Returns:
        Summary dictionary
    """
    if tracer is None:
        tracer = get_tracer()

    return tracer.get_session_summary(session_id)


def print_trace_visualization(trace_id: str, tracer: AgentExecutionTracer | None = None) -> None:
    """Print ASCII visualization of a trace.

    Args:
        trace_id: Trace identifier
        tracer: AgentExecutionTracer instance (uses global if not provided)
    """
    if tracer is None:
        tracer = get_tracer()

    visualization = tracer.visualize_trace(trace_id)
    if visualization:
        print(visualization)
    else:
        logger.warning(f"Trace {trace_id} not found")
