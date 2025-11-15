"""Visualization utilities for execution traces."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent_execution.trace_models import AgentExecutionTrace


def visualize_trace_ascii(trace: "AgentExecutionTrace") -> str:
    """Create ASCII visualization of execution flow.

    Args:
        trace: AgentExecutionTrace to visualize

    Returns:
        Formatted ASCII art showing the execution flow
    """
    lines = [
        "=" * 100,
        f"Agent Execution Trace: {trace.trace_id}",
        "=" * 100,
        f"Session: {trace.session_id}",
        f"Query: {trace.query}",
        f"Total Time: {trace.total_execution_time_ms:.2f}ms"
        if trace.total_execution_time_ms
        else "In Progress",
        f"Tool Calls: {trace.total_tool_calls}",
        f"Tool Sequence: {' → '.join(trace.get_tool_sequence())}",
        "",
        "Execution Flow:",
        "-" * 100,
    ]

    for step in trace.steps:
        step_header = f"[Step {step.step_number}] {step.step_type.value.upper()}"
        if step.duration_ms:
            step_header += f" ({step.duration_ms:.2f}ms)"
        lines.append(step_header)

        if step.reasoning_text:
            lines.append(f"  💭 Reasoning: {step.reasoning_text}")

        if step.tool_call:
            tc = step.tool_call
            lines.append(f"  🔧 Tool: {tc.tool_name}")
            lines.append(f"     Arguments: {json.dumps(tc.arguments, indent=2)}")
            if tc.result:
                result_preview = str(tc.result)[:200]
                lines.append(f"     Result: {result_preview}...")
            if tc.error:
                lines.append(f"     ❌ Error: {tc.error}")

        if step.final_response:
            response_preview = step.final_response[:300]
            lines.append(f"  ✅ Final Response: {response_preview}...")

        if step.error_message:
            lines.append(f"  ❌ Error: {step.error_message}")

        lines.append("")

    lines.append("=" * 100)
    return "\n".join(lines)
