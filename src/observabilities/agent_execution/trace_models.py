"""Execution trace data models for agent observabilities.

This module defines the core data structures for capturing agent execution:
- ExecutionStepType: Enum for different step types in agent execution
- ToolCall: Represents a single tool invocation with timing and results
- ExecutionStep: Represents a single step in the agent's decision-making process
- AgentExecutionTrace: Complete trace for a single query execution
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ExecutionStepType(str, Enum):
    """Types of execution steps in agent workflow."""

    REASONING = "reasoning"  # Agent thinking/planning
    TOOL_CALL = "tool_call"  # Tool invocation
    TOOL_RESULT = "tool_result"  # Tool response
    ERROR = "error"  # Error occurred
    COMPLETION = "completion"  # Final response


@dataclass
class ToolCall:
    """Represents a single tool invocation.

    Captures:
    - Tool name and arguments
    - Execution timing
    - Result and error information
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float | None = None
    result: Any | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert tool call to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "execution_time_ms": f"{self.execution_time_ms:.2f}"
            if self.execution_time_ms
            else None,
            "result": str(self.result)[:200] if self.result else None,  # Truncate for readability
            "error": self.error,
        }


@dataclass
class ExecutionStep:
    """Represents a single step in agent execution.

    Captures:
    - Step type (reasoning, tool call, result, error, completion)
    - Timing information
    - Step-specific data (reasoning text, tool call, response, etc.)
    - Metadata for extensibility
    """

    step_number: int
    step_type: ExecutionStepType
    timestamp: float = field(default_factory=time.time)
    duration_ms: float | None = None

    # For reasoning steps
    reasoning_text: str | None = None

    # For tool call steps
    tool_call: ToolCall | None = None

    # For completion steps
    final_response: str | None = None

    # For error steps
    error_message: str | None = None

    # Metadata for extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert execution step to dictionary for serialization."""
        data = {
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "duration_ms": f"{self.duration_ms:.2f}" if self.duration_ms else None,
        }

        if self.reasoning_text:
            data["reasoning"] = self.reasoning_text
        if self.tool_call:
            data["tool_call"] = self.tool_call.to_dict()
        if self.final_response:
            data["final_response"] = self.final_response[:500]  # Truncate
        if self.error_message:
            data["error"] = self.error_message
        if self.metadata:
            data["metadata"] = self.metadata

        return data


@dataclass
class AgentExecutionTrace:
    """Complete execution trace for a single query.

    Captures the entire lifecycle of a query from submission to completion:
    - Query and session information
    - All execution steps
    - Tool sequence and statistics
    - Total execution time
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    query: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    steps: list[ExecutionStep] = field(default_factory=list)

    # Summary statistics
    total_tool_calls: int = 0
    unique_tools_used: set[str] = field(default_factory=set)
    total_execution_time_ms: float | None = None

    def add_step(self, step: ExecutionStep) -> None:
        """Add an execution step and update statistics.

        Args:
            step: ExecutionStep to add
        """
        self.steps.append(step)

        # Update statistics
        if step.tool_call:
            self.total_tool_calls += 1
            self.unique_tools_used.add(step.tool_call.tool_name)

    def finalize(self) -> None:
        """Finalize the trace (calculate totals)."""
        self.end_time = time.time()
        self.total_execution_time_ms = (self.end_time - self.start_time) * 1000

    def get_tool_sequence(self) -> list[str]:
        """Get the sequence of tools called in order.

        Returns:
            List of tool names in execution order
        """
        return [step.tool_call.tool_name for step in self.steps if step.tool_call]

    def get_summary(self) -> dict[str, Any]:
        """Get high-level execution summary.

        Returns:
            Dictionary with key metrics about the execution
        """
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat()
            if self.end_time
            else None,
            "total_execution_time_ms": f"{self.total_execution_time_ms:.2f}"
            if self.total_execution_time_ms
            else None,
            "total_steps": len(self.steps),
            "total_tool_calls": self.total_tool_calls,
            "unique_tools_used": list(self.unique_tools_used),
            "tool_sequence": self.get_tool_sequence(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for serialization.

        Returns:
            Dictionary with summary and detailed steps
        """
        return {
            "summary": self.get_summary(),
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_json(self) -> str:
        """Convert trace to JSON string.

        Returns:
            JSON representation of the trace
        """
        return json.dumps(self.to_dict(), indent=2)

    def visualize_ascii(self) -> str:
        """Create ASCII visualization of execution flow.

        Returns:
            Formatted ASCII art showing the execution flow
        """
        # Import here to avoid circular imports
        from ..utils.visualization import visualize_trace_ascii

        return visualize_trace_ascii(self)
