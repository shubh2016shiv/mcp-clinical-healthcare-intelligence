"""Agent execution observability module.

This module provides comprehensive observabilities into agent execution:
- Execution tracing: Capture what tools the agent uses and why
- Tool call tracking: Log tool invocations, arguments, results, and timing
- Session-based history: Track execution across multiple queries
- Performance profiling: Measure execution time per tool and overall
- Visualization: ASCII and JSON export of execution traces

File Organization:
- trace_models.py: Data models (ExecutionTrace, ToolCall, ExecutionStep)
- execution_tracer.py: Core tracer implementation (AgentExecutionTracer)
- trace_analyzer.py: Analysis utilities (TraceAnalyzer)
"""

from .execution_tracer import AgentExecutionTracer, get_tracer
from .trace_analyzer import TraceAnalyzer
from .trace_models import (
    AgentExecutionTrace,
    ExecutionStep,
    ExecutionStepType,
    ToolCall,
)

__all__ = [
    "AgentExecutionTracer",
    "get_tracer",
    "AgentExecutionTrace",
    "ExecutionStep",
    "ExecutionStepType",
    "ToolCall",
    "TraceAnalyzer",
]
