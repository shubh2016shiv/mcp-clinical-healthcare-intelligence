# Agent Execution Observability

This module provides comprehensive tracing and analysis of agent execution flows.

## Overview

The agent execution observability module captures the complete lifecycle of query processing:
- **Execution Tracing**: What tools the agent chooses and why
- **Tool Call Tracking**: Tool invocations with arguments, results, and timing
- **Session Management**: Track execution across multiple queries
- **Performance Profiling**: Measure execution time per tool and overall
- **Visualization**: ASCII and JSON export of execution traces
- **Analysis**: Extract insights and statistics from traces

## Components

### `models.py`
Core data structures:
- `ExecutionStepType`: Enum for different step types
- `ToolCall`: Single tool invocation
- `ExecutionStep`: Single step in agent execution
- `AgentExecutionTrace`: Complete trace for a query

### `tracer.py`
Main tracer implementation:
- `AgentExecutionTracer`: Core tracer class
- `get_tracer()`: Singleton accessor

### `analyzer.py`
Analysis utilities:
- `TraceAnalyzer`: Extract insights from traces
- `get_tool_execution_stats()`: Tool statistics
- `get_reasoning_steps()`: Extract reasoning
- `get_errors()`: Extract errors
- `get_session_performance_report()`: Session metrics

## Quick Start

```python
from src.observabilities import get_tracer

tracer = get_tracer()

# Start trace
trace = tracer.start_trace("user_123", "Find patients with diabetes")

# Log reasoning
tracer.log_reasoning(trace.trace_id, "Searching patient database...")

# Log tool call
call_id = tracer.log_tool_call(
    trace.trace_id,
    "search_patients",
    {"condition": "diabetes"}
)

# Log result
tracer.log_tool_result(
    trace.trace_id,
    call_id,
    {"count": 5},
    execution_time_ms=150.5
)

# Finalize
tracer.finalize_trace(trace.trace_id, "Found 5 patients")

# Visualize
print(trace.visualize_ascii())
```

## Data Flow

```
Query Submission
    ↓
start_trace() → Create AgentExecutionTrace
    ↓
log_reasoning() → Add REASONING step
    ↓
log_tool_call() → Add TOOL_CALL step
    ↓
[Tool Execution]
    ↓
log_tool_result() → Update with result/timing
    ↓
[Repeat for each tool call]
    ↓
log_error() → Add ERROR step (if error)
    ↓
finalize_trace() → Add COMPLETION step + Calculate totals
    ↓
Move to session history + LRU eviction
    ↓
Available for analysis/export
```

## Analysis

```python
from src.observabilities import TraceAnalyzer

analyzer = TraceAnalyzer()

# Get tool stats
stats = analyzer.get_tool_execution_stats(trace_id)

# Get reasoning steps
reasoning = analyzer.get_reasoning_steps(trace_id)

# Get errors
errors = analyzer.get_errors(trace_id)

# Get session performance report
report = analyzer.get_session_performance_report(session_id)
```

## Best Practices

1. **Always finalize traces**: Ensures totals are calculated
2. **Use session IDs**: Enables audit trails and analysis
3. **Log reasoning**: Helps understand agent decisions
4. **Handle errors**: Use `log_error()` for exceptions
5. **Clean up old sessions**: Manage memory usage
6. **Use metadata**: Add custom context for debugging

