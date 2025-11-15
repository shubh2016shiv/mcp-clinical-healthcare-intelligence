"""Agent execution observabilities module.

This module provides comprehensive observabilities into agent execution:
- Execution tracing: Capture what tools the agent uses and why
- Tool call tracking: Log tool invocations, arguments, results, and timing
- Session-based history: Track execution across multiple queries
- Performance profiling: Measure execution time per tool and overall
- Visualization: ASCII and JSON export of execution traces

Main Components:
- AgentExecutionTracer: Core tracer for capturing execution
- ExecutionTrace: Data models for traces and steps
- OrchestratorIntegration: Integration with MCPAgentOrchestrator
- LlamaIndexCallback: Integration with LlamaIndex agents
- TraceUtils: Analysis and export utilities

Quick Start:
    >>> from src.observabilities import get_tracer
    >>> tracer = get_tracer()
    >>> trace = tracer.start_trace("session_123", "Find patients with diabetes")
    >>> tracer.log_reasoning(trace.trace_id, "Need to search patient database")
    >>> call_id = tracer.log_tool_call(trace.trace_id, "search_patients", {"condition": "diabetes"})
    >>> tracer.log_tool_result(trace.trace_id, call_id, {"count": 5}, 150.5)
    >>> tracer.finalize_trace(trace.trace_id, "Found 5 patients")
    >>> print(trace.visualize_ascii())
"""

# Import from new locations - backward compatible
from .agent_execution import (
    AgentExecutionTrace,
    AgentExecutionTracer,
    ExecutionStep,
    ExecutionStepType,
    ToolCall,
    TraceAnalyzer,
    get_tracer,
)
from .clinical_data_query import (
    ClinicalQueryManager,
    MongoDBQueryContext,
    MongoDBQueryTrace,
    MongoDBQueryTracer,
    QueryType,
    get_query_tracer,
)
from .integrations import (
    AgentTracingCallback,
    AsyncTraceBuffer,
    ExecutionContext,
    LowLatencyTracingMixin,
    OrchestratorTracingMixin,
    SessionContext,
    ToolCallContext,
    ToolCallTracer,
    TraceContext,
    TracedToolWrapper,
    create_agent_with_tracing,
    create_traced_tool_wrapper,
    wrap_tools_for_tracing,
)
from .performance import LatencyMonitor, get_latency_monitor
from .utils import (
    TraceExporter,
    get_session_summary,
    get_trace_summary,
    print_trace_visualization,
)

__all__ = [
    # Core tracer
    "AgentExecutionTracer",
    "get_tracer",
    # Data models
    "AgentExecutionTrace",
    "ExecutionStep",
    "ExecutionStepType",
    "ToolCall",
    # Clinical data query observability (HIPAA compliant)
    "MongoDBQueryTrace",
    "MongoDBQueryTracer",
    "MongoDBQueryContext",
    "ClinicalQueryManager",
    "QueryType",
    "get_query_tracer",
    # Low-latency integration (recommended)
    "LowLatencyTracingMixin",
    "ToolCallTracer",
    "AsyncTraceBuffer",
    "LatencyMonitor",
    "get_latency_monitor",
    # Orchestrator integration
    "OrchestratorTracingMixin",
    "TracedToolWrapper",
    # Context Managers
    "TraceContext",
    "ToolCallContext",
    "ExecutionContext",
    "SessionContext",
    "create_traced_tool_wrapper",
    "wrap_tools_for_tracing",
    # LlamaIndex integration
    "AgentTracingCallback",
    "create_agent_with_tracing",
    # Utilities
    "TraceAnalyzer",
    "TraceExporter",
    "get_trace_summary",
    "get_session_summary",
    "print_trace_visualization",
]
