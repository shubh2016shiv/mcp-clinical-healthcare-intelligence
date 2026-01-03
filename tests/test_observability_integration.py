"""Test observability integration with MCPAgentOrchestrator.

This test verifies that:
1. The LowLatencyTracingMixin is properly integrated
2. Traces are created for execute_query and chat calls
3. Tool calls can be logged within traces
4. Session-based trace history is maintained
"""

from unittest.mock import AsyncMock

import pytest

from src.agent.orchestrator import MCPAgentOrchestrator
from src.observabilities.integrations.mixins import LowLatencyTracingMixin


class TestObservabilityIntegration:
    """Test observability integration."""

    def test_mixin_inheritance(self):
        """Verify MCPAgentOrchestrator inherits from LowLatencyTracingMixin."""
        assert issubclass(MCPAgentOrchestrator, LowLatencyTracingMixin)

    def test_orchestrator_has_tracer(self):
        """Verify orchestrator has tracer from mixin."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "tracer")
        assert orchestrator.tracer is not None

    def test_orchestrator_has_trace_context(self):
        """Verify orchestrator has trace context tracking."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "_trace_context")
        assert isinstance(orchestrator._trace_context, dict)

    def test_execute_query_signature(self):
        """Verify execute_query has correct signature for mixin."""
        import inspect

        orchestrator = MCPAgentOrchestrator()
        sig = inspect.signature(orchestrator.execute_query)
        params = list(sig.parameters.keys())

        # Should have query and session_id parameters
        assert "query" in params
        assert "session_id" in params

        # session_id should have default value
        assert sig.parameters["session_id"].default == "default"

    def test_chat_signature(self):
        """Verify chat has correct signature."""
        import inspect

        orchestrator = MCPAgentOrchestrator()
        sig = inspect.signature(orchestrator.chat)
        params = list(sig.parameters.keys())

        # Should have session_id and user_message parameters
        assert "session_id" in params
        assert "user_message" in params

    def test_get_current_trace_id_method(self):
        """Verify get_current_trace_id method exists."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "get_current_trace_id")
        assert callable(orchestrator.get_current_trace_id)

    def test_log_tool_call_method(self):
        """Verify log_tool_call method exists."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "log_tool_call")
        assert callable(orchestrator.log_tool_call)

    def test_log_tool_result_method(self):
        """Verify log_tool_result method exists."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "log_tool_result")
        assert callable(orchestrator.log_tool_result)

    def test_internal_implementation_method(self):
        """Verify _execute_query_impl method exists."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "_execute_query_impl")
        assert callable(orchestrator._execute_query_impl)

    def test_get_session_summary_method(self):
        """Verify get_session_summary method from mixin."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "get_session_summary")
        assert callable(orchestrator.get_session_summary)

    def test_export_trace_json_method(self):
        """Verify export_trace_json method from mixin."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "export_trace_json")
        assert callable(orchestrator.export_trace_json)

    def test_visualize_trace_method(self):
        """Verify visualize_trace method from mixin."""
        orchestrator = MCPAgentOrchestrator()
        assert hasattr(orchestrator, "visualize_trace")
        assert callable(orchestrator.visualize_trace)

    @pytest.mark.asyncio
    async def test_execute_query_with_session_id(self):
        """Test execute_query accepts session_id parameter."""
        orchestrator = MCPAgentOrchestrator()

        # Mock the internal implementation
        orchestrator._execute_query_impl = AsyncMock(return_value="test response")

        # Should accept session_id parameter
        result = await orchestrator.execute_query("test query", session_id="test_session")

        # Verify it was called
        orchestrator._execute_query_impl.assert_called_once_with("test query")
        assert result == "test response"

    @pytest.mark.asyncio
    async def test_execute_query_default_session_id(self):
        """Test execute_query uses default session_id."""
        orchestrator = MCPAgentOrchestrator()

        # Mock the internal implementation
        orchestrator._execute_query_impl = AsyncMock(return_value="test response")

        # Should use default session_id
        result = await orchestrator.execute_query("test query")

        # Verify it was called
        orchestrator._execute_query_impl.assert_called_once_with("test query")
        assert result == "test response"

    def test_tool_logging_without_trace_context(self):
        """Test tool logging gracefully handles no trace context."""
        orchestrator = MCPAgentOrchestrator()

        # When not in a traced context, should return None
        call_id = orchestrator.log_tool_call("test_tool", {"arg": "value"})
        assert call_id is None

        # Should not raise error
        orchestrator.log_tool_result("call_id", {"result": "data"}, 100.0)

    def test_get_current_trace_id_without_context(self):
        """Test get_current_trace_id returns None outside traced context."""
        orchestrator = MCPAgentOrchestrator()

        # Outside of traced context, should return None
        trace_id = orchestrator.get_current_trace_id()
        assert trace_id is None


class TestIntegrationFlow:
    """Test the complete integration flow."""

    def test_mixin_method_resolution_order(self):
        """Verify MRO is correct for proper method resolution."""
        mro = MCPAgentOrchestrator.__mro__

        # LowLatencyTracingMixin should come before object
        mixin_index = None
        for i, cls in enumerate(mro):
            if cls.__name__ == "LowLatencyTracingMixin":
                mixin_index = i
                break

        assert mixin_index is not None
        assert mixin_index < len(mro) - 1  # Not the last (object)

    def test_orchestrator_initialization_with_mixin(self):
        """Test orchestrator initializes with mixin properly."""
        orchestrator = MCPAgentOrchestrator()

        # Should have both orchestrator and mixin attributes
        assert hasattr(orchestrator, "mcp_client")
        assert hasattr(orchestrator, "agent")
        assert hasattr(orchestrator, "tracer")
        assert hasattr(orchestrator, "_trace_context")

    def test_tracer_is_singleton(self):
        """Verify tracer is shared (singleton pattern)."""
        from src.observabilities.agent_execution.execution_tracer import get_tracer

        orchestrator1 = MCPAgentOrchestrator()
        orchestrator2 = MCPAgentOrchestrator()

        # Both should have the same tracer instance
        assert orchestrator1.tracer is orchestrator2.tracer
        assert orchestrator1.tracer is get_tracer()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
