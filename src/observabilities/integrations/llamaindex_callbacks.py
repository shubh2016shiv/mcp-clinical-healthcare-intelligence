"""LlamaIndex callback handler for agent tracing."""

import logging
from typing import Any

from ..agent_execution.execution_tracer import AgentExecutionTracer, get_tracer

logger = logging.getLogger(__name__)


class AgentTracingCallback:
    """LlamaIndex callback handler for agent tracing.

    Integrates with LlamaIndex's callback system to capture:
    - Agent reasoning steps
    - Tool calls and results
    - Errors and exceptions
    """

    def __init__(
        self,
        tracer: AgentExecutionTracer | None = None,
        trace_id: str | None = None,
    ):
        """Initialize callback handler.

        Args:
            tracer: AgentExecutionTracer instance (uses global if not provided)
            trace_id: Current trace ID (required for logging events)
        """
        self.tracer = tracer or get_tracer()
        self.trace_id = trace_id
        self._tool_call_ids: dict[str, str] = {}  # Map event_id to call_id

        if not trace_id:
            logger.warning("AgentTracingCallback initialized without trace_id")

    def set_trace_id(self, trace_id: str) -> None:
        """Set the current trace ID.

        Args:
            trace_id: Trace identifier
        """
        self.trace_id = trace_id
        logger.debug(f"Callback trace_id set to {trace_id[:8]}")

    def on_event_start(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when an event starts.

        Args:
            event_type: Type of event (e.g., "function_call", "agent_step")
            payload: Event payload
            event_id: Event identifier
            **kwargs: Additional arguments

        Returns:
            Event ID
        """
        if not self.trace_id:
            return event_id

        try:
            # Handle function call events
            if event_type in ("function_call", "tool_call"):
                tool_name = self._extract_tool_name(payload)
                arguments = self._extract_arguments(payload)

                call_id = self.tracer.log_tool_call(
                    trace_id=self.trace_id,
                    tool_name=tool_name,
                    arguments=arguments,
                )

                # Store for matching with result
                self._tool_call_ids[event_id] = call_id
                logger.debug(f"Tool call event started: {tool_name}")

            # Handle agent step events
            elif event_type in ("agent_step", "reasoning"):
                thought = self._extract_thought(payload)
                if thought:
                    self.tracer.log_reasoning(
                        self.trace_id,
                        thought,
                    )
                    logger.debug("Reasoning event logged")

        except Exception as e:
            logger.error(f"Error in on_event_start: {e}")

        return event_id

    def on_event_end(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an event ends.

        Args:
            event_type: Type of event
            payload: Event payload
            event_id: Event identifier
            **kwargs: Additional arguments
        """
        if not self.trace_id:
            return

        try:
            # Handle function call completion
            if event_type in ("function_call", "tool_call"):
                call_id = self._tool_call_ids.get(event_id)
                if not call_id:
                    return

                result = self._extract_result(payload)
                execution_time_ms = self._extract_execution_time(payload)

                self.tracer.log_tool_result(
                    trace_id=self.trace_id,
                    call_id=call_id,
                    result=result,
                    execution_time_ms=execution_time_ms,
                )

                # Clean up
                del self._tool_call_ids[event_id]
                logger.debug(f"Tool call event ended: {call_id}")

        except Exception as e:
            logger.error(f"Error in on_event_end: {e}")

    def on_error(
        self,
        error: Exception,
        event_type: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an error occurs.

        Args:
            error: Exception that occurred
            event_type: Type of event where error occurred
            **kwargs: Additional arguments
        """
        if not self.trace_id:
            return

        try:
            error_message = f"{event_type}: {str(error)}" if event_type else str(error)
            self.tracer.log_error(self.trace_id, error_message)
            logger.debug(f"Error event logged: {error_message}")

        except Exception as e:
            logger.error(f"Error in on_error: {e}")

    # Helper methods for extracting data from payloads

    @staticmethod
    def _extract_tool_name(payload: dict[str, Any] | None) -> str:
        """Extract tool name from event payload.

        Args:
            payload: Event payload

        Returns:
            Tool name or "unknown"
        """
        if not payload:
            return "unknown"

        # Try different payload formats
        for key in ("tool_name", "function_call", "name", "tool"):
            if key in payload:
                value = payload[key]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict) and "name" in value:
                    return value["name"]

        return "unknown"

    @staticmethod
    def _extract_arguments(payload: dict[str, Any] | None) -> dict[str, Any]:
        """Extract tool arguments from event payload.

        Args:
            payload: Event payload

        Returns:
            Arguments dictionary
        """
        if not payload:
            return {}

        # Try different payload formats
        for key in ("arguments", "args", "kwargs", "params"):
            if key in payload:
                value = payload[key]
                if isinstance(value, dict):
                    return value

        # If no explicit arguments, return whole payload (excluding metadata)
        return {
            k: v
            for k, v in payload.items()
            if not k.startswith("_") and k not in ("tool_name", "name", "type")
        }

    @staticmethod
    def _extract_thought(payload: dict[str, Any] | None) -> str:
        """Extract agent thought/reasoning from event payload.

        Args:
            payload: Event payload

        Returns:
            Thought text or empty string
        """
        if not payload:
            return ""

        # Try different payload formats
        for key in ("thought", "reasoning", "message", "text", "content"):
            if key in payload:
                value = payload[key]
                if isinstance(value, str):
                    return value

        return ""

    @staticmethod
    def _extract_result(payload: dict[str, Any] | None) -> Any:
        """Extract tool result from event payload.

        Args:
            payload: Event payload

        Returns:
            Result or None
        """
        if not payload:
            return None

        # Try different payload formats
        for key in ("result", "response", "output", "data"):
            if key in payload:
                return payload[key]

        return None

    @staticmethod
    def _extract_execution_time(payload: dict[str, Any] | None) -> float:
        """Extract execution time from event payload.

        Args:
            payload: Event payload

        Returns:
            Execution time in milliseconds (default 0)
        """
        if not payload:
            return 0.0

        # Try different payload formats
        for key in ("execution_time_ms", "duration_ms", "elapsed_ms", "time_ms"):
            if key in payload:
                value = payload[key]
                if isinstance(value, int | float):
                    return float(value)

        return 0.0


def create_agent_with_tracing(
    agent_class: type,
    trace_id: str,
    tracer: AgentExecutionTracer | None = None,
    **agent_kwargs: Any,
) -> Any:
    """Create LlamaIndex agent with tracing enabled.

    Args:
        agent_class: LlamaIndex agent class (e.g., ReActAgent)
        trace_id: Current trace ID
        tracer: AgentExecutionTracer instance (uses global if not provided)
        **agent_kwargs: Additional agent arguments (llm, tools, system_prompt, etc.)

    Returns:
        Agent with tracing callback

    Example:
        >>> from llama_index.core.agent import ReActAgent
        >>> agent = create_agent_with_tracing(
        ...     ReActAgent,
        ...     trace_id="trace_123",
        ...     tools=tools,
        ...     llm=llm,
        ... )
    """
    if tracer is None:
        tracer = get_tracer()

    # Create callback handler
    callback_handler = AgentTracingCallback(tracer, trace_id)

    # Try to create callback manager (LlamaIndex v0.10+)
    try:
        from llama_index.core.callbacks import CallbackManager

        callback_manager = CallbackManager([callback_handler])
        agent_kwargs["callback_manager"] = callback_manager

    except ImportError:
        logger.warning(
            "CallbackManager not available. Tracing may not work with this LlamaIndex version."
        )

    # Create agent with callback
    agent = agent_class.from_tools(**agent_kwargs)

    return agent
