"""Async-safe buffer for batching trace operations.

This module provides buffering utilities to reduce tracing overhead
by batching multiple operations and flushing them asynchronously.
"""

import asyncio
import logging
from typing import Any

from ..agent_execution.execution_tracer import AgentExecutionTracer, get_tracer

logger = logging.getLogger(__name__)


class AsyncTraceBuffer:
    """Async-safe buffer for batching trace operations.

    Reduces overhead by batching multiple trace operations and
    flushing them asynchronously.

    Example:
        buffer = AsyncTraceBuffer(batch_size=10)
        await buffer.add_operation("reasoning", {
            "trace_id": trace_id,
            "text": "Processing query..."
        })
        await buffer.flush()  # Manually flush if needed
    """

    def __init__(self, tracer: AgentExecutionTracer | None = None, batch_size: int = 10):
        """Initialize async trace buffer.

        Args:
            tracer: AgentExecutionTracer instance (uses global if not provided)
            batch_size: Number of operations to batch before flushing
        """
        self.tracer = tracer or get_tracer()
        self.batch_size = batch_size
        self._buffer: list[tuple[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add_operation(self, op_type: str, op_data: Any) -> None:
        """Add operation to buffer (O(1) operation).

        Args:
            op_type: Type of operation (e.g., "reasoning", "tool_call")
            op_data: Operation data

        Example:
            await buffer.add_operation("reasoning", {
                "trace_id": trace_id,
                "text": "Processing query...",
                "metadata": {}
            })
        """
        async with self._lock:
            self._buffer.append((op_type, op_data))

            # Flush if buffer is full
            if len(self._buffer) >= self.batch_size:
                await self._flush()

    async def flush(self) -> None:
        """Flush all buffered operations.

        Can be called manually to ensure operations are processed.

        Example:
            await buffer.flush()
        """
        async with self._lock:
            await self._flush()

    async def _flush(self) -> None:
        """Internal flush operation (must be called with lock held)."""
        if not self._buffer:
            return

        # Process all buffered operations
        for op_type, op_data in self._buffer:
            try:
                if op_type == "reasoning":
                    self.tracer.log_reasoning(
                        op_data["trace_id"],
                        op_data["text"],
                        op_data.get("metadata"),
                    )
                elif op_type == "tool_call":
                    self.tracer.log_tool_call(
                        op_data["trace_id"],
                        op_data["tool_name"],
                        op_data["arguments"],
                        op_data.get("metadata"),
                    )
                elif op_type == "error":
                    self.tracer.log_error(
                        op_data["trace_id"],
                        op_data["message"],
                        op_data.get("metadata"),
                    )
            except Exception as e:
                logger.error(f"Error flushing trace operation: {e}")

        self._buffer.clear()
