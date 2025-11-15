"""Context manager for MongoDB query tracing in clinical data observability.

This module provides a context manager that automatically traces MongoDB queries
executed on clinical data collections. It ensures proper setup and cleanup of
query traces with minimal code changes.

Key Features:
- Automatic trace ID discovery from agent execution context
- Minimal overhead context manager pattern
- HIPAA compliance (no PHI storage)
- Error handling and cleanup
- Integration with security validation

Usage Patterns:
1. Wrap entire query execution for automatic tracing
2. Manual result setting for precise metrics
3. Exception handling for failed queries
"""

import logging
import time
from typing import Any

from .query_models import MongoDBQueryTrace, QueryType
from .query_tracer import get_query_tracer

# Import from integrations for trace ID discovery
try:
    from ..integrations.context_managers import ExecutionContext
except ImportError:
    # Fallback for when integrations not available
    ExecutionContext = None

logger = logging.getLogger(__name__)


class MongoDBQueryContext:
    """Context manager for tracing MongoDB queries on clinical data.

    Automatically captures MongoDB query execution with minimal overhead.
    Designed for HIPAA compliance - only captures query metadata, not patient data.

    Key Responsibilities:
    - Start and finalize query traces automatically
    - Discover trace_id from agent execution context
    - Capture execution metrics and errors
    - Ensure proper cleanup even on exceptions
    - Support security compliance tracking

    Usage:
        # Automatic result counting
        with MongoDBQueryContext("patients", {"age": {"$gt": 65}}) as ctx:
            results = await collection.find(query).to_list(100)
            ctx.set_result(len(results))

        # Manual result setting
        with MongoDBQueryContext("conditions", query_dict) as ctx:
            count = await collection.count_documents(query_dict)
            ctx.set_result(count)

    Integration:
        - Automatically links to agent execution traces
        - Captures security validation status
        - Tracks query performance metrics
        - Supports audit trail generation
    """

    def __init__(
        self,
        collection_name: str,
        query_filter: dict[str, Any],
        query_type: QueryType = QueryType.FIND,
        projection: dict[str, Any] | None = None,
        limit: int | None = None,
        security_validated: bool = False,
    ):
        """Initialize MongoDB query context.

        Args:
            collection_name: MongoDB collection name (e.g., 'patients', 'conditions')
            query_filter: Query filter dictionary (HIPAA safe)
            query_type: Type of MongoDB query operation
            projection: Optional field projection specification
            limit: Optional result limit
            security_validated: Whether query passed security validation

        Example:
            >>> ctx = MongoDBQueryContext(
            ...     collection_name="conditions",
            ...     query_filter={"code.coding.display": {"$regex": "diabetes"}},
            ...     query_type=QueryType.FIND,
            ...     limit=20,
            ...     security_validated=True
            ... )
        """
        self.collection_name = collection_name
        self.query_filter = query_filter
        self.query_type = query_type
        self.projection = projection
        self.limit = limit
        self.security_validated = security_validated

        self.query_trace: MongoDBQueryTrace | None = None
        self.tracer = get_query_tracer()

        # State tracking
        self._entered = False
        self._result_count: int | None = None
        self._start_time: float | None = None

    def __enter__(self) -> "MongoDBQueryContext":
        """Enter the query tracing context.

        Starts query trace and sets up automatic cleanup.

        Returns:
            Self for method chaining

        Example:
            >>> with MongoDBQueryContext("patients", query) as ctx:
            ...     # Query execution here
            ...     pass
        """
        if self._entered:
            raise RuntimeError("Cannot re-enter MongoDBQueryContext")

        self._entered = True
        self._start_time = time.time()

        # Discover trace_id from agent execution context
        trace_id = None
        if ExecutionContext:
            try:
                trace_id = ExecutionContext.get_current_trace_id()
            except Exception:
                # Gracefully handle missing context
                pass

        # Start query trace
        self.query_trace = self.tracer.start_query_trace(
            trace_id=trace_id,
            collection_name=self.collection_name,
            query_filter=self.query_filter,
            query_type=self.query_type,
            projection=self.projection,
            limit=self.limit,
        )

        # Mark security validation status
        if self.query_trace:
            self.query_trace.security_checks_passed = self.security_validated
            self.query_trace.query_validated = True

        logger.debug(
            f"Entered MongoDB query context: {self.query_trace.query_id[:8] if self.query_trace else 'unknown'} "
            f"on {self.collection_name}"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the query tracing context.

        Automatically finalizes the query trace with execution results.
        Handles both successful and failed query executions.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised

        Example:
            # Automatic cleanup on normal exit
            with MongoDBQueryContext("patients", query) as ctx:
                results = await collection.find(query).to_list(10)
                ctx.set_result(len(results))
            # Trace automatically finalized

            # Automatic cleanup on exception
            with MongoDBQueryContext("patients", query) as ctx:
                raise ValueError("Query failed")
            # Error automatically captured in trace
        """
        if not self._entered or not self.query_trace:
            return

        # Calculate execution time
        execution_time_ms = None
        if self._start_time:
            execution_time_ms = (time.time() - self._start_time) * 1000

        # Handle execution results
        if exc_type is not None:
            # Query failed with exception
            error_msg = f"{exc_type.__name__}: {str(exc_val) if exc_val else 'Unknown error'}"
            self.tracer.finalize_query_trace(
                query_id=self.query_trace.query_id,
                success=False,
                execution_time_ms=execution_time_ms,
                error=error_msg,
            )
            logger.warning(f"MongoDB query failed: {self.query_trace.query_id[:8]} - {error_msg}")
        else:
            # Query completed successfully
            self.tracer.finalize_query_trace(
                query_id=self.query_trace.query_id,
                success=True,
                result_count=self._result_count,
                execution_time_ms=execution_time_ms,
            )
            logger.debug(
                f"MongoDB query completed: {self.query_trace.query_id[:8]} "
                f"({self._result_count} results, {execution_time_ms:.1f}ms)"
            )

        self._entered = False

    def set_result(self, count: int) -> None:
        """Set the result count for the query.

        Call this method after executing the query to record how many
        documents were returned or affected.

        Args:
            count: Number of documents returned/affected by the query

        Example:
            >>> with MongoDBQueryContext("patients", query) as ctx:
            ...     results = await collection.find(query).to_list(50)
            ...     ctx.set_result(len(results))  # Records 50 results

            >>> with MongoDBQueryContext("conditions", query) as ctx:
            ...     count = await collection.count_documents(query)
            ...     ctx.set_result(count)  # Records count result
        """
        if not self._entered:
            raise RuntimeError("Cannot set result outside of context")

        self._result_count = count
        logger.debug(
            f"Set result count: {count} for query {self.query_trace.query_id[:8] if self.query_trace else 'unknown'}"
        )

    def set_security_status(self, validated: bool, checks_passed: bool = True) -> None:
        """Update security validation status during query execution.

        Args:
            validated: Whether query structure was validated
            checks_passed: Whether security checks passed

        Example:
            >>> with MongoDBQueryContext("patients", query) as ctx:
            ...     # After security validation
            ...     ctx.set_security_status(validated=True, checks_passed=True)
        """
        if self.query_trace:
            self.query_trace.query_validated = validated
            self.query_trace.security_checks_passed = checks_passed

    @property
    def query_id(self) -> str | None:
        """Get the current query trace ID.

        Returns:
            Query trace ID if context is active, None otherwise
        """
        return self.query_trace.query_id if self.query_trace else None

    @property
    def trace_id(self) -> str | None:
        """Get the linked agent execution trace ID.

        Returns:
            Agent trace ID if linked, None otherwise
        """
        return self.query_trace.trace_id if self.query_trace else None


class ClinicalQueryManager:
    """Manager for clinical data queries with tracing and compliance.

    Provides high-level utilities for managing clinical data queries
    with automatic tracing, security compliance, and audit trails.

    Key Features:
    - Batch query tracing
    - Security compliance checking
    - Performance monitoring
    - Audit trail generation

    Example:
        >>> manager = ClinicalQueryManager()
        >>> with manager.trace_query("patients", patient_query) as ctx:
        ...     results = await collection.find(patient_query).to_list(10)
        ...     ctx.set_result(len(results))
    """

    def __init__(self):
        """Initialize clinical query manager."""
        self.tracer = get_query_tracer()

    def trace_query(
        self,
        collection_name: str,
        query_filter: dict[str, Any],
        query_type: QueryType = QueryType.FIND,
        projection: dict[str, Any] | None = None,
        limit: int | None = None,
        security_validated: bool = False,
    ) -> MongoDBQueryContext:
        """Create a traced query context for clinical data.

        Convenience method that pre-configures tracing for clinical queries.

        Args:
            collection_name: Clinical collection name
            query_filter: Query filter (HIPAA safe)
            query_type: Query operation type
            projection: Field projection
            limit: Result limit
            security_validated: Security validation status

        Returns:
            Configured MongoDBQueryContext

        Example:
            >>> manager = ClinicalQueryManager()
            >>> with manager.trace_query("conditions", diabetes_query) as ctx:
            ...     results = await execute_clinical_query()
            ...     ctx.set_result(len(results))
        """
        return MongoDBQueryContext(
            collection_name=collection_name,
            query_filter=query_filter,
            query_type=query_type,
            projection=projection,
            limit=limit,
            security_validated=security_validated,
        )

    def get_query_performance(self, trace_id: str | None = None) -> dict[str, Any]:
        """Get performance statistics for clinical queries.

        Args:
            trace_id: Optional agent trace ID filter

        Returns:
            Performance report for clinical queries
        """
        return self.tracer.get_performance_report(trace_id)

    def get_security_summary(self, trace_id: str | None = None) -> dict[str, Any]:
        """Get security compliance summary for clinical queries.

        Args:
            trace_id: Optional agent trace ID filter

        Returns:
            Security compliance report
        """
        queries = []
        if trace_id:
            queries = self.tracer.get_trace_queries(trace_id)
        else:
            for query_list in self.tracer._trace_queries.values():
                queries.extend(query_list)

        if not queries:
            return {"message": "No clinical queries found"}

        phi_queries = [q for q in queries if q.contains_phi]
        validated_queries = [q for q in queries if q.query_validated]
        security_passed = [q for q in queries if q.security_checks_passed]

        return {
            "total_queries": len(queries),
            "phi_queries": len(phi_queries),
            "validated_queries": len(validated_queries),
            "security_passed": len(security_passed),
            "validation_rate": round(len(validated_queries) / max(1, len(queries)) * 100, 2),
            "security_pass_rate": round(len(security_passed) / max(1, len(queries)) * 100, 2),
        }
