"""Data models for MongoDB query traces in clinical data observability.

This module defines data structures for capturing MongoDB query execution metadata
without storing actual patient data. Designed for HIPAA compliance and query analysis.

Key Features:
- Query metadata capture (no PII)
- Execution timing and performance metrics
- Linkage to agent execution traces
- Security compliance tracking
- Query pattern analysis support
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """Supported MongoDB query types for clinical data."""

    FIND = "find"  # Standard document queries
    AGGREGATE = "aggregate"  # Aggregation pipelines
    COUNT = "count"  # Document counting
    DISTINCT = "distinct"  # Unique value extraction


@dataclass
class MongoDBQueryTrace:
    """Trace for a single MongoDB query execution in clinical data.

    Captures query metadata and execution metrics without storing actual patient data.
    Designed for HIPAA compliance - no Protected Health Information (PHI) is stored.

    Key Responsibilities:
    - Record query structure (filter, projection, limits)
    - Track execution performance (timing, success/failure)
    - Link to agent execution traces for end-to-end observability
    - Support security compliance tracking
    - Enable query pattern analysis

    HIPAA Compliance:
    - No patient names, IDs, or clinical data stored
    - Only query structure and execution metadata
    - Collection names (safe metadata)
    - Query patterns for audit purposes

    Example:
        >>> trace = MongoDBQueryTrace(
        ...     collection_name="conditions",
        ...     query_type=QueryType.FIND,
        ...     query_filter={"code.coding.display": {"$regex": "diabetes"}},
        ...     limit=20
        ... )
        >>> # Execution metrics added automatically
        >>> trace.execution_time_ms = 45.2
        >>> trace.result_count = 15
    """

    # Unique identifiers
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str | None = None  # Links to AgentExecutionTrace

    # Query structure (HIPAA safe - no PII)
    collection_name: str = ""
    query_type: QueryType = QueryType.FIND
    query_filter: dict[str, Any] = field(default_factory=dict)
    projection: dict[str, Any] | None = None
    limit: int | None = None

    # Execution metrics
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    execution_time_ms: float | None = None

    # Results metadata (no actual data stored)
    result_count: int | None = None
    success: bool = False
    error: str | None = None

    # Security and compliance
    contains_phi: bool = False  # Does collection contain PHI?
    security_checks_passed: bool = False  # Pre-execution validation
    query_validated: bool = False  # Query syntax validation

    def finalize(
        self,
        success: bool,
        result_count: int | None = None,
        execution_time_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        """Finalize the query trace with execution results.

        Args:
            success: Whether query executed successfully
            result_count: Number of documents returned (if applicable)
            execution_time_ms: Total execution time in milliseconds
            error: Error message if execution failed
        """
        self.end_time = time.time()
        self.success = success
        self.result_count = result_count
        self.execution_time_ms = execution_time_ms
        self.error = error

    def get_query_complexity_score(self) -> int:
        """Calculate query complexity score for performance analysis.

        Higher scores indicate more complex queries that may need optimization.

        Returns:
            Complexity score (0-10 scale)
        """
        score = 0

        # Collection type impact
        if self.contains_phi:
            score += 2

        # Query structure complexity
        filter_depth = self._get_dict_depth(self.query_filter)
        score += min(filter_depth, 3)

        # Aggregation pipeline complexity
        if self.query_type == QueryType.AGGREGATE:
            score += 3
            # Count pipeline stages
            if isinstance(self.query_filter, list):
                score += min(len(self.query_filter), 3)

        # Projection complexity
        if self.projection:
            score += 1

        # Result limiting (good practice)
        if self.limit and self.limit < 100:
            score -= 1

        return max(0, min(10, score))

    def _get_dict_depth(self, d: Any, current_depth: int = 0) -> int:
        """Calculate nested dictionary depth."""
        if not isinstance(d, dict):
            return current_depth

        if not d:
            return current_depth

        return (
            max(
                self._get_dict_depth(v, current_depth + 1)
                for v in d.values()
                if isinstance(v, dict | list)
            )
            or current_depth
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for serialization.

        Returns:
            Dictionary representation safe for logging/storage
        """
        return {
            "query_id": self.query_id,
            "trace_id": self.trace_id,
            "collection_name": self.collection_name,
            "query_type": self.query_type.value,
            "query_filter": self._sanitize_query_filter(self.query_filter),
            "projection": self.projection,
            "limit": self.limit,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time_ms": self.execution_time_ms,
            "result_count": self.result_count,
            "success": self.success,
            "error": self.error,
            "contains_phi": self.contains_phi,
            "security_checks_passed": self.security_checks_passed,
            "query_validated": self.query_validated,
            "complexity_score": self.get_query_complexity_score(),
        }

    def _sanitize_query_filter(self, query_filter: dict[str, Any]) -> dict[str, Any]:
        """Sanitize query filter for safe logging (remove potential PII).

        Args:
            query_filter: Original query filter

        Returns:
            Sanitized filter safe for logging
        """
        # Deep copy and sanitize
        import copy

        sanitized = copy.deepcopy(query_filter)

        # Remove or mask potentially sensitive field values
        sensitive_patterns = [
            "patient_id",
            "first_name",
            "last_name",
            "ssn",
            "address",
            "phone",
            "email",
            "birth_date",
        ]

        def _sanitize_dict(d: dict) -> dict:
            for key, value in d.items():
                if any(pattern in key.lower() for pattern in sensitive_patterns):
                    if isinstance(value, str) and len(value) > 4:
                        # Mask string values
                        d[key] = f"{value[:2]}***{value[-2:]}"
                elif isinstance(value, dict):
                    _sanitize_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            _sanitize_dict(item)
            return d

        return _sanitize_dict(sanitized)

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "SUCCESS" if self.success else f"FAILED: {self.error or 'Unknown error'}"
        return (
            f"MongoDBQueryTrace("
            f"id={self.query_id[:8]}..., "
            f"collection={self.collection_name}, "
            f"type={self.query_type.value}, "
            f"time={self.execution_time_ms:.1f}ms, "
            f"results={self.result_count}, "
            f"status={status}"
            f")"
        )
