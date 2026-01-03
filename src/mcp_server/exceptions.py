"""Enterprise-grade exception hierarchy for MCP Healthcare Server.

ARCHITECTURAL DECISION RECORD (ADR):
====================================

Problem:
--------
The current codebase uses generic Python exceptions (Exception, ValueError, etc.) which makes it:
1. Difficult to handle errors at the right granularity
2. Hard to distinguish between different failure modes
3. Challenging to provide meaningful error responses to clients
4. Impossible to implement targeted retry logic or circuit breakers

Solution:
---------
Implement a hierarchical exception system following these design principles:

1. **Single Root Exception**: All MCP exceptions inherit from MCPServerError
   - Enables catching all server-specific errors with a single except clause
   - Distinguishes our errors from third-party library errors
   - Provides consistent error structure across the application

2. **Domain-Driven Exception Categories**: Exceptions organized by failure domain
   - DatabaseError: All database-related failures (connection, query, timeout)
   - ValidationError: Input validation and data integrity issues
   - SecurityError: Authentication, authorization, and access control
   - ResourceError: Resource availability and lifecycle issues

3. **Rich Error Context**: Each exception carries structured metadata
   - error_code: Machine-readable error identifier (e.g., "DB_CONNECTION_FAILED")
   - message: Human-readable error description
   - details: Additional context (query, collection, user_id, etc.)
   - timestamp: When the error occurred
   - request_id: For distributed tracing and debugging

4. **HTTP Status Code Mapping**: Exceptions map to appropriate HTTP status codes
   - Enables consistent API responses
   - Follows REST best practices
   - Simplifies error handling in API layer

Trade-offs:
-----------
✅ Pros:
- Clear error handling strategy
- Easy to add new exception types
- Enables sophisticated error recovery (retry, circuit breaker)
- Better observability and debugging
- Consistent error responses

❌ Cons:
- More boilerplate code
- Developers must learn the exception hierarchy
- Risk of over-engineering for simple cases

Implementation Notes:
---------------------
- Exceptions are immutable (frozen dataclass) to prevent accidental modification
- All exceptions are serializable to JSON for API responses
- Error codes follow a consistent naming convention: DOMAIN_SPECIFIC_ERROR
- Exceptions include __str__ and __repr__ for debugging

Usage Example:
--------------
```python
try:
    result = await db.patients.find(query).to_list(None)
except pymongo.errors.ConnectionFailure as e:
    # Convert third-party exception to our domain exception
    raise DatabaseConnectionError(
        message="Failed to connect to MongoDB",
        details={"host": settings.mongodb_host, "error": str(e)},
        original_exception=e
    )
except pymongo.errors.OperationFailure as e:
    raise QueryExecutionError(
        message="Query execution failed",
        details={"collection": "patients", "query": query},
        original_exception=e
    )
```

Future Enhancements:
--------------------
1. Add exception metrics (count by type, error rates)
2. Implement automatic error reporting (Sentry, DataDog)
3. Add exception chaining for better root cause analysis
4. Create exception middleware for automatic logging
"""

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

# =============================================================================
# BASE EXCEPTION CLASS
# =============================================================================


@dataclass(frozen=True)
class MCPServerError(Exception):
    """Base exception for all MCP server errors.

    Design Rationale:
    -----------------
    This is the root of our exception hierarchy. All custom exceptions inherit
    from this class, enabling:

    1. **Unified Error Handling**: Catch all MCP errors with one except clause
    2. **Consistent Error Structure**: All errors have the same attributes
    3. **Separation from Library Errors**: Distinguish our errors from third-party
    4. **Rich Context**: Every error carries structured metadata

    Attributes:
    -----------
    message : str
        Human-readable error description for developers and logs
    error_code : str
        Machine-readable error identifier (e.g., "VALIDATION_FAILED")
        Used for client-side error handling and metrics
    details : dict
        Additional context about the error (query params, user_id, etc.)
        Helps with debugging and root cause analysis
    timestamp : str
        ISO 8601 timestamp when error occurred
        Essential for distributed system debugging
    request_id : str
        Unique identifier for this request/operation
        Enables tracing errors across multiple services
    http_status_code : int
        HTTP status code for API responses (400, 500, etc.)
        Enables consistent REST API error responses
    original_exception : Optional[Exception]
        The underlying exception that caused this error
        Preserves the full exception chain for debugging

    Example:
    --------
    >>> try:
    ...     raise ValueError("Invalid input")
    ... except ValueError as e:
    ...     raise MCPServerError(
    ...         message="Request validation failed",
    ...         error_code="VALIDATION_ERROR",
    ...         details={"field": "patient_id", "value": "invalid"},
    ...         original_exception=e
    ...     )
    """

    message: str
    error_code: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: str = field(default_factory=lambda: str(uuid4()))
    http_status_code: int = 500  # Default to internal server error
    original_exception: Exception | None = None

    def __str__(self) -> str:
        """Human-readable error representation for logs."""
        error_msg = f"[{self.error_code}] {self.message}"
        if self.details:
            error_msg += f" | Details: {self.details}"
        if self.original_exception:
            error_msg += (
                f" | Caused by: {type(self.original_exception).__name__}: {self.original_exception}"
            )
        return error_msg

    def __repr__(self) -> str:
        """Developer-friendly representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"error_code='{self.error_code}', "
            f"message='{self.message}', "
            f"request_id='{self.request_id}', "
            f"timestamp='{self.timestamp}'"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization.

        Design Rationale:
        -----------------
        Enables consistent error responses in API layer. The dictionary
        structure follows JSON:API error object specification.

        Returns:
        --------
        dict with keys: error, error_code, details, timestamp, request_id

        Example:
        --------
        >>> error = ValidationError(message="Invalid patient ID")
        >>> error.to_dict()
        {
            "error": "Invalid patient ID",
            "error_code": "VALIDATION_ERROR",
            "details": {},
            "timestamp": "2024-01-20T10:30:00.000Z",
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "http_status_code": 400
        }
        """
        error_dict = {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "http_status_code": self.http_status_code,
        }

        # Include original exception info if available (for debugging)
        if self.original_exception:
            error_dict["original_error"] = {
                "type": type(self.original_exception).__name__,
                "message": str(self.original_exception),
                "traceback": traceback.format_exception(
                    type(self.original_exception),
                    self.original_exception,
                    self.original_exception.__traceback__,
                ),
            }

        return error_dict


# =============================================================================
# DATABASE EXCEPTIONS
# =============================================================================
# Design Rationale: Database errors are the most common failure mode in
# data-intensive applications. We separate them by failure type to enable
# targeted recovery strategies (retry for connection errors, fail fast for
# query errors, etc.)


@dataclass(frozen=True)
class DatabaseError(MCPServerError):
    """Base class for all database-related errors.

    Design Rationale:
    -----------------
    Separates database failures from other error types, enabling:
    - Database-specific error handling (retry logic, circuit breakers)
    - Monitoring and alerting on database health
    - Graceful degradation when database is unavailable

    All database exceptions inherit from this class.
    """

    error_code: str = "DATABASE_ERROR"
    http_status_code: int = 503  # Service Unavailable


@dataclass(frozen=True)
class DatabaseConnectionError(DatabaseError):
    """Database connection failures.

    Use Case:
    ---------
    - MongoDB server unreachable
    - Network timeout
    - Authentication failure
    - Connection pool exhausted

    Recovery Strategy:
    ------------------
    - Retry with exponential backoff
    - Circuit breaker pattern
    - Fallback to cached data

    Example:
    --------
    >>> raise DatabaseConnectionError(
    ...     message="Failed to connect to MongoDB",
    ...     details={
    ...         "host": "mongodb://localhost:27017",
    ...         "timeout_ms": 5000,
    ...         "retry_attempt": 3
    ...     }
    ... )
    """

    error_code: str = "DB_CONNECTION_FAILED"
    http_status_code: int = 503


@dataclass(frozen=True)
class QueryExecutionError(DatabaseError):
    """Query execution failures.

    Use Case:
    ---------
    - Invalid query syntax
    - Query timeout
    - Index missing
    - Resource limits exceeded

    Recovery Strategy:
    ------------------
    - Log query for analysis
    - Return partial results if available
    - Suggest query optimization

    Example:
    --------
    >>> raise QueryExecutionError(
    ...     message="Query execution timed out",
    ...     details={
    ...         "collection": "patients",
    ...         "query": {"state": "CA"},
    ...         "timeout_ms": 30000,
    ...         "documents_scanned": 1000000
    ...     }
    ... )
    """

    error_code: str = "QUERY_EXECUTION_FAILED"
    http_status_code: int = 500


@dataclass(frozen=True)
class DatabaseTimeoutError(DatabaseError):
    """Database operation timeout.

    Use Case:
    ---------
    - Long-running query exceeded timeout
    - Network latency issues
    - Database overloaded

    Recovery Strategy:
    ------------------
    - Retry with smaller batch size
    - Implement query pagination
    - Add appropriate indexes

    Example:
    --------
    >>> raise DatabaseTimeoutError(
    ...     message="Database operation timed out after 30 seconds",
    ...     details={
    ...         "operation": "find",
    ...         "collection": "encounters",
    ...         "timeout_ms": 30000
    ...     }
    ... )
    """

    error_code: str = "DB_TIMEOUT"
    http_status_code: int = 504  # Gateway Timeout


@dataclass(frozen=True)
class DatabaseIntegrityError(DatabaseError):
    """Data integrity violations.

    Use Case:
    ---------
    - Duplicate key error
    - Foreign key constraint violation
    - Schema validation failure

    Recovery Strategy:
    ------------------
    - Return detailed validation errors
    - Suggest corrective action
    - Do NOT retry (data issue, not transient)

    Example:
    --------
    >>> raise DatabaseIntegrityError(
    ...     message="Duplicate patient ID",
    ...     details={
    ...         "collection": "patients",
    ...         "field": "patient_id",
    ...         "value": "P12345",
    ...         "constraint": "unique_patient_id"
    ...     }
    ... )
    """

    error_code: str = "DB_INTEGRITY_ERROR"
    http_status_code: int = 409  # Conflict


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================
# Design Rationale: Input validation errors should fail fast and provide
# clear feedback to clients. These are client errors (4xx), not server errors.


@dataclass(frozen=True)
class ValidationError(MCPServerError):
    """Input validation failures.

    Design Rationale:
    -----------------
    Separates client input errors from server errors. These are NOT bugs,
    they're expected failures that should return 400 Bad Request.

    Use Case:
    ---------
    - Invalid field values
    - Missing required fields
    - Type mismatches
    - Business rule violations

    Recovery Strategy:
    ------------------
    - Return detailed validation errors
    - Do NOT retry
    - Log for analytics (common validation failures)

    Example:
    --------
    >>> raise ValidationError(
    ...     message="Invalid patient search criteria",
    ...     details={
    ...         "errors": [
    ...             {"field": "birth_date", "error": "Invalid date format"},
    ...             {"field": "limit", "error": "Must be between 1 and 1000"}
    ...         ]
    ...     }
    ... )
    """

    error_code: str = "VALIDATION_ERROR"
    http_status_code: int = 400  # Bad Request


@dataclass(frozen=True)
class InvalidQueryError(ValidationError):
    """Invalid query parameters or structure.

    Use Case:
    ---------
    - NoSQL injection attempt detected
    - Invalid MongoDB operators
    - Malformed query syntax

    Example:
    --------
    >>> raise InvalidQueryError(
    ...     message="Query contains invalid operators",
    ...     details={
    ...         "query": {"$where": "malicious code"},
    ...         "invalid_operators": ["$where"],
    ...         "reason": "Operator not allowed for security"
    ...     }
    ... )
    """

    error_code: str = "INVALID_QUERY"
    http_status_code: int = 400


# =============================================================================
# SECURITY EXCEPTIONS
# =============================================================================
# Design Rationale: Security errors require special handling - they should
# be logged for audit, may trigger rate limiting, and should NOT expose
# sensitive information in error messages.


@dataclass(frozen=True)
class SecurityError(MCPServerError):
    """Base class for security-related errors.

    Design Rationale:
    -----------------
    Security errors are treated specially:
    1. Always logged for audit trail
    2. Error details sanitized before sending to client
    3. May trigger rate limiting or account lockout
    4. Monitored for attack patterns
    """

    error_code: str = "SECURITY_ERROR"
    http_status_code: int = 403  # Forbidden


@dataclass(frozen=True)
class AuthenticationError(SecurityError):
    """Authentication failures.

    Use Case:
    ---------
    - Invalid API key
    - Expired session token
    - Missing authentication credentials

    Security Note:
    --------------
    Error messages should be vague to prevent information disclosure.
    Do NOT reveal whether username exists, password is wrong, etc.

    Example:
    --------
    >>> raise AuthenticationError(
    ...     message="Authentication failed",  # Vague message
    ...     details={"reason": "invalid_credentials"}  # Internal only
    ... )
    """

    error_code: str = "AUTHENTICATION_FAILED"
    http_status_code: int = 401  # Unauthorized


@dataclass(frozen=True)
class AuthorizationError(SecurityError):
    """Authorization/permission failures.

    Use Case:
    ---------
    - User lacks permission for requested operation
    - Attempting to access another user's data
    - Role-based access control violation

    Example:
    --------
    >>> raise AuthorizationError(
    ...     message="Insufficient permissions to access patient data",
    ...     details={
    ...         "required_permission": "read:patient_phi",
    ...         "user_permissions": ["read:patient_basic"],
    ...         "resource": "patient:P12345"
    ...     }
    ... )
    """

    error_code: str = "AUTHORIZATION_FAILED"
    http_status_code: int = 403


@dataclass(frozen=True)
class RateLimitError(SecurityError):
    """Rate limit exceeded.

    Use Case:
    ---------
    - Too many requests from same IP
    - API quota exceeded
    - DDoS protection triggered

    Example:
    --------
    >>> raise RateLimitError(
    ...     message="Rate limit exceeded",
    ...     details={
    ...         "limit": 100,
    ...         "window": "1 minute",
    ...         "retry_after_seconds": 60
    ...     }
    ... )
    """

    error_code: str = "RATE_LIMIT_EXCEEDED"
    http_status_code: int = 429  # Too Many Requests


# =============================================================================
# RESOURCE EXCEPTIONS
# =============================================================================
# Design Rationale: Resource errors indicate issues with data availability
# or lifecycle. These help distinguish between "not found" (404) and
# "server error" (500).


@dataclass(frozen=True)
class ResourceError(MCPServerError):
    """Base class for resource-related errors."""

    error_code: str = "RESOURCE_ERROR"
    http_status_code: int = 404


@dataclass(frozen=True)
class ResourceNotFoundError(ResourceError):
    """Requested resource does not exist.

    Use Case:
    ---------
    - Patient ID not found
    - Collection doesn't exist
    - Report not available

    Example:
    --------
    >>> raise ResourceNotFoundError(
    ...     message="Patient not found",
    ...     details={
    ...         "resource_type": "patient",
    ...         "patient_id": "P12345",
    ...         "collection": "patients"
    ...     }
    ... )
    """

    error_code: str = "RESOURCE_NOT_FOUND"
    http_status_code: int = 404


@dataclass(frozen=True)
class ResourceUnavailableError(ResourceError):
    """Resource temporarily unavailable.

    Use Case:
    ---------
    - Cache miss and database down
    - External service unavailable
    - Resource locked by another process

    Example:
    --------
    >>> raise ResourceUnavailableError(
    ...     message="Drug reference database temporarily unavailable",
    ...     details={
    ...         "resource": "rxnorm_api",
    ...         "retry_after_seconds": 300
    ...     }
    ... )
    """

    error_code: str = "RESOURCE_UNAVAILABLE"
    http_status_code: int = 503


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================


@dataclass(frozen=True)
class ConfigurationError(MCPServerError):
    """Configuration or initialization errors.

    Use Case:
    ---------
    - Missing environment variables
    - Invalid configuration values
    - Initialization failures

    Design Note:
    ------------
    These should typically crash the application at startup rather than
    being caught and handled. Better to fail fast than run with bad config.

    Example:
    --------
    >>> raise ConfigurationError(
    ...     message="MongoDB connection string not configured",
    ...     details={
    ...         "env_var": "MONGODB_URI",
    ...         "current_value": None
    ...     }
    ... )
    """

    error_code: str = "CONFIGURATION_ERROR"
    http_status_code: int = 500


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def convert_to_mcp_exception(
    exception: Exception,
    default_message: str = "An unexpected error occurred",
    context: dict[str, Any] | None = None,
) -> MCPServerError:
    """Convert any exception to an appropriate MCP exception.

    Design Rationale:
    -----------------
    This function is used at API boundaries to ensure all errors are
    converted to our exception hierarchy. This enables:
    1. Consistent error responses
    2. Proper HTTP status codes
    3. Structured error logging
    4. Exception metrics

    Args:
    -----
    exception : Exception
        The original exception to convert
    default_message : str
        Fallback message if exception type is unknown
    context : dict, optional
        Additional context to include in error details

    Returns:
    --------
    MCPServerError or subclass
        Appropriate MCP exception for the given error

    Example:
    --------
    >>> try:
    ...     await db.patients.find(query).to_list(None)
    ... except Exception as e:
    ...     mcp_error = convert_to_mcp_exception(
    ...         e,
    ...         context={"collection": "patients", "query": query}
    ...     )
    ...     raise mcp_error
    """
    import pymongo.errors

    context = context or {}

    # Already an MCP exception - return as-is
    if isinstance(exception, MCPServerError):
        return exception

    # MongoDB connection errors
    if isinstance(
        exception, (pymongo.errors.ConnectionFailure, pymongo.errors.ServerSelectionTimeoutError)
    ):
        return DatabaseConnectionError(
            message="Failed to connect to database",
            details={**context, "error": str(exception)},
            original_exception=exception,
        )

    # MongoDB timeout errors (check BEFORE OperationFailure since ExecutionTimeout inherits from it)
    if isinstance(exception, pymongo.errors.ExecutionTimeout):
        return DatabaseTimeoutError(
            message="Database operation timed out",
            details={**context, "error": str(exception)},
            original_exception=exception,
        )

    # MongoDB operation errors (check AFTER ExecutionTimeout)
    if isinstance(exception, pymongo.errors.OperationFailure):
        return QueryExecutionError(
            message="Database query failed",
            details={**context, "error": str(exception)},
            original_exception=exception,
        )

    # Pydantic validation errors
    if exception.__class__.__name__ == "ValidationError":
        return ValidationError(
            message="Request validation failed",
            details={**context, "validation_errors": str(exception)},
            original_exception=exception,
        )

    # Generic fallback
    return MCPServerError(
        message=default_message,
        error_code="INTERNAL_ERROR",
        details={**context, "error_type": type(exception).__name__, "error": str(exception)},
        original_exception=exception,
    )
