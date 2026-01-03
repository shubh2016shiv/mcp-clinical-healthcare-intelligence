"""Unit tests for exception hierarchy.

TESTING PHILOSOPHY:
===================
These tests verify that our exception hierarchy works correctly and provides
the expected functionality for error handling throughout the application.

Test Coverage:
--------------
1. Exception instantiation and attributes
2. Exception inheritance hierarchy
3. Error code and HTTP status code mapping
4. Exception serialization (to_dict)
5. Exception string representations
6. Exception conversion utilities

Why Test Exceptions?
--------------------
Exceptions are critical infrastructure code that:
- Must work correctly under all conditions
- Are used throughout the application
- Affect error handling and recovery strategies
- Impact API responses and user experience
- Are difficult to test manually (edge cases)
"""

import pytest
from datetime import datetime

from src.mcp_server.exceptions import (
    MCPServerError,
    DatabaseError,
    DatabaseConnectionError,
    QueryExecutionError,
    DatabaseTimeoutError,
    DatabaseIntegrityError,
    ValidationError,
    InvalidQueryError,
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ResourceError,
    ResourceNotFoundError,
    ResourceUnavailableError,
    ConfigurationError,
    convert_to_mcp_exception,
)


# =============================================================================
# BASE EXCEPTION TESTS
# =============================================================================


@pytest.mark.unit
class TestMCPServerError:
    """Test the base MCPServerError class."""
    
    def test_basic_instantiation(self):
        """Test creating a basic exception with required fields."""
        error = MCPServerError(
            message="Test error",
            error_code="TEST_ERROR"
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.http_status_code == 500  # Default
        assert isinstance(error.details, dict)
        assert isinstance(error.timestamp, str)
        assert isinstance(error.request_id, str)
    
    def test_with_details(self):
        """Test exception with additional context details."""
        error = MCPServerError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"user_id": "123", "action": "search"}
        )
        
        assert error.details["user_id"] == "123"
        assert error.details["action"] == "search"
    
    def test_with_original_exception(self):
        """Test exception chaining with original exception."""
        original = ValueError("Invalid value")
        error = MCPServerError(
            message="Wrapped error",
            error_code="WRAPPED_ERROR",
            original_exception=original
        )
        
        assert error.original_exception is original
        assert isinstance(error.original_exception, ValueError)
    
    def test_string_representation(self):
        """Test __str__ method for logging."""
        error = MCPServerError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        error_str = str(error)
        assert "TEST_ERROR" in error_str
        assert "Test error" in error_str
        assert "key" in error_str
    
    def test_repr_representation(self):
        """Test __repr__ method for debugging."""
        error = MCPServerError(
            message="Test error",
            error_code="TEST_ERROR"
        )
        
        error_repr = repr(error)
        assert "MCPServerError" in error_repr
        assert "TEST_ERROR" in error_repr
        assert "request_id" in error_repr
    
    def test_to_dict_serialization(self):
        """Test converting exception to dictionary for API responses."""
        error = MCPServerError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"field": "value"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error"] == "Test error"
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["details"]["field"] == "value"
        assert error_dict["http_status_code"] == 500
        assert "timestamp" in error_dict
        assert "request_id" in error_dict
    
    def test_to_dict_with_original_exception(self):
        """Test serialization includes original exception info."""
        original = ValueError("Original error")
        error = MCPServerError(
            message="Wrapped error",
            error_code="WRAPPED_ERROR",
            original_exception=original
        )
        
        error_dict = error.to_dict()
        
        assert "original_error" in error_dict
        assert error_dict["original_error"]["type"] == "ValueError"
        assert "Original error" in error_dict["original_error"]["message"]


# =============================================================================
# DATABASE EXCEPTION TESTS
# =============================================================================


@pytest.mark.unit
class TestDatabaseExceptions:
    """Test database-related exceptions."""
    
    def test_database_connection_error(self):
        """Test database connection failure exception."""
        error = DatabaseConnectionError(
            message="Failed to connect",
            details={"host": "localhost", "port": 27017}
        )
        
        assert error.error_code == "DB_CONNECTION_FAILED"
        assert error.http_status_code == 503
        assert isinstance(error, DatabaseError)
        assert isinstance(error, MCPServerError)
    
    def test_query_execution_error(self):
        """Test query execution failure exception."""
        error = QueryExecutionError(
            message="Query failed",
            details={"collection": "patients", "query": {}}
        )
        
        assert error.error_code == "QUERY_EXECUTION_FAILED"
        assert error.http_status_code == 500
    
    def test_database_timeout_error(self):
        """Test database timeout exception."""
        error = DatabaseTimeoutError(
            message="Operation timed out",
            details={"timeout_ms": 30000}
        )
        
        assert error.error_code == "DB_TIMEOUT"
        assert error.http_status_code == 504
    
    def test_database_integrity_error(self):
        """Test data integrity violation exception."""
        error = DatabaseIntegrityError(
            message="Duplicate key",
            details={"field": "patient_id", "value": "P123"}
        )
        
        assert error.error_code == "DB_INTEGRITY_ERROR"
        assert error.http_status_code == 409


# =============================================================================
# VALIDATION EXCEPTION TESTS
# =============================================================================


@pytest.mark.unit
class TestValidationExceptions:
    """Test validation-related exceptions."""
    
    def test_validation_error(self):
        """Test input validation failure exception."""
        error = ValidationError(
            message="Invalid input",
            details={"field": "patient_id", "error": "Required"}
        )
        
        assert error.error_code == "VALIDATION_ERROR"
        assert error.http_status_code == 400
    
    def test_invalid_query_error(self):
        """Test invalid query structure exception."""
        error = InvalidQueryError(
            message="Invalid query operators",
            details={"invalid_operators": ["$where"]}
        )
        
        assert error.error_code == "INVALID_QUERY"
        assert error.http_status_code == 400
        assert isinstance(error, ValidationError)


# =============================================================================
# SECURITY EXCEPTION TESTS
# =============================================================================


@pytest.mark.unit
class TestSecurityExceptions:
    """Test security-related exceptions."""
    
    def test_authentication_error(self):
        """Test authentication failure exception."""
        error = AuthenticationError(
            message="Authentication failed",
            details={"reason": "invalid_token"}
        )
        
        assert error.error_code == "AUTHENTICATION_FAILED"
        assert error.http_status_code == 401
        assert isinstance(error, SecurityError)
    
    def test_authorization_error(self):
        """Test authorization failure exception."""
        error = AuthorizationError(
            message="Insufficient permissions",
            details={"required": "read:patient", "has": []}
        )
        
        assert error.error_code == "AUTHORIZATION_FAILED"
        assert error.http_status_code == 403
    
    def test_rate_limit_error(self):
        """Test rate limit exceeded exception."""
        error = RateLimitError(
            message="Rate limit exceeded",
            details={"limit": 100, "window": "1 minute"}
        )
        
        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.http_status_code == 429


# =============================================================================
# RESOURCE EXCEPTION TESTS
# =============================================================================


@pytest.mark.unit
class TestResourceExceptions:
    """Test resource-related exceptions."""
    
    def test_resource_not_found_error(self):
        """Test resource not found exception."""
        error = ResourceNotFoundError(
            message="Patient not found",
            details={"patient_id": "P123"}
        )
        
        assert error.error_code == "RESOURCE_NOT_FOUND"
        assert error.http_status_code == 404
    
    def test_resource_unavailable_error(self):
        """Test resource temporarily unavailable exception."""
        error = ResourceUnavailableError(
            message="Service unavailable",
            details={"service": "rxnorm_api"}
        )
        
        assert error.error_code == "RESOURCE_UNAVAILABLE"
        assert error.http_status_code == 503


# =============================================================================
# EXCEPTION CONVERSION TESTS
# =============================================================================


@pytest.mark.unit
class TestExceptionConversion:
    """Test converting third-party exceptions to MCP exceptions."""
    
    def test_convert_mcp_exception_returns_same(self):
        """Test that MCP exceptions are returned unchanged."""
        original_error = ValidationError(
            message="Test error",
            error_code="TEST_ERROR"
        )
        
        converted = convert_to_mcp_exception(original_error)
        
        assert converted is original_error
    
    def test_convert_pymongo_connection_error(self):
        """Test converting pymongo connection errors."""
        import pymongo.errors
        
        original = pymongo.errors.ConnectionFailure("Connection failed")
        converted = convert_to_mcp_exception(
            original,
            context={"host": "localhost"}
        )
        
        assert isinstance(converted, DatabaseConnectionError)
        assert converted.original_exception is original
        assert "host" in converted.details
    
    def test_convert_pymongo_operation_error(self):
        """Test converting pymongo operation errors."""
        import pymongo.errors
        
        original = pymongo.errors.OperationFailure("Query failed")
        converted = convert_to_mcp_exception(
            original,
            context={"collection": "patients"}
        )
        
        assert isinstance(converted, QueryExecutionError)
        assert "collection" in converted.details
    
    def test_convert_pymongo_timeout_error(self):
        """Test converting pymongo timeout errors."""
        import pymongo.errors
        
        original = pymongo.errors.ExecutionTimeout("Timeout")
        converted = convert_to_mcp_exception(original)
        
        assert isinstance(converted, DatabaseTimeoutError)
    
    def test_convert_generic_exception(self):
        """Test converting unknown exceptions to generic MCP error."""
        original = RuntimeError("Unknown error")
        converted = convert_to_mcp_exception(
            original,
            default_message="Something went wrong",
            context={"operation": "test"}
        )
        
        assert isinstance(converted, MCPServerError)
        assert converted.message == "Something went wrong"
        assert converted.error_code == "INTERNAL_ERROR"
        assert "operation" in converted.details
        assert converted.original_exception is original


# =============================================================================
# EXCEPTION HIERARCHY TESTS
# =============================================================================


@pytest.mark.unit
class TestExceptionHierarchy:
    """Test that exception inheritance is correct."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from MCPServerError."""
        exception_classes = [
            DatabaseError,
            DatabaseConnectionError,
            QueryExecutionError,
            ValidationError,
            SecurityError,
            AuthenticationError,
            ResourceError,
            ConfigurationError,
        ]
        
        for exc_class in exception_classes:
            error = exc_class(message="Test", error_code="TEST")
            assert isinstance(error, MCPServerError)
            assert isinstance(error, Exception)
    
    def test_database_exception_hierarchy(self):
        """Test database exception inheritance."""
        error = DatabaseConnectionError(message="Test", error_code="TEST")
        
        assert isinstance(error, DatabaseConnectionError)
        assert isinstance(error, DatabaseError)
        assert isinstance(error, MCPServerError)
    
    def test_security_exception_hierarchy(self):
        """Test security exception inheritance."""
        error = AuthenticationError(message="Test", error_code="TEST")
        
        assert isinstance(error, AuthenticationError)
        assert isinstance(error, SecurityError)
        assert isinstance(error, MCPServerError)
