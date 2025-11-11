"""Shared utilities for MCP healthcare tools.

This module provides common functionality used across multiple MCP tools:
- Error handling decorators
- MongoDB document serialization
- Date filtering utilities
- Query building helpers

Key Features:
    - Decorators for consistent error handling
    - MongoDB document to JSON serialization
    - Date range filtering helpers
    - Type-safe query building
"""

import logging
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from bson import ObjectId
from pymongo.errors import OperationFailure, PyMongoError

from ..security import get_security_manager
from .models import ErrorResponse

logger = logging.getLogger(__name__)


def handle_mongo_errors(func: Callable) -> Callable:
    """Decorator for consistent error handling across MongoDB operations.

    This decorator wraps MongoDB operations with comprehensive error handling,
    providing consistent error responses and logging.

    Args:
        func: The function to decorate

    Returns:
        Wrapped function with error handling

    Example:
        @handle_mongo_errors
        async def query_patients(self, request):
            # MongoDB operations here
            return result
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed in {func.__name__}: {e.code} {e.details}")
            return ErrorResponse(
                error="Database operation failed",
                details=f"Operation failed with code {e.code}: {e.details}",
                operation=func.__name__,
            )
        except PyMongoError as e:
            logger.error(f"MongoDB error in {func.__name__}: {e}")
            return ErrorResponse(
                error="Database error occurred", details=str(e), operation=func.__name__
            )
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return ErrorResponse(
                error="Unexpected error occurred", details=str(e), operation=func.__name__
            )

    return wrapper


def serialize_mongo_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert MongoDB document to JSON-serializable format.

    MongoDB documents contain special types (ObjectId, datetime) that aren't
    JSON serializable. This function handles conversion for MCP responses.

    Args:
        doc: MongoDB document dictionary

    Returns:
        JSON-serializable dictionary

    Example:
        >>> doc = {"_id": ObjectId(), "date": datetime.now()}
        >>> serialized = serialize_mongo_doc(doc)
        >>> # Now safe for JSON serialization
    """
    if doc is None:
        return {}

    serialized = {}
    for key, value in doc.items():
        if key == "_id":
            # Convert ObjectId to string
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            # Convert datetime to ISO format string
            serialized[key] = value.isoformat()
        elif isinstance(value, ObjectId):
            # Convert any other ObjectIds to strings
            serialized[key] = str(value)
        elif isinstance(value, dict):
            # Recursively serialize nested documents
            serialized[key] = serialize_mongo_doc(value)
        elif isinstance(value, list):
            # Handle arrays with potential nested documents
            serialized[key] = [
                serialize_mongo_doc(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            serialized[key] = value

    return serialized


def serialize_mongo_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert multiple MongoDB documents to JSON-serializable format.

    Args:
        docs: List of MongoDB document dictionaries

    Returns:
        List of JSON-serializable dictionaries
    """
    return [serialize_mongo_doc(doc) for doc in docs]


def build_date_filter(
    field_name: str, start_date: str | None, end_date: str | None
) -> dict[str, Any]:
    """Build MongoDB date range filter with proper ISO date handling.

    This utility handles date parsing and creates MongoDB-compatible date filters.

    Args:
        field_name: The field name to filter on
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)

    Returns:
        MongoDB filter dictionary for date range

    Example:
        >>> filter = build_date_filter("birth_date", "1990-01-01", "2000-12-31")
        >>> # Returns: {"birth_date": {"$gte": datetime(1990, 1, 1), "$lte": datetime(2000, 12, 31)}}
    """
    filter_dict = {}

    if start_date:
        try:
            # Parse date string and create datetime at start of day
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            filter_dict["$gte"] = start_dt
        except ValueError as e:
            logger.warning(f"Invalid start_date format: {start_date} - {e}")

    if end_date:
        try:
            # Parse date string and create datetime at end of day
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            # Set to end of day for inclusive range
            end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            filter_dict["$lte"] = end_dt
        except ValueError as e:
            logger.warning(f"Invalid end_date format: {end_date} - {e}")

    return {field_name: filter_dict} if filter_dict else {}


def build_text_filter(
    field_name: str, search_text: str | None, case_insensitive: bool = True
) -> dict[str, Any]:
    """Build MongoDB text search filter with regex support.

    Args:
        field_name: The field name to search in
        search_text: Text to search for (partial match)
        case_insensitive: Whether to perform case-insensitive search

    Returns:
        MongoDB filter dictionary for text search

    Example:
        >>> filter = build_text_filter("first_name", "john")
        >>> # Returns: {"first_name": {"$regex": "john", "$options": "i"}}
    """
    if not search_text:
        return {}

    options = "i" if case_insensitive else ""
    return {field_name: {"$regex": search_text, "$options": options}}


def validate_patient_id(patient_id: str) -> bool:
    """Validate patient ID format.

    Args:
        patient_id: Patient ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not patient_id or not isinstance(patient_id, str):
        return False

    # Patient IDs should be non-empty strings
    # Add more specific validation as needed based on your ID format
    return len(patient_id.strip()) > 0


def create_pagination_filter(skip: int = 0, limit: int = 100) -> tuple:
    """Create pagination parameters for MongoDB queries.

    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return

    Returns:
        Tuple of (skip, limit) for MongoDB find() method
    """
    # Ensure reasonable limits
    safe_limit = max(1, min(limit, 1000))  # Max 1000 results
    safe_skip = max(0, skip)

    return safe_skip, safe_limit


def build_compound_filter(*filters: dict[str, Any]) -> dict[str, Any]:
    """Combine multiple MongoDB filters into a compound filter with security validation.

    This utility merges multiple filter dictionaries, handling nested structures.
    Includes security validation to prevent NoSQL injection attacks.

    Rationale: Query filters are a common attack vector for NoSQL injection.
    All filters must be validated before execution to ensure they contain
    only safe, expected field names and operators.

    Args:
        *filters: Variable number of filter dictionaries

    Returns:
        Combined and validated filter dictionary

    Raises:
        ValueError: If filter validation fails

    Example:
        >>> filter1 = {"status": "active"}
        >>> filter2 = {"patient_id": "123"}
        >>> combined = build_compound_filter(filter1, filter2)
        >>> # Returns: {"status": "active", "patient_id": "123"}
    """
    combined = {}
    for filter_dict in filters:
        if filter_dict:  # Skip empty filters
            # Validate filter for security
            try:
                security_manager = get_security_manager()
                validated_filter = security_manager.validator.validate_query_params(filter_dict)
                combined.update(validated_filter)
            except Exception as e:
                logger.error(f"Query validation failed: {e}")
                raise ValueError(f"Invalid query filter: {e}") from e

    return combined


def safe_get_nested_value(doc: dict[str, Any], path: str, default=None) -> Any:
    """Safely get nested value from document using dot notation.

    Args:
        doc: Document to extract value from
        path: Dot-separated path (e.g., "address.city")
        default: Default value if path doesn't exist

    Returns:
        Value at path or default

    Example:
        >>> doc = {"address": {"city": "Austin"}}
        >>> city = safe_get_nested_value(doc, "address.city", "Unknown")
        >>> # Returns: "Austin"
    """
    try:
        keys = path.split(".")
        value = doc
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
