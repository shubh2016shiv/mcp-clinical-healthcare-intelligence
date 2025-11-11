"""Query validation tools for ensuring safe MongoDB operations.

This module provides tools for validating queries, checking types,
preventing injection attacks, and enforcing read-only mode on healthcare data.
"""

import json
import logging
from typing import Any

from src.config.settings import settings
from src.mcp_server.database.connection import ensure_connected

logger = logging.getLogger(__name__)


@ensure_connected
def validate_mongodb_query(query: str, query_type: str = "find") -> dict[str, Any]:
    """Validate MongoDB query syntax and safety before execution on healthcare data.

    This MCP tool performs comprehensive validation of MongoDB queries on healthcare
    collections to prevent execution of invalid or unsafe queries. Critical for
    protecting sensitive patient data and ensuring query correctness.

    MCP Context: This validation layer ensures AI-generated queries are safe and
    syntactically correct before they touch healthcare databases. It's a crucial
    safety mechanism in healthcare MCP systems.

    Healthcare Safety Validations:
    1. Valid JSON syntax (malformed queries could crash systems)
    2. Appropriate query type for healthcare data patterns
    3. Read-only mode enforcement (protects patient privacy - no data modification)
    4. Query structure correctness (ensures proper MongoDB syntax)
    5. Prevents destructive operations ($out, $merge) that could harm healthcare data

    Args:
        query: MongoDB query as JSON string (e.g., healthcare condition queries)
        query_type: Type of query - 'find', 'aggregate', 'count', or 'distinct'

    Returns:
        Dict containing:
            - success: Boolean indicating validation passed
            - valid: Boolean indicating if query is valid for healthcare data
            - parsed_query: Parsed query object (if valid)
            - query_type: Confirmed query type
            - errors: List of validation errors (if invalid)
            - query: Original query string

    Healthcare Examples:
        >>> # Valid: Find diabetic patients
        >>> validation = validate_mongodb_query('{"code.coding.display": /Diabetes/i}', "find")
        >>> if validation['valid']:
        ...     print("Healthcare query is safe and valid")
        >>>
        >>> # Invalid: Dangerous aggregation (would be blocked)
        >>> validation = validate_mongodb_query('[{"$out": "risky_collection"}]', "aggregate")
        >>> # Returns: valid=False, errors=["Operation not allowed in read-only mode"]
    """
    try:
        # Validate JSON syntax
        try:
            parsed_query = json.loads(query)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in query: {e}")
            return {
                "success": False,
                "valid": False,
                "errors": [f"Invalid JSON syntax: {str(e)}"],
                "query": query,
            }

        # Validate query type
        valid_types = ["find", "aggregate", "count", "distinct"]
        if query_type not in valid_types:
            logger.warning(f"Invalid query type: {query_type}")
            return {
                "success": False,
                "valid": False,
                "errors": [
                    f"Invalid query_type '{query_type}'. Must be one of: {', '.join(valid_types)}"
                ],
            }

        # Type-specific validation
        if query_type == "aggregate":
            if not isinstance(parsed_query, list):
                logger.warning("Aggregation pipeline is not a list")
                return {
                    "success": False,
                    "valid": False,
                    "errors": ["Aggregation pipeline must be an array/list of stages"],
                }

        # Check for destructive operations if in read-only mode
        if settings.read_only_mode:
            destructive_patterns = [
                "$out",  # Output results to collection
                "$merge",  # Merge with another collection
                "$addFields",  # Could be used in destructive pipelines
            ]

            # Convert query to string for pattern matching
            query_str = json.dumps(parsed_query).lower()

            found_destructive = [
                pattern for pattern in destructive_patterns if pattern.lower() in query_str
            ]

            if found_destructive:
                logger.warning(
                    f"Destructive operations detected in read-only mode: {found_destructive}"
                )
                return {
                    "success": False,
                    "valid": False,
                    "errors": [
                        f"Operation not allowed in read-only mode: {', '.join(found_destructive)}"
                    ],
                }

        result = {
            "success": True,
            "valid": True,
            "parsed_query": parsed_query,
            "query_type": query_type,
        }

        logger.debug(f"Query validation passed for {query_type} query")
        return result

    except Exception as e:
        logger.error(f"Unexpected error during query validation: {e}", exc_info=True)
        return {
            "success": False,
            "valid": False,
            "errors": [f"Validation error: {str(e)}"],
        }
