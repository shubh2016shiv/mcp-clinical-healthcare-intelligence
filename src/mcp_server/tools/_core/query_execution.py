"""Query execution tools for running validated MongoDB queries.

This module provides tools for executing queries with proper error handling,
logging, performance tracking, and query observability for healthcare data.
"""

import json
import logging
import time
from typing import Any

from pymongo.errors import OperationFailure

from src.mcp_server.database.connection import (
    QueryExecutionError,
    ensure_connected,
    get_connection_manager,
)

from .query_validation import validate_mongodb_query
from .result_serialization import serialize_mongodb_result

logger = logging.getLogger(__name__)


@ensure_connected
def execute_mongodb_query(
    collection_name: str,
    query: str,
    query_type: str = "find",
    projection: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Execute a validated MongoDB query on healthcare collections and return results.

    This is the PRIMARY MCP tool for healthcare database queries. It executes validated
    queries on FHIR healthcare collections and returns structured results for AI analysis.
    This function bridges natural language healthcare questions to MongoDB results.

    OBSERVABILITY: All queries are logged before execution with full details for verification.

    MCP Context: This is the core execution engine that AI assistants use to run queries
    on healthcare data. When a user asks "Show me diabetic patients", this tool executes
    the corresponding MongoDB query and returns patient data in JSON format for AI analysis.

    Healthcare Query Types Supported:
    - find: Basic patient/condition queries with optional field projection
    - aggregate: Complex healthcare analytics (grouping, counting, statistics)
    - count: Count patients with specific conditions or demographics
    - distinct: Get unique values (e.g., all diagnosis codes, medication names)

    Healthcare Safety Features:
    - Pre-validation ensures queries are safe for patient data
    - Read-only operations protect healthcare privacy
    - Result limiting prevents memory issues with large patient datasets
    - Execution time tracking for performance monitoring
    - Comprehensive error handling for healthcare data integrity
    - Query logging for observability and verification

    Args:
        collection_name: Healthcare collection name (patients, conditions, observations, medications, etc.)
        query: MongoDB query as JSON string (validated healthcare query)
        query_type: Query type ('find', 'aggregate', 'count', 'distinct')
        projection: Optional JSON for field selection (e.g., show only patient names/diagnoses)
        limit: Maximum results to return (prevents large dataset memory issues)

    Returns:
        Dict containing:
            - success: Boolean indicating successful healthcare query execution
            - collection: Healthcare collection queried
            - query_type: Type of healthcare query executed
            - results: FHIR healthcare data results (serialized to JSON)
            - count: Number of healthcare records returned
            - execution_time_ms: Query execution time for performance tracking
            - limit_applied: Result limit applied (protects against large datasets)
            - error: Error message if healthcare query failed

    Raises:
        QueryExecutionError: If healthcare collection doesn't exist or query fails

    Healthcare Examples:
        >>> # Find diabetic patients (natural language → "Show me diabetic patients")
        >>> result = execute_mongodb_query(
        ...     "conditions",
        ...     '{"code.coding.display": /Diabetes/i}',
        ...     query_type="find",
        ...     projection='{"subject.reference": 1, "code.coding.display": 1}',
        ...     limit=20
        ... )
        >>> print(f"Found {result['count']} diabetic patients")
        >>>
        >>> # Count patients by condition (analytics query)
        >>> result = execute_mongodb_query(
        ...     "conditions",
        ...     '[{"$group": {"_id": "$code.coding.display", "count": {"$sum": 1}}}]',
        ...     query_type="aggregate",
        ...     limit=10
        ... )
        >>> # Returns grouped counts of medical conditions
    """
    try:
        # STEP 1: Validate the AI-generated query before touching healthcare data
        # This ensures patient privacy and system safety - critical for healthcare!
        validation = validate_mongodb_query(query, query_type)
        if not validation.get("valid"):
            logger.warning(f"Query validation failed: {validation.get('errors')}")
            return {
                "success": False,
                "error": "Query validation failed",
                "validation_errors": validation.get("errors", []),
                "collection": collection_name,
            }

        # STEP 2: Get secure connection to healthcare database
        # Uses connection pooling for performance and health checks for reliability
        manager = get_connection_manager()
        db = manager.get_database()

        # STEP 3: Verify the healthcare collection exists
        # Ensures AI doesn't try to query non-existent collections (e.g., "patients" vs "patient")
        if collection_name not in db.list_collection_names():
            logger.error(f"Healthcare collection '{collection_name}' does not exist")
            return {
                "success": False,
                "error": f"Collection '{collection_name}' does not exist",
                "collection": collection_name,
            }

        collection = db[collection_name]
        parsed_query = validation["parsed_query"]

        # STEP 4: Apply security validation for HIPAA compliance
        # Additional security checks beyond basic query validation
        from ..security import get_security_manager

        security_manager = get_security_manager()

        # Validate query depth to prevent complex attacks
        try:
            security_manager.validator.validate_query_depth(parsed_query)
        except ValueError as e:
            logger.error(f"Query depth validation failed: {e}")
            return {
                "success": False,
                "error": f"Query too complex: {e}",
                "collection": collection_name,
            }

        # Enforce security limits on query results
        security_config = security_manager.config
        max_limit = security_config.max_query_results
        result_limit = limit if limit is not None else max_limit
        result_limit = min(result_limit, max_limit)  # Never exceed security limit

        # Additional security: Check if query contains PHI collections
        phi_collections = {"patients", "conditions", "medications", "observations", "encounters"}
        is_phi_query = collection_name in phi_collections

        if is_phi_query:
            # Log PHI access attempt (will be completed by audit decorator in actual tools)
            logger.info(f"PHI collection query: {collection_name} with limit {result_limit}")

        # STEP 5: Parse field projection (optional field selection)
        # Allows AI to request only specific fields (e.g., just patient names, not full records)
        projection_dict: dict[str, Any] | None = None
        if projection:
            try:
                projection_dict = json.loads(projection)
            except json.JSONDecodeError:
                logger.error(f"Invalid projection JSON: {projection}")
                return {
                    "success": False,
                    "error": f"Invalid projection JSON: {projection}",
                    "collection": collection_name,
                }

        # STEP 6: OBSERVABILITY - Log the query for verification before execution
        # This ensures users can verify that the correct intermediate query was generated
        logger.info(
            f"\n{'='*70}\n"
            f"EXECUTING QUERY FOR VERIFICATION:\n"
            f"  Collection: {collection_name}\n"
            f"  Query Type: {query_type}\n"
            f"  Query Filter: {json.dumps(parsed_query, indent=2)}\n"
            f"  Projection: {json.dumps(projection_dict) if projection_dict else 'None'}\n"
            f"  Limit: {result_limit}\n"
            f"{'='*70}"
        )

        # STEP 7: Execute the healthcare query with performance monitoring
        # This is where the actual database work happens - timing is critical for user experience
        start_time = time.time()

        try:
            if query_type == "find":
                # Standard find query with optional projection
                cursor = collection.find(parsed_query, projection_dict)
                results = list(cursor.limit(result_limit))
                count = len(results)

            elif query_type == "aggregate":
                # Aggregation pipeline with result limiting
                pipeline = parsed_query + [{"$limit": result_limit}]
                results = list(collection.aggregate(pipeline))
                count = len(results)

            elif query_type == "count":
                # Count matching documents
                count = collection.count_documents(parsed_query)
                results = {"count": count}

            elif query_type == "distinct":
                # Get distinct values for a field
                # Expected query: {"field": "field_name", "query": {...}}
                if not isinstance(parsed_query, dict):
                    raise ValueError("Distinct query must be a JSON object")

                field = parsed_query.get("field")
                filter_query = parsed_query.get("query", {})

                if not field:
                    raise ValueError("Distinct query must specify 'field' property")

                results = collection.distinct(field, filter_query)
                count = len(results)

            else:
                logger.error(f"Unsupported query type: {query_type}")
                return {
                    "success": False,
                    "error": f"Unsupported query type: {query_type}",
                    "collection": collection_name,
                }

            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # STEP 8: Serialize healthcare results for AI consumption
            # Convert MongoDB BSON types (ObjectId, datetime) to JSON that AI can understand
            # This is critical - AI needs clean JSON to analyze and present healthcare insights
            serialized_results = json.loads(serialize_mongodb_result(results))

            result = {
                "success": True,
                "collection": collection_name,
                "query_type": query_type,
                "results": serialized_results,
                "count": count,
                "execution_time_ms": round(execution_time, 2),
                "limit_applied": result_limit,
            }

            logger.info(
                f"✓ Query executed successfully on '{collection_name}': "
                f"{count} results in {execution_time:.2f}ms"
            )

            return result

        except OperationFailure as e:
            logger.error(f"MongoDB operation failed: {e}")
            return {
                "success": False,
                "error": f"MongoDB operation failed: {str(e)}",
                "collection": collection_name,
            }

    except QueryExecutionError as e:
        logger.error(f"Query execution error: {e}")
        return {
            "success": False,
            "error": str(e),
            "collection": collection_name,
        }

    except Exception as e:
        logger.error(f"Unexpected error executing query: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Query execution failed: {str(e)}",
            "collection": collection_name,
        }
