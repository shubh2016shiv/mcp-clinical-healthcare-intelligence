"""Core Tier 1 MongoDB MCP Tools - Essential database operations and introspection.

This module provides the foundation for all MongoDB interactions in the MCP system.
These tools are always loaded and form the essential layer that all other tools
depend upon.

Tier 1 tools focus on:
    1. Database introspection (collection and schema discovery)
    2. Query validation (syntax and safety checking)
    3. Query execution (find, aggregate, count, distinct)
    4. Result serialization (MongoDB to JSON conversion)

All tools use the @ensure_connected decorator to automatically manage database
connections and implement comprehensive error handling.

Key Features:
    - Thread-safe operations via connection pooling
    - Automatic JSON serialization of BSON types (ObjectId, datetime, etc.)
    - Query validation before execution (prevents invalid queries)
    - Read-only mode enforcement (prevents destructive operations)
    - Performance tracking (execution time monitoring)
    - Result limiting (prevents memory issues from large datasets)
    - Comprehensive logging for debugging and monitoring

Example:
    >>> from src.mcp_server.tools.core_tools import (
    ...     get_database_collections,
    ...     get_collection_schema,
    ...     execute_mongodb_query
    ... )
    >>> collections = get_database_collections()
    >>> schema = get_collection_schema("users")
    >>> results = execute_mongodb_query("users", '{"status": "active"}')
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from bson import json_util
from pymongo.errors import OperationFailure

from src.config.settings import settings
from src.mcp_server.database.connection import (
    QueryExecutionError,
    ensure_connected,
    get_connection_manager,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Serialization and Type Conversion
# ============================================================================


def serialize_mongodb_result(data: Any) -> str:
    """Serialize MongoDB query results to JSON string with BSON type support.

    This function handles serialization of MongoDB-specific types like ObjectId,
    datetime, Binary, etc., that are not natively JSON serializable. It uses
    bson.json_util to provide proper conversion while maintaining data integrity.

    Args:
        data: MongoDB result data (can be dict, list, or BSON types)

    Returns:
        JSON formatted string representation of the data with proper indentation

    Raises:
        TypeError: If data contains non-serializable types

    Example:
        >>> from bson import ObjectId
        >>> result = {
        ...     "_id": ObjectId(),
        ...     "name": "John",
        ...     "created_at": datetime.now()
        ... }
        >>> json_str = serialize_mongodb_result(result)
    """
    try:
        # json_util.default handles BSON types like ObjectId, datetime, etc.
        return json.dumps(data, default=json_util.default, indent=2)

    except TypeError as e:
        logger.error(f"Failed to serialize MongoDB result: {e}")
        raise


def deserialize_mongodb_result(json_str: str) -> Any:
    """Deserialize JSON string to MongoDB types using BSON utilities.

    Reverses the serialization process by converting JSON representations
    of BSON types back to their native Python/BSON equivalents.

    Args:
        json_str: JSON formatted string from serialize_mongodb_result()

    Returns:
        Deserialized data with BSON types restored

    Raises:
        json.JSONDecodeError: If JSON is malformed

    Example:
        >>> json_str = serialize_mongodb_result({"_id": ObjectId()})
        >>> original = deserialize_mongodb_result(json_str)
    """
    try:
        return json.loads(json_str, object_hook=json_util.object_hook)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to deserialize JSON: {e}")
        raise


# ============================================================================
# Database Introspection Tools
# ============================================================================


@ensure_connected
def get_database_collections() -> Dict[str, Any]:
    """List all available collections in the current MongoDB database.

    This tool provides database-level introspection to help the agent understand
    what data is available. It returns collection names and metadata useful for
    query planning.

    Returns:
        Dict containing:
            - success: Boolean indicating successful operation
            - collections: List of collection names in the database
            - count: Total number of collections
            - database: Name of the current database
            - error: Error message if operation failed

    Raises:
        QueryExecutionError: If database operation fails

    Example:
        >>> result = get_database_collections()
        >>> print(f"Database has {result['count']} collections")
        >>> print(f"Collections: {result['collections']}")
    """
    try:
        manager = get_connection_manager()
        db = manager.get_database()

        # Get list of all collection names in the database
        collections = db.list_collection_names()

        result = {
            "success": True,
            "collections": sorted(collections),  # Sort for consistency
            "count": len(collections),
            "database": settings.mongodb_database,
        }

        logger.info(f"Retrieved {len(collections)} collections from database")
        return result

    except QueryExecutionError as e:
        logger.error(f"Query execution error retrieving collections: {e}")
        return {
            "success": False,
            "error": str(e),
            "collections": [],
            "count": 0,
        }

    except Exception as e:
        logger.error(f"Unexpected error retrieving collections: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to retrieve collections: {str(e)}",
            "collections": [],
            "count": 0,
        }


@ensure_connected
def get_collection_schema(collection_name: str) -> Dict[str, Any]:
    """Analyze and retrieve schema information for a specific collection.

    This tool performs statistical analysis on a sample of documents to infer
    field names, data types, and provides example values. This information is
    critical for the agent to construct correct MongoDB queries.

    The analysis:
    1. Verifies the collection exists
    2. Samples up to 100 documents for analysis
    3. Analyzes nested structures (up to 3 levels deep)
    4. Collects type information and sample values
    5. Handles missing fields and null values

    Args:
        collection_name: Name of the collection to analyze

    Returns:
        Dict containing:
            - success: Boolean indicating successful operation
            - collection: Collection name being analyzed
            - fields: Dict mapping field paths to type/sample info
            - document_count: Total documents in collection
            - sample_count: Number of documents sampled for analysis
            - message: Additional info (e.g., "Collection is empty")
            - error: Error message if operation failed

    Raises:
        QueryExecutionError: If database operation fails

    Example:
        >>> schema = get_collection_schema("users")
        >>> print(f"Fields: {list(schema['fields'].keys())}")
        >>> print(f"Sample user: {schema['fields']['name']['sample']}")
    """
    try:
        manager = get_connection_manager()
        db = manager.get_database()
        collection = db[collection_name]

        # Verify collection exists
        if collection_name not in db.list_collection_names():
            logger.warning(f"Collection '{collection_name}' does not exist")
            return {
                "success": False,
                "error": f"Collection '{collection_name}' does not exist",
                "collection": collection_name,
            }

        # Get total document count
        total_count = collection.count_documents({})

        # Handle empty collection
        if total_count == 0:
            logger.info(f"Collection '{collection_name}' is empty")
            return {
                "success": True,
                "collection": collection_name,
                "fields": {},
                "document_count": 0,
                "sample_count": 0,
                "message": "Collection is empty",
            }

        # Sample documents for schema analysis (up to 100)
        sample_size = min(100, total_count)
        sample_docs = list(
            collection.aggregate([{"$sample": {"size": sample_size}}])
        )

        if not sample_docs:
            logger.warning(f"Failed to sample documents from '{collection_name}'")
            return {
                "success": True,
                "collection": collection_name,
                "fields": {},
                "document_count": total_count,
                "sample_count": 0,
                "message": "Could not sample documents",
            }

        # Analyze field types and samples
        field_info: Dict[str, Dict[str, Any]] = {}

        def analyze_value(value: Any) -> str:
            """Determine the data type of a value for schema analysis.

            Args:
                value: The value to analyze

            Returns:
                String representation of the value type
            """
            if value is None:
                return "null"
            if isinstance(value, bool):
                return "boolean"
            if isinstance(value, int):
                return "integer"
            if isinstance(value, float):
                return "number"
            if isinstance(value, str):
                return "string"
            if isinstance(value, list):
                return "array"
            if isinstance(value, dict):
                return "object"
            # Handle BSON types
            return str(type(value).__name__)

        def extract_fields(
            document: Dict[str, Any], prefix: str = "", depth: int = 0
        ) -> None:
            """Recursively extract field information from a document.

            Traverses the document structure and collects field names, types,
            and sample values. Limits depth to 3 levels to avoid overly nested
            analysis.

            Args:
                document: The document to analyze
                prefix: Current field path prefix for nested fields
                depth: Current nesting depth (max 3)
            """
            if depth > 2:  # Limit nesting depth to 3 levels
                return

            for key, value in document.items():
                # Construct full field path
                field_path = f"{prefix}.{key}" if prefix else key

                # Initialize field info if first time seeing this field
                if field_path not in field_info:
                    field_info[field_path] = {
                        "types": set(),
                        "samples": [],
                    }

                # Record the value type
                value_type = analyze_value(value)
                field_info[field_path]["types"].add(value_type)

                # Collect sample values (up to 3 different ones)
                if len(field_info[field_path]["samples"]) < 3:
                    if value not in field_info[field_path]["samples"]:
                        field_info[field_path]["samples"].append(value)

                # Recursively analyze nested objects
                if isinstance(value, dict):
                    extract_fields(value, field_path, depth + 1)

        # Extract fields from all sample documents
        for doc in sample_docs:
            extract_fields(doc)

        # Format field information for response
        formatted_fields: Dict[str, Dict[str, Any]] = {}
        for field_path, info in field_info.items():
            formatted_fields[field_path] = {
                "types": sorted(list(info["types"])),
                "sample": info["samples"][0] if info["samples"] else None,
            }

        result = {
            "success": True,
            "collection": collection_name,
            "fields": formatted_fields,
            "document_count": total_count,
            "sample_count": len(sample_docs),
        }

        logger.info(
            f"Analyzed schema for '{collection_name}': "
            f"{len(formatted_fields)} fields, {total_count} documents"
        )

        return result

    except QueryExecutionError as e:
        logger.error(f"Query execution error analyzing schema: {e}")
        return {
            "success": False,
            "error": str(e),
            "collection": collection_name,
        }

    except Exception as e:
        logger.error(
            f"Unexpected error analyzing schema for '{collection_name}': {e}",
            exc_info=True,
        )
        return {
            "success": False,
            "error": f"Failed to analyze schema: {str(e)}",
            "collection": collection_name,
        }


# ============================================================================
# Query Validation and Execution
# ============================================================================


@ensure_connected
def validate_mongodb_query(query: str, query_type: str = "find") -> Dict[str, Any]:
    """Validate MongoDB query syntax and safety before execution.

    This tool performs comprehensive validation of MongoDB queries to prevent
    execution of invalid or unsafe queries. It checks:
    1. Valid JSON syntax
    2. Appropriate query type
    3. Read-only mode restrictions
    4. Query structure correctness

    Args:
        query: MongoDB query as JSON string
        query_type: Type of query - 'find', 'aggregate', 'count', or 'distinct'

    Returns:
        Dict containing:
            - success: Boolean indicating validation passed
            - valid: Boolean indicating if query is valid
            - parsed_query: Parsed query object (if valid)
            - query_type: Confirmed query type
            - errors: List of validation errors (if invalid)
            - query: Original query string

    Example:
        >>> validation = validate_mongodb_query('{"status": "active"}', "find")
        >>> if validation['valid']:
        ...     print("Query is valid")
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
                    f"Invalid query_type '{query_type}'. "
                    f"Must be one of: {', '.join(valid_types)}"
                ],
            }

        # Type-specific validation
        if query_type == "aggregate":
            if not isinstance(parsed_query, list):
                logger.warning("Aggregation pipeline is not a list")
                return {
                    "success": False,
                    "valid": False,
                    "errors": [
                        "Aggregation pipeline must be an array/list of stages"
                    ],
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
                    f"Destructive operations detected in read-only mode: "
                    f"{found_destructive}"
                )
                return {
                    "success": False,
                    "valid": False,
                    "errors": [
                        f"Operation not allowed in read-only mode: "
                        f"{', '.join(found_destructive)}"
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


@ensure_connected
def execute_mongodb_query(
    collection_name: str,
    query: str,
    query_type: str = "find",
    projection: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a validated MongoDB query and return results with metadata.

    This is the primary tool for database queries. It executes queries after
    validation and provides comprehensive execution metadata including timing,
    result count, and limits applied.

    Supports multiple query types:
    - find: Basic document queries with optional projection
    - aggregate: Aggregation pipeline queries
    - count: Count matching documents
    - distinct: Get distinct values for a field

    Args:
        collection_name: Name of the collection to query
        query: MongoDB query as JSON string (validated first)
        query_type: Type of query ('find', 'aggregate', 'count', 'distinct')
        projection: Optional JSON string for field projection (find queries only)
        limit: Optional maximum number of results to return

    Returns:
        Dict containing:
            - success: Boolean indicating successful execution
            - collection: Collection name
            - query_type: Type of query executed
            - results: Query results (varies by query_type)
            - count: Number of results returned
            - execution_time_ms: Query execution time in milliseconds
            - limit_applied: Result limit that was applied
            - error: Error message if execution failed

    Raises:
        QueryExecutionError: If collection doesn't exist or query fails

    Example:
        >>> result = execute_mongodb_query(
        ...     "users",
        ...     '{"status": "active"}',
        ...     query_type="find",
        ...     projection='{"name": 1, "email": 1}',
        ...     limit=10
        ... )
        >>> print(f"Found {result['count']} users")
    """
    try:
        # Validate query before execution
        validation = validate_mongodb_query(query, query_type)
        if not validation.get("valid"):
            logger.warning(f"Query validation failed: {validation.get('errors')}")
            return {
                "success": False,
                "error": "Query validation failed",
                "validation_errors": validation.get("errors", []),
                "collection": collection_name,
            }

        manager = get_connection_manager()
        db = manager.get_database()

        # Verify collection exists
        if collection_name not in db.list_collection_names():
            logger.error(f"Collection '{collection_name}' does not exist")
            return {
                "success": False,
                "error": f"Collection '{collection_name}' does not exist",
                "collection": collection_name,
            }

        collection = db[collection_name]
        parsed_query = validation["parsed_query"]

        # Determine result limit
        result_limit = limit if limit is not None else settings.result_limit
        result_limit = min(result_limit, settings.result_limit)  # Never exceed max

        # Parse projection if provided (find queries only)
        projection_dict: Optional[Dict[str, Any]] = None
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

        # Execute query based on type
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

            # Serialize results to JSON
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
                f"Executed {query_type} query on '{collection_name}': "
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


if __name__ == "__main__":
    # Script: Test core tools
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        logger.info("Starting Core Tools Test Suite")
        logger.info("=" * 60)

        # Test 1: Get collections
        logger.info("\nTest 1: Retrieving database collections...")
        collections_result = get_database_collections()
        if collections_result.get("success"):
            logger.info(f"✓ Found {collections_result['count']} collections")
            logger.info(f"  Collections: {collections_result['collections']}")
        else:
            logger.warning(f"✗ Failed to get collections: {collections_result.get('error')}")

        # Test 2: Get schema for first collection (if any)
        if collections_result.get("collections"):
            collection_name = collections_result["collections"][0]
            logger.info(f"\nTest 2: Analyzing schema for '{collection_name}'...")

            schema_result = get_collection_schema(collection_name)
            if schema_result.get("success"):
                fields_count = len(schema_result.get("fields", {}))
                logger.info(f"✓ Analyzed {fields_count} fields")
                logger.info(f"  Document count: {schema_result['document_count']}")
                logger.info(f"  Sample size: {schema_result['sample_count']}")
            else:
                logger.warning(f"✗ Failed to get schema: {schema_result.get('error')}")

            # Test 3: Validate query
            logger.info("\nTest 3: Validating MongoDB query...")
            validation_result = validate_mongodb_query('{"_id": {"$exists": true}}')
            if validation_result.get("valid"):
                logger.info("✓ Query validation passed")
            else:
                logger.warning(f"✗ Query validation failed: {validation_result.get('errors')}")

            # Test 4: Execute query
            logger.info(f"\nTest 4: Executing query on '{collection_name}'...")
            exec_result = execute_mongodb_query(collection_name, "{}", limit=5)
            if exec_result.get("success"):
                logger.info(f"✓ Query executed successfully")
                logger.info(f"  Retrieved {exec_result['count']} documents")
                logger.info(f"  Execution time: {exec_result['execution_time_ms']}ms")
            else:
                logger.warning(f"✗ Query execution failed: {exec_result.get('error')}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ Core Tools Test Suite Completed")
        sys.exit(0)

    except Exception as e:
        logger.error(f"✗ Unexpected error in test suite: {e}", exc_info=True)
        sys.exit(1)
