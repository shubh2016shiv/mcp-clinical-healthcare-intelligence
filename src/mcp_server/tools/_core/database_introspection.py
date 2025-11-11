"""Database introspection tools for discovering MongoDB collections and schemas.

This module provides tools for listing collections, analyzing schemas,
and discovering database structure for healthcare data.
"""

import logging
from typing import Any

from src.config.settings import settings
from src.mcp_server.database.connection import (
    QueryExecutionError,
    ensure_connected,
    get_connection_manager,
)

logger = logging.getLogger(__name__)


@ensure_connected
def get_database_collections() -> dict[str, Any]:
    """List all available healthcare collections in the MongoDB database.

    This MCP tool provides database-level introspection to help AI assistants understand
    what healthcare data collections are available. In healthcare MCP systems, this helps
    the AI discover collections like patients, conditions, observations, medications, etc.

    MCP Context: This is a discovery tool that AI assistants call first to understand
    what healthcare data sources are available before formulating queries.

    Returns:
        Dict containing:
            - success: Boolean indicating successful operation
            - collections: List of healthcare collection names (patients, conditions, observations, etc.)
            - count: Total number of healthcare collections
            - database: Name of the healthcare database
            - error: Error message if operation failed

    Raises:
        QueryExecutionError: If database operation fails

    Healthcare Example:
        >>> result = get_database_collections()
        >>> print(f"Healthcare database has {result['count']} collections")
        >>> # Output: ['patients', 'encounters', 'conditions', 'observations', 'medicationrequests', ...]
        >>> print(f"Available healthcare collections: {result['collections']}")
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
def get_collection_schema(collection_name: str) -> dict[str, Any]:
    """Analyze and retrieve schema information for a specific healthcare collection.

    This MCP tool performs statistical analysis on a sample of healthcare documents
    to infer field names, data types, and provide example values. This information
    is critical for AI assistants to construct correct MongoDB queries on healthcare data.

    MCP Context: Before an AI can query healthcare data, it needs to understand the
    structure. This tool analyzes FHIR resources to understand fields like patient
    demographics, medical codes, timestamps, and nested healthcare data structures.

    Healthcare Analysis Process:
    1. Verifies the healthcare collection exists (patients, conditions, observations, etc.)
    2. Samples up to 100 FHIR documents for analysis (respects privacy by sampling)
    3. Analyzes nested FHIR structures (up to 3 levels deep - handles complex medical data)
    4. Collects healthcare data types (patient IDs, medical codes, vital signs, etc.)
    5. Handles missing fields and null values (common in healthcare data)

    Args:
        collection_name: Name of the healthcare collection to analyze (e.g., "patients", "conditions")

    Returns:
        Dict containing:
            - success: Boolean indicating successful operation
            - collection: Healthcare collection name being analyzed
            - fields: Dict mapping FHIR field paths to type/sample info
            - document_count: Total healthcare documents in collection
            - sample_count: Number of FHIR documents sampled for analysis
            - message: Additional info (e.g., "Collection is empty")
            - error: Error message if operation failed

    Raises:
        QueryExecutionError: If database operation fails

    Healthcare Examples:
        >>> schema = get_collection_schema("patients")
        >>> print(f"Patient fields: {list(schema['fields'].keys())}")
        >>> # Output: ['id', 'name', 'birthDate', 'gender', 'address', ...]
        >>> print(f"Sample patient name: {schema['fields']['name']['sample']}")
        >>>
        >>> schema = get_collection_schema("conditions")
        >>> print(f"Medical condition fields: {list(schema['fields'].keys())}")
        >>> # Output: ['id', 'code.coding', 'subject.reference', 'onsetDateTime', ...]
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
        sample_docs = list(collection.aggregate([{"$sample": {"size": sample_size}}]))

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
        field_info: dict[str, dict[str, Any]] = {}

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

        def extract_fields(document: dict[str, Any], prefix: str = "", depth: int = 0) -> None:
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
        formatted_fields: dict[str, dict[str, Any]] = {}
        for field_path, info in field_info.items():
            formatted_fields[field_path] = {
                "types": sorted(info["types"]),
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
            f"Analyzed schema for collection '{collection_name}': "
            f"{len(formatted_fields)} fields from {len(sample_docs)} samples"
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
        logger.error(f"Unexpected error analyzing schema: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to analyze schema: {str(e)}",
            "collection": collection_name,
        }
