"""Index management tools for MongoDB collections.

This module provides tools for creating, analyzing, and managing
MongoDB indexes for performance optimization.

OBSERVABILITY: All index operations are logged before execution with full
details for verification. Note: This tool provides analysis and suggestions only,
no index creation (read-only mode).
"""

import logging
from typing import Any

from ....base_tool import BaseTool
from ....utils import handle_mongo_errors

logger = logging.getLogger(__name__)


class IndexManagerTools(BaseTool):
    """Tools for analyzing and optimizing MongoDB indexes.

    This class provides methods for examining index usage, identifying missing
    indexes, and suggesting optimizations (read-only analysis only, no creation).
    """

    def __init__(self):
        """Initialize index manager tools."""
        super().__init__()

    @handle_mongo_errors
    async def analyze_collection_indexes(
        self,
        collection_name: str,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Analyze all indexes on a collection.

        This method retrieves and analyzes all indexes defined on a collection,
        showing their definitions, options, and basic statistics.

        Args:
            collection_name: Name of the collection to analyze
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether analysis was successful
                - collection: Collection analyzed
                - total_indexes: Total number of indexes
                - indexes: List of index details
                - default_index: Info about _id index
                - custom_indexes: List of user-defined indexes
        """
        db = self.get_database()

        # Validation: Check collection exists
        collection_names = await db.list_collection_names()
        if collection_name not in collection_names:
            logger.warning(f"Collection '{collection_name}' does not exist")
            return {
                "success": False,
                "collection": collection_name,
                "error": f"Collection '{collection_name}' does not exist",
            }

        collection = db[collection_name]

        # Observability: Log analysis attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"ANALYZING COLLECTION INDEXES:\n"
            f"  Collection: {collection_name}\n"
            f"{'=' * 70}"
        )

        # Execute index analysis directly with Motor (async-native)
        try:
            indexes = await collection.list_indexes().to_list(length=None)
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return {
                "success": False,
                "collection": collection_name,
                "error": f"Failed to list indexes: {e}",
            }

        # Organize indexes
        default_index = None
        custom_indexes = []

        for idx in indexes:
            index_info = {
                "name": idx.get("name", ""),
                "keys": idx.get("key", []),
                "unique": idx.get("unique", False),
                "sparse": idx.get("sparse", False),
                "ttl": idx.get("expireAfterSeconds", None),
            }

            if idx.get("name") == "_id_":
                default_index = index_info
            else:
                custom_indexes.append(index_info)

        logger.info(f"✓ Index analysis complete: {len(indexes)} total indexes")

        return {
            "success": True,
            "collection": collection_name,
            "total_indexes": len(indexes),
            "default_index": default_index,
            "custom_indexes": custom_indexes,
            "indexes": indexes,
        }

    @handle_mongo_errors
    async def suggest_indexes(
        self,
        collection_name: str,
        query_patterns: list[dict] | None = None,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Suggest indexes for a collection based on usage patterns.

        This method analyzes common query patterns and suggests missing indexes
        that could improve query performance. Provides recommendations only
        (read-only analysis, no creation).

        Args:
            collection_name: Name of the collection
            query_patterns: Optional list of sample queries for analysis
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether analysis was successful
                - collection: Collection analyzed
                - suggested_indexes: List of suggested indexes
                - explanation: Why each index is suggested
                - estimated_impact: Performance improvement estimate
        """
        db = self.get_database()

        # Validation: Check collection exists
        collection_names = await db.list_collection_names()
        if collection_name not in collection_names:
            logger.warning(f"Collection '{collection_name}' does not exist")
            return {
                "success": False,
                "collection": collection_name,
                "error": f"Collection '{collection_name}' does not exist",
            }

        collection = db[collection_name]

        # Observability: Log suggestion attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"SUGGESTING INDEXES:\n"
            f"  Collection: {collection_name}\n"
            f"  Query Patterns: {len(query_patterns) if query_patterns else 0}\n"
            f"{'=' * 70}"
        )

        # Validation: Check query patterns format
        if query_patterns:
            if not isinstance(query_patterns, list):
                raise ValueError("Query patterns must be a list of query objects")

        # Get existing indexes directly with Motor
        try:
            indexes = await collection.list_indexes().to_list(length=None)
            indexed_fields = set()
            for idx in indexes:
                for field, _ in idx.get("key", []):
                    indexed_fields.add(field)
        except Exception as e:
            logger.error(f"Failed to fetch indexes: {e}")
            indexed_fields = set()

        # Analyze sample documents to identify common fields
        try:
            sample_docs = await collection.find().limit(100).to_list(length=100)
            common_fields = {}
            for doc in sample_docs:
                for field in doc.keys():
                    if field != "_id":
                        common_fields[field] = common_fields.get(field, 0) + 1
        except Exception as e:
            logger.error(f"Failed to analyze collection: {e}")
            common_fields = {}

        # Generate suggestions
        suggestions = []

        # Suggest indexes on frequently queried fields
        for field, count in sorted(common_fields.items(), key=lambda x: x[1], reverse=True)[:5]:
            if field not in indexed_fields and count > 50:
                suggestions.append(
                    {
                        "field": field,
                        "type": "single_field",
                        "reason": f"Appears in {count}% of documents (likely query candidate)",
                        "estimated_improvement": "Moderate",
                    }
                )

        # Suggest compound indexes for related fields
        if "patient_id" in common_fields and "status" in common_fields:
            if ("patient_id", "status") not in indexed_fields:
                suggestions.append(
                    {
                        "fields": ["patient_id", "status"],
                        "type": "compound",
                        "reason": "Common filter combination in healthcare queries",
                        "estimated_improvement": "High",
                    }
                )

        logger.info(f"✓ Index suggestions complete: {len(suggestions)} suggestions")

        return {
            "success": True,
            "collection": collection_name,
            "suggested_indexes": suggestions,
            "existing_indexed_fields": sorted(indexed_fields),
            "note": "Index suggestions are read-only analysis. Use MongoDB admin tools to create indexes.",
        }

    @handle_mongo_errors
    async def analyze_index_usage(
        self,
        collection_name: str,
        index_name: str | None = None,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Analyze index usage statistics for a collection.

        This method provides insights into how indexes are being used,
        identifying potentially unused or inefficient indexes.

        Args:
            collection_name: Name of the collection
            index_name: Optional specific index name to analyze
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether analysis was successful
                - collection: Collection analyzed
                - index_statistics: Usage statistics for indexes
                - unused_indexes: Potentially unused indexes
                - recommendations: Optimization recommendations
        """
        db = self.get_database()

        # Validation: Check collection exists
        collection_names = await db.list_collection_names()
        if collection_name not in collection_names:
            logger.warning(f"Collection '{collection_name}' does not exist")
            return {
                "success": False,
                "collection": collection_name,
                "error": f"Collection '{collection_name}' does not exist",
            }

        collection = db[collection_name]

        # Observability: Log usage analysis
        logger.info(
            f"\n{'=' * 70}\n"
            f"ANALYZING INDEX USAGE:\n"
            f"  Collection: {collection_name}\n"
            f"  Specific Index: {index_name or 'All'}\n"
            f"{'=' * 70}"
        )

        # Execute analysis directly with Motor
        try:
            # Get all indexes
            indexes = await collection.list_indexes().to_list(length=None)

            # Build index statistics
            stats = []
            for idx in indexes:
                idx_name = idx.get("name", "")

                # Skip if specific index requested and this isn't it
                if index_name and idx_name != index_name:
                    continue

                stat = {
                    "name": idx_name,
                    "keys": idx.get("key", []),
                    "size_bytes": idx.get("size", 0),
                    "sparse": idx.get("sparse", False),
                    "unique": idx.get("unique", False),
                }
                stats.append(stat)

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            stats = []

        # Analyze usage patterns
        recommendations = []

        if len(stats) > 10:
            recommendations.append(
                f"Collection has {len(stats)} indexes. Consider consolidating overlapping indexes."
            )

        # Identify potentially unused indexes
        unused = []
        for stat in stats:
            if stat["name"] != "_id_" and stat["size_bytes"] == 0:
                unused.append(stat["name"])

        logger.info(f"✓ Index usage analysis complete: {len(stats)} indexes analyzed")

        return {
            "success": True,
            "collection": collection_name,
            "index_statistics": stats,
            "unused_indexes": unused,
            "recommendations": recommendations,
            "note": "Index analysis is read-only. Use MongoDB admin tools for index maintenance.",
        }
