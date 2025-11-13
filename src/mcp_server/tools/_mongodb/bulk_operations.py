"""Bulk operation tools for batch MongoDB operations.

This module provides tools for performing bulk read operations
with read-only enforcement for healthcare data.

OBSERVABILITY: All bulk operations are logged with full details including
batch counts, result limits, and execution metrics.
"""

import logging
from typing import Any

from ....base_tool import BaseTool
from ....utils import handle_mongo_errors

logger = logging.getLogger(__name__)

# Safety limits for bulk operations
MAX_BATCH_SIZE = 10000
MAX_QUERIES_PER_BATCH = 100
MAX_PIPELINES_PER_BATCH = 50


class BulkOperationsTools(BaseTool):
    """Tools for efficient batch read operations on large datasets.

    This class provides methods for executing multiple queries or pipelines
    in batch with safety limits and performance optimization.
    """

    def __init__(self):
        """Initialize bulk operations tools."""
        super().__init__()

    @handle_mongo_errors
    async def bulk_find(
        self,
        collection_name: str,
        queries: list[dict],
        limit_per_query: int = 100,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Execute multiple find queries in batch.

        This method efficiently executes multiple queries against the same
        collection with per-query limits for safety.

        Args:
            collection_name: Name of the collection to query
            queries: List of query filters (MongoDB find() queries)
            limit_per_query: Maximum results per query (default: 100)
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether all queries executed successfully
                - collection: Collection queried
                - total_queries: Total number of queries executed
                - results: List of results grouped by query
                - total_results: Total results across all queries
                - execution_stats: Performance statistics
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

        # Validation: Check input types
        if not isinstance(queries, list):
            raise ValueError("Queries must be a list")

        if len(queries) == 0:
            raise ValueError("Queries list cannot be empty")

        if len(queries) > MAX_QUERIES_PER_BATCH:
            raise ValueError(f"Too many queries: {len(queries)} (max: {MAX_QUERIES_PER_BATCH})")

        if not isinstance(limit_per_query, int) or limit_per_query <= 0:
            raise ValueError("Limit per query must be a positive integer")

        if limit_per_query > MAX_BATCH_SIZE:
            limit_per_query = MAX_BATCH_SIZE

        collection = db[collection_name]

        # Observability: Log batch execution
        logger.info(
            f"\n{'=' * 70}\n"
            f"EXECUTING BULK FIND QUERIES:\n"
            f"  Collection: {collection_name}\n"
            f"  Total Queries: {len(queries)}\n"
            f"  Limit per Query: {limit_per_query}\n"
            f"{'=' * 70}"
        )

        # Execute queries directly with Motor (async-native)
        results = []
        for query in queries:
            try:
                if not isinstance(query, dict):
                    results.append(
                        {
                            "success": False,
                            "error": f"Query must be dict, got {type(query).__name__}",
                            "results": [],
                        }
                    )
                    continue

                cursor = collection.find(query).limit(limit_per_query)
                docs = await cursor.to_list(length=limit_per_query)
                results.append(
                    {
                        "success": True,
                        "query": query,
                        "results": docs,
                        "count": len(docs),
                    }
                )
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                results.append({"success": False, "query": query, "error": str(e), "results": []})

        # Aggregate statistics
        total_results = sum(r.get("count", 0) for r in results if r.get("success"))
        successful_queries = sum(1 for r in results if r.get("success"))

        logger.info(
            f"✓ Bulk find complete: {successful_queries}/{len(queries)} queries successful, "
            f"{total_results} total results"
        )

        return {
            "success": successful_queries == len(queries),
            "collection": collection_name,
            "total_queries": len(queries),
            "successful_queries": successful_queries,
            "results": results,
            "total_results": total_results,
            "execution_stats": {
                "queries_executed": len(queries),
                "total_documents_returned": total_results,
                "limit_per_query": limit_per_query,
            },
        }

    @handle_mongo_errors
    async def bulk_aggregate(
        self,
        collection_name: str,
        pipelines: list[list[dict]],
        limit_per_pipeline: int = 100,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Execute multiple aggregation pipelines in batch.

        This method executes multiple aggregation pipelines against the same
        collection with per-pipeline result limits.

        Args:
            collection_name: Name of the collection to query
            pipelines: List of aggregation pipelines (each is a list of stages)
            limit_per_pipeline: Maximum results per pipeline (default: 100)
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether all pipelines executed successfully
                - collection: Collection queried
                - total_pipelines: Total number of pipelines
                - results: List of results grouped by pipeline
                - total_results: Total results across all pipelines
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

        # Validation: Check input types
        if not isinstance(pipelines, list):
            raise ValueError("Pipelines must be a list")

        if len(pipelines) == 0:
            raise ValueError("Pipelines list cannot be empty")

        if len(pipelines) > MAX_PIPELINES_PER_BATCH:
            raise ValueError(
                f"Too many pipelines: {len(pipelines)} (max: {MAX_PIPELINES_PER_BATCH})"
            )

        if not isinstance(limit_per_pipeline, int) or limit_per_pipeline <= 0:
            raise ValueError("Limit per pipeline must be a positive integer")

        if limit_per_pipeline > MAX_BATCH_SIZE:
            limit_per_pipeline = MAX_BATCH_SIZE

        collection = db[collection_name]

        # Observability: Log batch execution
        logger.info(
            f"\n{'=' * 70}\n"
            f"EXECUTING BULK AGGREGATION PIPELINES:\n"
            f"  Collection: {collection_name}\n"
            f"  Total Pipelines: {len(pipelines)}\n"
            f"  Limit per Pipeline: {limit_per_pipeline}\n"
            f"{'=' * 70}"
        )

        # Execute pipelines directly with Motor (async-native)
        results = []
        for pipeline in pipelines:
            try:
                if not isinstance(pipeline, list):
                    results.append(
                        {
                            "success": False,
                            "error": f"Pipeline must be list, got {type(pipeline).__name__}",
                            "results": [],
                        }
                    )
                    continue

                # Add limit stage for safety
                pipeline_with_limit = pipeline + [{"$limit": limit_per_pipeline}]
                cursor = collection.aggregate(pipeline_with_limit)
                docs = await cursor.to_list(length=limit_per_pipeline)
                results.append(
                    {
                        "success": True,
                        "pipeline_stages": len(pipeline),
                        "results": docs,
                        "count": len(docs),
                    }
                )
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                results.append({"success": False, "error": str(e), "results": []})

        # Aggregate statistics
        total_results = sum(r.get("count", 0) for r in results if r.get("success"))
        successful_pipelines = sum(1 for r in results if r.get("success"))

        logger.info(
            f"✓ Bulk aggregate complete: {successful_pipelines}/{len(pipelines)} pipelines successful, "
            f"{total_results} total results"
        )

        return {
            "success": successful_pipelines == len(pipelines),
            "collection": collection_name,
            "total_pipelines": len(pipelines),
            "successful_pipelines": successful_pipelines,
            "results": results,
            "total_results": total_results,
        }

    @handle_mongo_errors
    async def batch_read(
        self,
        collection_name: str,
        filter_query: dict,
        batch_size: int = 1000,
        limit: int = 10000,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Read large collections in batches using cursor batching.

        This method efficiently reads large collections by fetching data
        in batches to minimize memory usage and improve performance.

        Args:
            collection_name: Name of the collection to read
            filter_query: MongoDB filter query
            batch_size: Documents per batch (default: 1000)
            limit: Maximum total documents to return (default: 10000)
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether read was successful
                - collection: Collection read
                - total_documents: Total documents returned
                - batch_count: Number of batches used
                - batches: List of batch results
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

        # Validation: Check input types and values
        if not isinstance(filter_query, dict):
            raise ValueError("Filter query must be a dictionary")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")

        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Limit must be a positive integer")

        # Apply safety limits
        batch_size = min(batch_size, MAX_BATCH_SIZE)
        limit = min(limit, MAX_BATCH_SIZE * 10)  # Allow higher overall limit for batch reads

        collection = db[collection_name]

        # Observability: Log batch read
        logger.info(
            f"\n{'=' * 70}\n"
            f"EXECUTING BATCH READ:\n"
            f"  Collection: {collection_name}\n"
            f"  Filter: {filter_query}\n"
            f"  Batch Size: {batch_size}\n"
            f"  Total Limit: {limit}\n"
            f"{'=' * 70}"
        )

        # Execute batch read directly with Motor (async-native)
        batches = []
        cursor = collection.find(filter_query).batch_size(batch_size)

        total_fetched = 0
        batch_num = 0

        try:
            async for doc in cursor:
                if total_fetched >= limit:
                    break

                if len(batches) == 0 or len(batches[-1]["documents"]) >= batch_size:
                    batches.append(
                        {
                            "batch_number": batch_num,
                            "documents": [],
                            "count": 0,
                        }
                    )
                    batch_num += 1

                batches[-1]["documents"].append(doc)
                batches[-1]["count"] += 1
                total_fetched += 1

        except Exception as e:
            logger.error(f"Batch read failed: {e}")
            return {
                "success": False,
                "error": f"Batch read failed: {e}",
                "batches": [],
                "total_documents": 0,
            }

        logger.info(f"✓ Batch read complete: {len(batches)} batches, {total_fetched} documents")

        return {
            "success": True,
            "collection": collection_name,
            "total_documents": total_fetched,
            "batch_count": len(batches),
            "batches": batches,
            "execution_stats": {
                "batch_size": batch_size,
                "total_limit": limit,
                "documents_fetched": total_fetched,
            },
        }
