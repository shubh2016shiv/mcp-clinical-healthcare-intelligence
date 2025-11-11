"""Query optimization utilities for efficient data processing.

This module provides utilities for optimized data handling including:
- Batch processing of large result sets
- Streaming iteration without loading entire results into memory
- Query projection utilities for field-level data minimization
- Memory-efficient pagination support

Key Features:
    - Generator-based streaming for large datasets
    - Batch processing with configurable batch size
    - Query projection optimization
    - Memory-efficient pagination

Example:
    >>> from src.mcp_server.database.query_optimization import (
    ...     batch_cursor,
    ...     get_projection_for_fields
    ... )
    >>> # Stream results in batches
    >>> for batch in batch_cursor(cursor, batch_size=100):
    ...     process_batch(batch)
"""

import logging
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)


def batch_cursor(cursor: Any, batch_size: int = 100) -> Generator[list[dict[str, Any]], None, None]:
    """Stream cursor results in batches to avoid loading entire dataset into memory.

    This generator yields batches of documents from a MongoDB cursor without
    materializing the entire result set in memory. Useful for processing large
    datasets efficiently.

    Args:
        cursor: MongoDB cursor or iterable
        batch_size: Number of documents to yield per batch (default: 100)

    Yields:
        Lists of documents, each with maximum batch_size length

    Example:
        >>> cursor = collection.find(query)
        >>> for batch in batch_cursor(cursor, batch_size=500):
        ...     process_large_batch(batch)
    """
    batch = []
    for doc in cursor:
        batch.append(doc)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Yield remaining documents
    if batch:
        yield batch


def get_projection_for_fields(
    allowed_fields: list[str] | None = None, include_mode: bool = True
) -> dict[str, int] | None:
    """Generate MongoDB projection dictionary for field-level filtering.

    Creates a MongoDB projection to minimize data transfer and processing by
    selecting only relevant fields at the query level rather than filtering
    in application code.

    Args:
        allowed_fields: List of field names to include/exclude
        include_mode: If True, include only allowed fields; if False, exclude them

    Returns:
        MongoDB projection dictionary, or None if no filtering needed

    Example:
        >>> projection = get_projection_for_fields(
        ...     allowed_fields=['patient_id', 'name', 'birth_date'],
        ...     include_mode=True
        ... )
        >>> result = collection.find(query, projection)
    """
    if not allowed_fields:
        return None

    projection = {}
    value = 1 if include_mode else 0

    # MongoDB projection: 1 includes field, 0 excludes field
    for field in allowed_fields:
        projection[field] = value

    # Always include _id unless explicitly excluded
    if "_id" not in allowed_fields and include_mode:
        projection["_id"] = 1

    return projection if projection else None


def stream_aggregation_results(
    collection: Any, pipeline: list[dict[str, Any]], batch_size: int = 100
) -> Generator[list[dict[str, Any]], None, None]:
    """Stream aggregation results in batches to manage memory efficiently.

    Executes aggregation pipeline and streams results in batches without
    loading entire result set into memory.

    Args:
        collection: MongoDB collection object
        pipeline: Aggregation pipeline stages
        batch_size: Number of documents per batch

    Yields:
        Lists of aggregation result documents

    Example:
        >>> pipeline = [
        ...     {"$match": {"status": "active"}},
        ...     {"$group": {"_id": "$type", "count": {"$sum": 1}}}
        ... ]
        >>> for batch in stream_aggregation_results(collection, pipeline):
        ...     process_aggregated_batch(batch)
    """
    cursor = collection.aggregate(pipeline)
    yield from batch_cursor(cursor, batch_size)


def create_pagination_pipeline(skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
    """Create pagination stages for MongoDB aggregation pipeline.

    Generates skip and limit stages for cursor-based pagination in
    aggregation pipelines.

    Args:
        skip: Number of documents to skip (default: 0)
        limit: Number of documents to retrieve (default: 100)

    Returns:
        List of pipeline stages for pagination

    Example:
        >>> pipeline = [
        ...     {"$match": query},
        ...     *create_pagination_pipeline(skip=0, limit=50)
        ... ]
    """
    stages = []

    if skip > 0:
        stages.append({"$skip": skip})

    if limit > 0:
        stages.append({"$limit": limit})

    return stages


def add_field_projection_to_pipeline(
    pipeline: list[dict[str, Any]], allowed_fields: list[str] | None = None
) -> list[dict[str, Any]]:
    """Add field projection stage to aggregation pipeline.

    Adds a $project stage to aggregation pipeline to minimize data transfer
    by selecting only allowed fields.

    Args:
        pipeline: Existing aggregation pipeline
        allowed_fields: List of fields to include in projection

    Returns:
        Modified pipeline with projection stage appended

    Example:
        >>> pipeline = [{"$match": {"status": "active"}}]
        >>> fields = ['patient_id', 'name', 'date']
        >>> optimized = add_field_projection_to_pipeline(pipeline, fields)
    """
    if not allowed_fields:
        return pipeline

    # Build projection
    projection = {"_id": 1}  # Always include _id
    for field in allowed_fields:
        if field != "_id":
            projection[field] = 1

    # Create new pipeline with projection stage
    return pipeline + [{"$project": projection}]


def stream_find_results(
    cursor: Any, batch_size: int = 100, transformer: callable | None = None
) -> Generator[list[Any], None, None]:
    """Stream find() cursor results in batches with optional transformation.

    Streams MongoDB find() results in memory-efficient batches with optional
    per-document transformation.

    Args:
        cursor: MongoDB cursor from find()
        batch_size: Documents per batch (default: 100)
        transformer: Optional callable to transform each document

    Yields:
        Transformed document batches

    Example:
        >>> def transform_doc(doc):
        ...     return {
        ...         'id': str(doc['_id']),
        ...         'name': doc.get('name', '')
        ...     }
        >>> cursor = collection.find(query)
        >>> for batch in stream_find_results(cursor, batch_size=500, transformer=transform_doc):
        ...     send_batch_to_client(batch)
    """
    batch = []
    for doc in cursor:
        transformed = transformer(doc) if transformer else doc
        batch.append(transformed)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Yield remaining documents
    if batch:
        yield batch
