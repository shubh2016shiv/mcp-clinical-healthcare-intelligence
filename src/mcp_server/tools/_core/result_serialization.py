"""Result serialization tools for converting MongoDB results to JSON.

This module provides utilities for serializing BSON types to JSON-compatible formats,
handling pagination, and formatting results for consumption by AI assistants.
"""

import json
import logging
from typing import Any

from bson import json_util

logger = logging.getLogger(__name__)


def serialize_mongodb_result(data: Any) -> str:
    """Serialize MongoDB query results to JSON string with BSON type support.

    This function handles serialization of MongoDB-specific types like ObjectId,
    datetime, Binary, etc., that are not natively JSON serializable. It uses
    bson.json_util to provide proper conversion while maintaining data server_health_checks.

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
