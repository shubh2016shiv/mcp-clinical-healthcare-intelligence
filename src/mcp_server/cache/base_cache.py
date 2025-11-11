"""Base cache class with common Redis operations and error handling.

This module provides the foundation for all specialized cache implementations,
including connection management, error handling, and graceful fallback to
direct operations when Redis is unavailable.
"""

import json
import logging
from typing import Any, TypeVar

try:
    import redis
    from redis.exceptions import RedisError

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseCacheError(Exception):
    """Custom exception for cache operation errors."""

    pass


class BaseCache:
    """Abstract base class for Redis-backed cache implementations.

    Provides common functionality for all cache types including:
    - Redis connection management
    - Error handling with graceful fallback
    - JSON serialization/deserialization
    - TTL management
    - Logging and monitoring

    Rationale:
    - Centralized error handling prevents cascading failures
    - Graceful fallback ensures system reliability even if Redis fails
    - Consistent interface across all cache types
    """

    def __init__(self, redis_client: Any | None = None):
        """Initialize base cache with optional Redis client.

        Args:
            redis_client: Optional Redis client. If None, cache will be disabled.
        """
        self.redis_client = redis_client
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_available(self) -> bool:
        """Check if Redis cache is available and healthy.

        Returns:
            True if Redis is connected and responsive, False otherwise.
        """
        if self.redis_client is None:
            return False

        try:
            self.redis_client.ping()
            return True
        except RedisError as e:
            self.logger.warning(f"Redis health check failed: {e}")
            return False

    def get_json(self, key: str) -> Any | None:
        """Get and deserialize a JSON value from cache.

        Args:
            key: Cache key

        Returns:
            Deserialized value or None if key doesn't exist

        Raises:
            BaseCacheError: On JSON deserialization error
        """
        if not self.is_available():
            return None

        try:
            value = self.redis_client.get(key)
            if value is None:
                return None

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            return json.loads(value)

        except RedisError as e:
            self.logger.warning(f"Redis get_json error for {key}: {e}")
            return None

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON deserialization error for {key}: {e}")
            return None

    def set_json(self, key: str, value: Any, ttl: int) -> bool:
        """Serialize and store a JSON value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if operation succeeded, False otherwise
        """
        if not self.is_available():
            return False

        try:
            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            self.logger.debug(f"Cached {key} with TTL {ttl}s")
            return True

        except RedisError as e:
            self.logger.warning(f"Redis set_json error for {key}: {e}")
            return False

        except (TypeError, ValueError) as e:
            self.logger.error(f"JSON serialization error for {key}: {e}")
            return False

    def set_string(self, key: str, value: str, ttl: int) -> bool:
        """Store a string value in cache.

        Args:
            key: Cache key
            value: String value to cache
            ttl: Time to live in seconds

        Returns:
            True if operation succeeded, False otherwise
        """
        if not self.is_available():
            return False

        try:
            self.redis_client.setex(key, ttl, value)
            self.logger.debug(f"Cached string {key} with TTL {ttl}s")
            return True

        except RedisError as e:
            self.logger.warning(f"Redis set_string error for {key}: {e}")
            return False

    def get_string(self, key: str) -> str | None:
        """Get a string value from cache.

        Args:
            key: Cache key

        Returns:
            String value or None if key doesn't exist
        """
        if not self.is_available():
            return None

        try:
            value = self.redis_client.get(key)
            if value is None:
                return None

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            return value

        except RedisError as e:
            self.logger.warning(f"Redis get_string error for {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False otherwise
        """
        if not self.is_available():
            return False

        try:
            self.redis_client.delete(key)
            self.logger.debug(f"Deleted cache key {key}")
            return True

        except RedisError as e:
            self.logger.warning(f"Redis delete error for {key}: {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "agg:*")

        Returns:
            Number of keys deleted
        """
        if not self.is_available():
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if not keys:
                return 0

            deleted_count = self.redis_client.delete(*keys)
            self.logger.debug(f"Deleted {deleted_count} keys matching {pattern}")
            return deleted_count

        except RedisError as e:
            self.logger.warning(f"Redis delete_pattern error for {pattern}: {e}")
            return 0

    def exists(self, key: str) -> bool:
        """Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self.is_available():
            return False

        try:
            return self.redis_client.exists(key) > 0

        except RedisError as e:
            self.logger.warning(f"Redis exists error for {key}: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all cache entries (use with caution).

        Returns:
            True if operation succeeded, False otherwise
        """
        if not self.is_available():
            return False

        try:
            self.redis_client.flushdb()
            self.logger.warning("Cleared all Redis cache entries")
            return True

        except RedisError as e:
            self.logger.warning(f"Redis flushdb error: {e}")
            return False
