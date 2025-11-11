"""Centralized cache manager coordinating all cache components.

This module provides a singleton CacheManager that initializes and coordinates
all cache implementations, similar to ConnectionManager pattern used for MongoDB.
"""

import logging
import threading
from typing import Any, Optional

try:
    import redis
    from redis.connection import ConnectionPool
    from redis.exceptions import ConnectionError, RedisError

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # Create dummy types for type hints when redis is not available
    redis = None  # type: ignore
    ConnectionPool = None  # type: ignore
    ConnectionError = Exception  # type: ignore
    RedisError = Exception  # type: ignore

from src.config.settings import settings

from .aggregation_cache import AggregationCache
from .field_schema_cache import FieldSchemaCache
from .prompt_cache import PromptCache
from .rbac_cache import RBACCache
from .session_cache import SessionCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Centralized manager for all Redis cache components.

    Implements singleton pattern to ensure single Redis connection pool
    across the application. Manages initialization, health checking, and
    graceful fallback when Redis is unavailable.

    Attributes:
        rbac_cache: RBAC decision cache
        session_cache: Session management cache
        prompt_cache: Tool prompt cache
        field_schema_cache: Data minimization schema cache
        aggregation_cache: Analytics result cache
    """

    _instance: Optional["CacheManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "CacheManager":
        """Implement singleton pattern with thread-safe instantiation.

        Returns:
            The single CacheManager instance for the application
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the cache manager.

        Sets up Redis connection and initializes all cache components.
        Called automatically but guards against re-initialization.
        """
        if self._initialized:
            return

        self._redis_client: Any | None = None  # type: ignore
        self._available: bool = False
        self._initialized = True

        # Initialize Redis connection if enabled and available
        if settings.redis_enabled and REDIS_AVAILABLE:
            self._initialize_redis()
        elif settings.redis_enabled and not REDIS_AVAILABLE:
            logger.warning("Redis is enabled in settings but redis package is not installed")
            logger.warning("Install redis with: pip install redis (or uv add redis)")
            logger.warning("Continuing without Redis cache")
            self._available = False
            self._redis_client = None

        # Initialize all cache components
        self._initialize_caches()

        logger.info(
            f"CacheManager initialized (Redis {'enabled' if self._available else 'disabled'})"
        )

    def _initialize_redis(self) -> None:
        """Initialize Redis connection with connection pooling.

        Handles SSL configuration robustly across different redis-py versions.
        Gracefully falls back if SSL configuration is incompatible with the installed version.
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis package not available, cannot initialize Redis connection")
            self._available = False
            self._redis_client = None
            return

        try:
            # Detect redis-py version for compatibility handling
            redis_version = getattr(redis, "__version__", "0.0.0")
            version_major = int(redis_version.split(".")[0]) if redis_version else 0

            # For redis-py v5.0+, use from_url which handles SSL via protocol
            # For redis-py v4.x, use ConnectionPool with ssl parameter
            if version_major >= 5 and settings.redis_ssl:
                # Use URL-based connection for SSL (recommended for v5+)
                protocol = "rediss"  # rediss:// indicates SSL/TLS
                auth_part = f":{settings.redis_password}@" if settings.redis_password else ""
                redis_url = f"{protocol}://{auth_part}{settings.redis_host}:{settings.redis_port}"

                self._redis_client = redis.from_url(
                    redis_url,
                    socket_connect_timeout=settings.redis_socket_timeout,
                    socket_timeout=settings.redis_socket_timeout,
                    max_connections=settings.redis_max_connections,
                    decode_responses=True,
                )
            else:
                # Use ConnectionPool for non-SSL or redis-py v4.x
                connection_params = {
                    "host": settings.redis_host,
                    "port": settings.redis_port,
                    "socket_connect_timeout": settings.redis_socket_timeout,
                    "socket_timeout": settings.redis_socket_timeout,
                    "max_connections": settings.redis_max_connections,
                    "decode_responses": True,
                }

                # Add password only if provided
                if settings.redis_password:
                    connection_params["password"] = settings.redis_password

                # Add SSL parameter only for redis-py v4.x when SSL is enabled
                if settings.redis_ssl and version_major < 5:
                    connection_params["ssl"] = True

                # Create connection pool
                pool = ConnectionPool(**connection_params)
                self._redis_client = redis.Redis(connection_pool=pool)  # type: ignore

            # Test connection
            self._redis_client.ping()
            self._available = True

            ssl_status = "with SSL" if settings.redis_ssl else "without SSL"
            connection_method = (
                "URL-based" if (version_major >= 5 and settings.redis_ssl) else "ConnectionPool"
            )
            logger.info(
                f"Connected to Redis at {settings.redis_host}:{settings.redis_port} {ssl_status} (redis-py {redis_version}, {connection_method})"
            )

        except TypeError as e:
            # Handle SSL parameter incompatibility
            if "ssl" in str(e).lower() or "unexpected keyword argument" in str(e).lower():
                logger.warning(
                    f"Redis SSL configuration incompatible with installed redis-py version: {e}"
                )
                logger.info("Retrying Redis connection without SSL parameter...")

                # Retry without SSL parameter - build connection URL instead
                try:
                    protocol = "rediss" if settings.redis_ssl else "redis"
                    auth_part = f":{settings.redis_password}@" if settings.redis_password else ""
                    redis_url = (
                        f"{protocol}://{auth_part}{settings.redis_host}:{settings.redis_port}"
                    )

                    # Use from_url which handles SSL automatically via protocol
                    self._redis_client = redis.from_url(
                        redis_url,
                        socket_connect_timeout=settings.redis_socket_timeout,
                        socket_timeout=settings.redis_socket_timeout,
                        max_connections=settings.redis_max_connections,
                        decode_responses=True,
                    )

                    # Test connection
                    self._redis_client.ping()
                    self._available = True

                    ssl_status = "with SSL" if settings.redis_ssl else "without SSL"
                    logger.info(
                        f"Connected to Redis at {settings.redis_host}:{settings.redis_port} {ssl_status} (via URL)"
                    )
                except Exception as retry_error:
                    logger.error(f"Failed to connect to Redis after SSL retry: {retry_error}")
                    logger.warning("Continuing without Redis cache (performance degraded)")
                    self._available = False
                    self._redis_client = None
            else:
                # Re-raise if it's a different TypeError
                raise

        except (ConnectionError, RedisError) as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            logger.warning("Continuing without Redis cache (performance degraded)")
            self._available = False
            self._redis_client = None

        except Exception as e:
            logger.error(f"Unexpected error initializing Redis: {e}", exc_info=True)
            self._available = False
            self._redis_client = None

    def _initialize_caches(self) -> None:
        """Initialize all cache components."""
        # Initialize caches with Redis client (or None if unavailable)
        self.rbac_cache = RBACCache(self._redis_client)
        self.session_cache = SessionCache(self._redis_client)
        self.prompt_cache = PromptCache(self._redis_client)
        self.field_schema_cache = FieldSchemaCache(self._redis_client)
        self.aggregation_cache = AggregationCache(self._redis_client)

    def is_available(self) -> bool:
        """Check if Redis cache is available.

        Returns:
            True if Redis is connected and healthy, False otherwise
        """
        return self._available

    def is_connected(self) -> bool:
        """Check if currently connected to Redis.

        Returns:
            True if connected, False otherwise
        """
        return self._redis_client is not None

    def health_check(self) -> bool:
        """Perform health check on Redis connection.

        Returns:
            True if health check passes, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            self._redis_client.ping()
            return True
        except RedisError as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    def disconnect(self) -> bool:
        """Close Redis connection and cleanup resources.

        Returns:
            True if disconnection successful, False on error
        """
        if self._redis_client is None:
            logger.debug("Not connected to Redis, nothing to disconnect")
            return True

        try:
            connection_pool = self._redis_client.connection_pool
            connection_pool.disconnect()
            self._redis_client = None
            self._available = False
            logger.info("Disconnected from Redis")
            return True

        except Exception as e:
            logger.error(f"Error during Redis disconnection: {e}")
            return False

    def get_status(self) -> dict:
        """Get comprehensive cache status for monitoring.

        Returns:
            Dict with cache status information
        """
        return {
            "available": self.is_available(),
            "connected": self.is_connected(),
            "health_check": self.health_check(),
            "redis_host": settings.redis_host if self.is_available() else None,
            "redis_port": settings.redis_port if self.is_available() else None,
        }


# Global singleton instance
_cache_manager: CacheManager | None = None


def initialize_cache() -> CacheManager:
    """Initialize the cache manager.

    Returns:
        Initialized CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_cache_manager() -> CacheManager:
    """Get the global singleton CacheManager instance.

    Creates the cache manager on first call, subsequent calls return
    the same instance.

    Returns:
        The global CacheManager singleton instance

    Raises:
        RuntimeError: If cache manager fails to initialize
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
