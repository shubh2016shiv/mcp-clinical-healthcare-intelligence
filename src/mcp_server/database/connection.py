"""MongoDB Connection Manager with connection pooling and automatic lifecycle management.

This module provides a thread-safe connection manager for MongoDB operations with built-in
connection pooling, retry logic, and graceful error handling. It implements the singleton
pattern to ensure a single shared connection pool across the application.

The module also provides the @ensure_connected decorator for automatic connection
management on tool functions.

Key Features:
    - Singleton connection pool management
    - Automatic retry logic with exponential backoff
    - Health check capabilities
    - Thread-safe operations
    - Graceful degradation and error recovery
    - Comprehensive logging for debugging

Example:
    >>> from src.mcp_server.database.connection import (
    ...     get_connection_manager,
    ...     ensure_connected
    ... )
    >>> manager = get_connection_manager()
    >>> manager.connect()
    >>> db = manager.get_database()
    >>> manager.disconnect()
"""

import functools
import logging
import threading
import time
from typing import Any, Callable, Coroutine, Optional, TypeVar

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import (
    ConnectionFailure,
    ConnectionPoolClosed,
    NetworkTimeout,
    OperationFailure,
    ServerSelectionTimeoutError,
)

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


class QueryExecutionError(Exception):
    """Custom exception raised when a database query execution fails.

    This exception wraps MongoDB-specific errors and provides a consistent
    error interface for the rest of the application. It includes details
    about the operation, collection, and underlying cause.

    Attributes:
        message: Human-readable error description
        operation: The type of operation that failed (e.g., 'find', 'insert')
        collection: The collection name where the error occurred
        cause: The underlying MongoDB exception
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        collection: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize QueryExecutionError.

        Args:
            message: Descriptive error message
            operation: Type of operation that failed
            collection: Collection involved in the operation
            cause: Original exception that caused this error
        """
        self.message = message
        self.operation = operation
        self.collection = collection
        self.cause = cause
        full_message = message
        if operation:
            full_message = f"{message} (operation: {operation})"
        if collection:
            full_message = f"{full_message}, collection: {collection}"
        super().__init__(full_message)


class ConnectionManager:
    """Thread-safe MongoDB connection manager with automatic connection pooling.

    This class manages the lifecycle of MongoDB connections and provides a single
    entry point for database access. It implements connection pooling, retry logic,
    and health checking to ensure reliable database connectivity.

    Implements singleton pattern - use get_connection_manager() to retrieve instance.

    Attributes:
        MAX_RETRIES: Maximum number of connection retry attempts
        RETRY_DELAY: Initial delay in seconds between retries
        MAX_RETRY_DELAY: Maximum delay between retries (with exponential backoff)
        HEALTH_CHECK_INTERVAL: Seconds between health checks
    """

    # Class-level constants for retry behavior
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0  # Initial 1 second
    MAX_RETRY_DELAY: float = 10.0  # Cap at 10 seconds
    HEALTH_CHECK_INTERVAL: int = 300  # Check health every 5 minutes

    _instance: Optional["ConnectionManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ConnectionManager":
        """Implement singleton pattern with thread-safe instantiation.

        Returns:
            The single ConnectionManager instance for the application

        Note:
            Uses double-checked locking pattern for thread safety and performance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the connection manager.

        Sets up internal state and logging. Called automatically by __new__
        but guards against re-initialization.
        """
        if self._initialized:
            return

        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        self._connected: bool = False
        self._last_health_check: float = 0.0
        self._lock = threading.Lock()
        self._initialized = True

        logger.debug("ConnectionManager initialized")

    def connect(self) -> bool:
        """Establish connection to MongoDB with automatic retry logic.

        This method connects to MongoDB using the configured URI and applies
        connection pool settings. Implements exponential backoff retry strategy
        for resilience against temporary network issues.

        Returns:
            True if connection successful, False otherwise

        Raises:
            QueryExecutionError: If connection fails after all retries

        Example:
            >>> manager = get_connection_manager()
            >>> if manager.connect():
            ...     db = manager.get_database()
        """
        with self._lock:
            if self._connected:
                logger.debug("Already connected to MongoDB")
                return True

            logger.info("Attempting to connect to MongoDB...")

            for attempt in range(self.MAX_RETRIES):
                try:
                    # Create client with configured pool settings
                    self._client = MongoClient(
                        settings.mongodb_connection_string,
                        serverSelectionTimeoutMS=settings.mongodb_timeout * 1000,
                        connectTimeoutMS=settings.mongodb_timeout * 1000,
                        minPoolSize=settings.mongodb_min_pool_size,
                        maxPoolSize=settings.mongodb_max_pool_size,
                    )

                    # Test connection by pinging the server
                    self._client.admin.command("ping")
                    self._database = self._client[settings.mongodb_database]
                    self._connected = True
                    self._last_health_check = time.time()

                    logger.info(
                        f"Successfully connected to MongoDB at "
                        f"{settings.mongodb_uri} (database: {settings.mongodb_database})"
                    )
                    return True

                except (ServerSelectionTimeoutError, ConnectionFailure, NetworkTimeout) as e:
                    attempt_num = attempt + 1
                    if attempt_num < self.MAX_RETRIES:
                        # Calculate exponential backoff delay
                        delay = min(
                            self.RETRY_DELAY * (2 ** attempt),
                            self.MAX_RETRY_DELAY,
                        )
                        logger.warning(
                            f"Connection attempt {attempt_num}/{self.MAX_RETRIES} failed. "
                            f"Retrying in {delay:.1f} seconds... Error: {e}"
                        )
                        time.sleep(delay)
                    else:
                        error_msg = (
                            f"Failed to connect to MongoDB after {self.MAX_RETRIES} attempts"
                        )
                        logger.error(f"{error_msg}: {e}")
                        raise QueryExecutionError(
                            error_msg,
                            operation="connect",
                            cause=e,
                        ) from e

                except Exception as e:
                    logger.error(f"Unexpected error during MongoDB connection: {e}")
                    raise QueryExecutionError(
                        f"Unexpected error connecting to MongoDB: {e}",
                        operation="connect",
                        cause=e,
                    ) from e

            return False

    def disconnect(self) -> bool:
        """Gracefully close the MongoDB connection and cleanup resources.

        This method should be called during application shutdown to ensure
        all resources are properly released.

        Returns:
            True if disconnection successful or not connected, False on error

        Example:
            >>> manager = get_connection_manager()
            >>> try:
            ...     db = manager.get_database()
            ...     # use database
            ... finally:
            ...     manager.disconnect()
        """
        with self._lock:
            if not self._connected or self._client is None:
                logger.debug("Not connected to MongoDB, nothing to disconnect")
                return True

            try:
                logger.info("Disconnecting from MongoDB...")
                self._client.close()
                self._connected = False
                self._database = None
                self._client = None
                logger.info("Successfully disconnected from MongoDB")
                return True

            except Exception as e:
                logger.error(f"Error during MongoDB disconnection: {e}")
                return False

    def get_database(self) -> Database:
        """Get the MongoDB database instance for query execution.

        This method returns the configured database object for performing
        operations. Ensures connection is established before returning.

        Returns:
            MongoDB Database object for executing queries

        Raises:
            QueryExecutionError: If not connected or database is unavailable

        Example:
            >>> manager = get_connection_manager()
            >>> db = manager.get_database()
            >>> collection = db['users']
        """
        if not self._connected or self._database is None:
            raise QueryExecutionError(
                "Not connected to MongoDB. Call connect() first",
                operation="get_database",
            )

        return self._database

    def is_connected(self) -> bool:
        """Check if connection to MongoDB is active.

        Returns:
            True if connected, False otherwise

        Example:
            >>> manager = get_connection_manager()
            >>> if manager.is_connected():
            ...     print("Connected to MongoDB")
        """
        return self._connected

    def health_check(self) -> bool:
        """Perform health check on MongoDB connection.

        Executes a ping command to verify the connection is still active.
        Includes rate limiting to avoid excessive checks.

        Returns:
            True if health check passes, False otherwise

        Example:
            >>> manager = get_connection_manager()
            >>> if manager.health_check():
            ...     print("MongoDB is healthy")
        """
        # Rate limit health checks
        current_time = time.time()
        if current_time - self._last_health_check < self.HEALTH_CHECK_INTERVAL:
            return self._connected

        try:
            if not self._connected or self._client is None:
                return False

            self._client.admin.command("ping")
            self._last_health_check = current_time
            logger.debug("MongoDB health check passed")
            return True

        except Exception as e:
            logger.warning(f"MongoDB health check failed: {e}")
            self._connected = False
            return False


# Global singleton instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global singleton ConnectionManager instance.

    Creates the connection manager on first call, subsequent calls return
    the same instance. Thread-safe implementation.

    Returns:
        The global ConnectionManager singleton instance

    Example:
        >>> from src.mcp_server.database.connection import get_connection_manager
        >>> manager = get_connection_manager()
        >>> manager.connect()
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


def ensure_connected(func: F) -> F:
    """Decorator to ensure MongoDB connection before executing a function.

    This decorator automatically handles connection establishment and health
    checking before function execution. If not connected, it attempts to
    establish a connection. Gracefully handles connection failures.

    The decorated function receives an active database connection through
    the connection manager.

    Args:
        func: The function to decorate

    Returns:
        Wrapped function that ensures MongoDB connection

    Raises:
        QueryExecutionError: If connection cannot be established

    Example:
        >>> @ensure_connected
        ... def get_collections() -> Dict[str, Any]:
        ...     manager = get_connection_manager()
        ...     db = manager.get_database()
        ...     return {"collections": db.list_collection_names()}
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function that ensures connection before execution.

        Args:
            *args: Positional arguments for the decorated function
            **kwargs: Keyword arguments for the decorated function

        Returns:
            Result from the decorated function

        Raises:
            QueryExecutionError: If connection setup fails
        """
        manager = get_connection_manager()

        try:
            # Check if connected, if not attempt connection
            if not manager.is_connected():
                logger.debug(
                    f"Connection not active. Attempting to establish connection "
                    f"before executing {func.__name__}..."
                )
                if not manager.connect():
                    raise QueryExecutionError(
                        f"Failed to establish MongoDB connection for {func.__name__}",
                        operation=func.__name__,
                    )

            # Perform health check to ensure connection is still valid
            if not manager.health_check():
                logger.warning("Health check failed. Attempting to reconnect...")
                if not manager.connect():
                    raise QueryExecutionError(
                        f"Failed to reconnect to MongoDB for {func.__name__}",
                        operation=func.__name__,
                    )

            # Execute the decorated function
            logger.debug(f"Executing {func.__name__} with active MongoDB connection")
            return func(*args, **kwargs)

        except ConnectionPoolClosed as e:
            logger.error(f"Connection pool was closed during {func.__name__}: {e}")
            raise QueryExecutionError(
                f"MongoDB connection pool closed during {func.__name__}",
                operation=func.__name__,
                cause=e,
            ) from e

        except QueryExecutionError:
            # Re-raise our custom exceptions
            raise

        except Exception as e:
            logger.error(
                f"Unexpected error in ensure_connected for {func.__name__}: {e}",
                exc_info=True,
            )
            raise QueryExecutionError(
                f"Unexpected error during {func.__name__}: {e}",
                operation=func.__name__,
                cause=e,
            ) from e

    return wrapper  # type: ignore


if __name__ == "__main__":
    # Script: Test database connectivity
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        logger.info("Starting MongoDB connection test...")

        # Get connection manager and connect
        manager = get_connection_manager()
        logger.info("Attempting to connect to MongoDB...")

        if not manager.connect():
            logger.error("Failed to connect to MongoDB")
            sys.exit(1)

        # Test database access
        logger.info("Testing database access...")
        db = manager.get_database()
        collections = db.list_collection_names()
        logger.info(f"✓ Successfully retrieved collections: {collections}")

        # Health check
        logger.info("Running health check...")
        if manager.health_check():
            logger.info("✓ Health check passed")
        else:
            logger.error("✗ Health check failed")

        # Cleanup
        manager.disconnect()
        logger.info("✓ Connection test completed successfully")
        sys.exit(0)

    except QueryExecutionError as e:
        logger.error(f"✗ Query execution error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)
