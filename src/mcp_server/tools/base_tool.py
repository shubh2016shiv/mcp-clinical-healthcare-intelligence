"""Base tool class with optimized connection management.

This module provides a base class for all MCP tools with centralized,
efficient connection handling. Eliminates redundant connection checks and
optimizes database access patterns.

Key Features:
    - Shared connection instance across all tools
    - Connection pooling optimization
    - Lazy initialization with health checks
    - Thread-safe operations
    - Comprehensive logging

Example:
    >>> from src.mcp_server.tools.base_tool import BaseTool
    >>> class MyTool(BaseTool):
    ...     async def my_operation(self):
    ...         db = self.get_database()
    ...         collection = db['my_collection']
    ...         # ... query operations ...
"""

import logging
import threading
from typing import TYPE_CHECKING, Optional

from pymongo.database import Database

from ..database.connection import QueryExecutionError, get_connection_manager

if TYPE_CHECKING:
    from ..database.connection import ConnectionManager

logger = logging.getLogger(__name__)


class BaseTool:
    """Base class for all MCP tools with optimized connection management.

    Provides a shared, efficiently managed database connection instance for
    all tools. Implements connection pooling and lazy initialization patterns.

    This class should be inherited by all tool classes to ensure consistent,
    optimal connection management across the application.

    Attributes:
        _shared_db: Class-level shared database instance
        _shared_connection_manager: Class-level shared connection manager
    """

    # Class-level shared resources (singleton pattern)
    _shared_db: Database | None = None
    _shared_connection_manager: Optional["ConnectionManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize tool with optimized connection setup.

        Uses class-level shared resources to avoid redundant connection
        management per tool instance.
        """
        # No need for instance-level db - use class-level shared resource
        logger.debug(f"Initialized {self.__class__.__name__}")

    @classmethod
    def _ensure_connection(cls) -> None:
        """Ensure shared connection is established (called once at class load).

        This method is called automatically when first tool method accesses
        the database, establishing the shared connection pool.

        Thread-safe: Uses locking to prevent race conditions during initialization.

        Raises:
            QueryExecutionError: If connection cannot be established
        """
        if cls._shared_connection_manager is None:
            with cls._lock:
                # Double-check pattern to avoid race conditions
                if cls._shared_connection_manager is None:
                    cls._shared_connection_manager = get_connection_manager()

        if not cls._shared_connection_manager.is_connected():
            logger.debug("Establishing shared database connection for tools...")
            if not cls._shared_connection_manager.connect():
                raise QueryExecutionError(
                    "Failed to establish shared database connection", operation="connection_setup"
                )

    @classmethod
    def get_shared_database(cls) -> Database:
        """Get the shared, pooled database instance.

        This is the optimized method for accessing the database. All tool
        instances share a single database connection, eliminating connection
        pooling overhead.

        Thread-safe: Uses locking to prevent race conditions during initialization.

        Returns:
            Shared MongoDB Database instance

        Raises:
            QueryExecutionError: If connection is not available

        Example:
            >>> db = MyTool.get_shared_database()
            >>> collection = db['my_collection']
        """
        cls._ensure_connection()

        if cls._shared_db is None:
            with cls._lock:
                # Double-check pattern to avoid race conditions
                if cls._shared_db is None:
                    cls._shared_db = cls._shared_connection_manager.get_database()
                    logger.debug("Initialized shared database instance for tools")

        # Add health check
        if not cls._shared_connection_manager.health_check():
            logger.warning("Connection health check failed, reconnecting...")
            cls.reset_shared_connection()
            # After reset, get the database instance again
            with cls._lock:
                cls._shared_db = cls._shared_connection_manager.get_database()

        return cls._shared_db

    def get_database(self) -> Database:
        """Get database instance (instance method for compatibility).

        Instance method that delegates to the class-level shared database.
        Maintains backward compatibility with existing tool implementations.

        Returns:
            Shared MongoDB Database instance

        Example:
            >>> tool = MyTool()
            >>> db = tool.get_database()
        """
        return self.__class__.get_shared_database()

    @classmethod
    def get_connection_manager(cls) -> "ConnectionManager":
        """Get the shared connection manager instance.

        Returns:
            Shared ConnectionManager instance

        Example:
            >>> manager = MyTool.get_connection_manager()
            >>> if manager.is_connected():
            ...     print("Connected to MongoDB")
        """
        if cls._shared_connection_manager is None:
            cls._ensure_connection()
        return cls._shared_connection_manager

    @classmethod
    def health_check(cls) -> bool:
        """Perform health check on shared database connection.

        Checks the status of the shared connection and reconnects if needed.

        Returns:
            True if connection is healthy, False otherwise

        Example:
            >>> if not MyTool.health_check():
            ...     # Handle connection issue
            ...     pass
        """
        if cls._shared_connection_manager is None:
            return False

        return cls._shared_connection_manager.health_check()

    @classmethod
    def reset_shared_connection(cls) -> None:
        """Reset and re-establish the shared connection.

        Thread-safe: Uses locking to prevent race conditions during reset.

        Useful for handling reconnection after network issues or
        for testing purposes.

        Example:
            >>> MyTool.reset_shared_connection()
        """
        logger.info(f"Resetting shared connection for {cls.__name__}")
        with cls._lock:
            if cls._shared_connection_manager:
                cls._shared_connection_manager.disconnect()
            cls._shared_db = None
        cls._ensure_connection()

    def __repr__(self) -> str:
        """Return string representation of the tool instance.

        Returns:
            String representation including class name and connection status
        """
        connection_status = "connected" if self.__class__.health_check() else "disconnected"
        return f"{self.__class__.__name__}(connection={connection_status})"
