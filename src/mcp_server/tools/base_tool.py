"""Simplified base tool class for async database access.

This module provides a minimal base class for all MCP tools with direct
access to the Motor database connection pool.

Key Features:
    - Direct access to Motor database
    - Zero locks (asyncio is single-threaded)
    - No connection managers
    - Pure async/await

Example:
    >>> from src.mcp_server.tools.base_tool import BaseTool
    >>> class MyTool(BaseTool):
    ...     async def my_operation(self):
    ...         db = self.get_database()
    ...         collection = db['my_collection']
    ...         async for doc in collection.find(query):
    ...             process(doc)
"""

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..database import database

logger = logging.getLogger(__name__)


class BaseTool:
    """Simplified base class for all MCP tools.

    Provides direct access to the Motor database connection pool.
    No locks, no managers, just simple async access.

    This class should be inherited by all tool classes to ensure consistent
    database access across the application.
    """

    def __init__(self) -> None:
        """Initialize tool.

        No connection setup needed - database is initialized at startup.
        """
        logger.debug(f"Initialized {self.__class__.__name__}")

    def get_database(self) -> AsyncIOMotorDatabase:
        """Get the Motor database instance.

        Returns the shared Motor database. No health checks, no locks.
        Motor handles connection management automatically.

        Returns:
            AsyncIOMotorDatabase instance

        Raises:
            RuntimeError: If database not initialized at startup

        Example:
            >>> tool = MyTool()
            >>> db = tool.get_database()
            >>> collection = db['patients']
        """
        return database.get_database()

    @classmethod
    def get_shared_database(cls) -> AsyncIOMotorDatabase:
        """Get database instance (class method for compatibility).

        Class method that delegates to the database module.
        Maintains backward compatibility with existing tool implementations.

        Returns:
            AsyncIOMotorDatabase instance

        Example:
            >>> db = MyTool.get_shared_database()
            >>> collection = db['my_collection']
        """
        return database.get_database()

    @classmethod
    async def health_check(cls) -> bool:
        """Check database connection status.

        Optional health check - Motor handles reconnection automatically.
        Use this for health endpoints or diagnostics.

        Returns:
            True if database responds to ping, False otherwise

        Example:
            >>> if not await MyTool.health_check():
            ...     print("Database unavailable")
        """
        return await database.health_check()

    def __repr__(self) -> str:
        """Return string representation of the tool instance.

        Returns:
            String representation including class name
        """
        return f"{self.__class__.__name__}()"
