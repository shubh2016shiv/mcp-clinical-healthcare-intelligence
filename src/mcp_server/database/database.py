"""Dead-simple database access for all tools.

No managers. No locks. No complexity.
Just Motor's async connection pool.
"""

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================


class QueryExecutionError(Exception):
    """Error during query execution."""

    pass


# ============================================================================
# MODULE STATE (Simple and Explicit)
# ============================================================================

_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None
_database_name: str | None = None


# ============================================================================
# INITIALIZATION (Call Once at Startup)
# ============================================================================


def initialize(
    connection_uri: str, database_name: str, min_pool_size: int = 10, max_pool_size: int = 50
) -> None:
    """Initialize database connection pool.

    Call this ONCE when your application starts.
    Motor handles all connection pooling, thread safety, and async operations.

    Args:
        connection_uri: MongoDB connection string
        database_name: Name of database to use
        min_pool_size: Minimum connections in pool
        max_pool_size: Maximum connections in pool
    """
    global _client, _database, _database_name

    logger.info(f"Initializing Motor connection pool for database: {database_name}")

    _client = AsyncIOMotorClient(
        connection_uri,
        minPoolSize=min_pool_size,
        maxPoolSize=max_pool_size,
        maxIdleTimeMS=45000,  # Close idle connections after 45s
        serverSelectionTimeoutMS=5000,  # Fail fast if MongoDB unavailable
        retryWrites=True,
        retryReads=True,
    )

    _database = _client[database_name]
    _database_name = database_name

    logger.info(f"✓ Database connection pool ready (pool: {min_pool_size}-{max_pool_size})")


# ============================================================================
# ACCESS (Simple Getters)
# ============================================================================


def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance.

    Returns the Motor database. No health checks, no locks.
    Motor handles connection management automatically.

    Raises:
        RuntimeError: If database not initialized
    """
    if _database is None:
        raise RuntimeError("Database not initialized. Call database.initialize() at startup.")
    return _database


def get_client() -> AsyncIOMotorClient:
    """Get the Motor client instance.

    Useful for admin operations or accessing multiple databases.

    Raises:
        RuntimeError: If client not initialized
    """
    if _client is None:
        raise RuntimeError(
            "Database client not initialized. Call database.initialize() at startup."
        )
    return _client


def get_database_name() -> str:
    """Get the current database name."""
    if _database_name is None:
        raise RuntimeError("Database not initialized.")
    return _database_name


# ============================================================================
# HEALTH CHECK (Optional, Non-Blocking)
# ============================================================================


async def health_check() -> bool:
    """Check if database is reachable.

    This is optional - Motor handles reconnection automatically.
    Use this for health endpoints or diagnostics.

    Returns:
        True if database responds to ping, False otherwise
    """
    if _database is None:
        return False

    try:
        await _client.admin.command("ping")
        return True
    except Exception as error:
        logger.warning(f"Database health check failed: {error}")
        return False


# ============================================================================
# SHUTDOWN (Call on Application Exit)
# ============================================================================


def shutdown() -> None:
    """Close database connections gracefully.

    Call this when your application shuts down.
    Motor will close all connections in the pool.
    """
    global _client, _database, _database_name

    if _client is not None:
        logger.info("Closing database connections...")
        _client.close()
        _client = None
        _database = None
        _database_name = None
        logger.info("✓ Database connections closed")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


async def example_usage():
    """Example showing typical usage pattern."""

    # 1. At application startup
    initialize(
        connection_uri="mongodb://localhost:27017",
        database_name="healthcare",
        min_pool_size=10,
        max_pool_size=50,
    )

    # 2. In your async functions
    db = get_database()
    patients = db["patients"]

    # Motor handles everything - just use async/await
    async for patient in patients.find({"state": "CA"}).limit(10):
        print(f"Patient: {patient['first_name']} {patient['last_name']}")

    # 3. Optional health check
    is_healthy = await health_check()
    print(f"Database healthy: {is_healthy}")

    # 4. On application shutdown
    shutdown()


# ============================================================================
# WHY THIS WORKS
# ============================================================================
#
# Motor (MongoDB's async driver) is designed for this exact pattern:
#
# 1. Thread-safe: One client can be shared across all async tasks
# 2. Connection pooling: Built-in, no need for custom pool management
# 3. Automatic reconnection: Handles network issues transparently
# 4. Non-blocking: All operations are async by default
# 5. No locks needed: Motor uses asyncio, which is single-threaded
#
# Common pattern in production async Python apps:
# - FastAPI: Uses this exact pattern
# - Django Async: Uses this exact pattern
# - Sanic: Uses this exact pattern
#
# ============================================================================
