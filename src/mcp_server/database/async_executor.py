"""Async executor pool for running blocking MongoDB operations in thread pool.

This module provides a centralized ThreadPoolExecutor for running synchronous
pymongo operations without blocking the asyncio event loop. Implements singleton
pattern for efficient resource management.

Key Features:
    - Singleton pattern for shared executor across application
    - Thread-safe operations
    - Configurable pool size based on application settings
    - Graceful shutdown support
    - Comprehensive logging for debugging

Example:
    >>> from src.mcp_server.database.async_executor import get_executor_pool
    >>> executor = get_executor_pool()
    >>> loop = asyncio.get_event_loop()
    >>> result = await loop.run_in_executor(executor, blocking_operation, arg1, arg2)
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)


class AsyncExecutorPool:
    """Thread pool executor for running blocking operations asynchronously.

    This class provides a singleton thread pool executor optimized for running
    synchronous pymongo operations without blocking the asyncio event loop.

    Implements singleton pattern - use get_executor_pool() to retrieve instance.

    Attributes:
        DEFAULT_MAX_WORKERS: Default number of threads in pool (10)
    """

    DEFAULT_MAX_WORKERS: int = 10
    _instance: Optional["AsyncExecutorPool"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AsyncExecutorPool":
        """Implement singleton pattern with thread-safe instantiation.

        Returns:
            The single AsyncExecutorPool instance for the application

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
        """Initialize the async executor pool.

        Sets up internal state and logging. Called automatically by __new__
        but guards against re-initialization.
        """
        if self._initialized:
            return

        # Determine pool size - use config if available, otherwise default
        max_workers = getattr(settings, "async_executor_max_workers", self.DEFAULT_MAX_WORKERS)

        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mcp-db-worker-"
        )
        self._initialized = True

        logger.debug(f"AsyncExecutorPool initialized with {max_workers} workers")

    def get_executor(self) -> ThreadPoolExecutor:
        """Get the thread pool executor instance.

        Returns:
            ThreadPoolExecutor for running blocking operations

        Example:
            >>> executor = pool.get_executor()
            >>> loop.run_in_executor(executor, sync_operation)
        """
        return self._executor

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown the executor pool.

        Should be called during application shutdown to ensure all threads
        are properly cleaned up.

        Args:
            wait: If True, wait for all pending operations to complete

        Example:
            >>> pool = get_executor_pool()
            >>> pool.shutdown(wait=True)
        """
        if self._executor is not None:
            try:
                logger.info(f"Shutting down AsyncExecutorPool (wait={wait})...")
                self._executor.shutdown(wait=wait)
                logger.info("AsyncExecutorPool shutdown complete")
            except Exception as e:
                logger.error(f"Error during AsyncExecutorPool shutdown: {e}")

    def __del__(self) -> None:
        """Cleanup executor on object deletion."""
        if hasattr(self, "_executor") and self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass


# Global singleton instance
_executor_pool: AsyncExecutorPool | None = None


def get_executor_pool() -> AsyncExecutorPool:
    """Get the global singleton AsyncExecutorPool instance.

    Creates the executor pool on first call, subsequent calls return
    the same instance. Thread-safe implementation.

    Returns:
        The global AsyncExecutorPool singleton instance

    Example:
        >>> from src.mcp_server.database.async_executor import get_executor_pool
        >>> pool = get_executor_pool()
        >>> executor = pool.get_executor()
    """
    global _executor_pool
    if _executor_pool is None:
        _executor_pool = AsyncExecutorPool()
    return _executor_pool
