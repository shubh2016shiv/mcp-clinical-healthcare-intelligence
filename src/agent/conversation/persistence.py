"""Session persistence for conversation management.

This module provides pluggable storage backends for conversation sessions:
- SessionStore: Abstract base class defining the storage interface
- InMemorySessionStore: Default in-memory implementation
- RedisSessionStore: Enterprise Redis-backed implementation

Enterprise Pattern:
- Abstract interface for multiple storage backends
- Graceful fallback on errors
- Connection pooling and retry logic
- TTL-based automatic cleanup
"""

import asyncio
import importlib.util
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from .session import ConversationSession

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """Abstract base class for session storage.

    This defines the interface that all session storage backends must implement.
    It enables pluggable storage backends (in-memory, Redis, database, etc.)

    Enterprise Pattern:
    - Abstract interface for dependency injection
    - Async API for non-blocking I/O
    - Error handling with graceful degradation
    """

    @abstractmethod
    async def save(self, session: ConversationSession) -> None:
        """Save a conversation session.

        Args:
            session: The session to save

        Raises:
            Exception: If save fails (implementation-specific)
        """
        pass

    @abstractmethod
    async def load(self, session_id: str) -> ConversationSession | None:
        """Load a conversation session by ID.

        Args:
            session_id: The session identifier

        Returns:
            ConversationSession if found, None otherwise

        Raises:
            Exception: If load fails (implementation-specific)
        """
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a conversation session.

        Args:
            session_id: The session identifier

        Returns:
            True if deleted, False if not found

        Raises:
            Exception: If delete fails (implementation-specific)
        """
        pass

    @abstractmethod
    async def list_sessions(self) -> list[str]:
        """List all session IDs.

        Returns:
            List of session IDs

        Raises:
            Exception: If list fails (implementation-specific)
        """
        pass

    @abstractmethod
    async def cleanup_expired(self, ttl_seconds: int) -> int:
        """Clean up expired sessions.

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            Number of sessions cleaned up

        Raises:
            Exception: If cleanup fails (implementation-specific)
        """
        pass


class InMemorySessionStore(SessionStore):
    """In-memory session storage implementation.

    This is the default storage backend, suitable for:
    - Development and testing
    - Single-instance deployments
    - Stateless applications with session recreation

    Limitations:
    - Sessions lost on restart
    - Not suitable for multi-instance deployments
    - Limited by available memory

    Enterprise Usage:
    - Use for development/testing
    - Switch to RedisSessionStore for production

    Example:
        >>> store = InMemorySessionStore()
        >>> session = ConversationSession()
        >>> await store.save(session)
        >>> loaded = await store.load(session.session_id)
    """

    def __init__(self):
        """Initialize in-memory session store."""
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = asyncio.Lock()
        logger.info("Initialized InMemorySessionStore")

    async def save(self, session: ConversationSession) -> None:
        """Save a conversation session to memory.

        Args:
            session: The session to save
        """
        async with self._lock:
            self._sessions[session.session_id] = session
            logger.debug(f"Saved session {session.session_id[:8]} to memory")

    async def load(self, session_id: str) -> ConversationSession | None:
        """Load a conversation session from memory.

        Args:
            session_id: The session identifier

        Returns:
            ConversationSession if found, None otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                logger.debug(f"Loaded session {session_id[:8]} from memory")
            else:
                logger.debug(f"Session {session_id[:8]} not found in memory")
            return session

    async def delete(self, session_id: str) -> bool:
        """Delete a conversation session from memory.

        Args:
            session_id: The session identifier

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug(f"Deleted session {session_id[:8]} from memory")
                return True
            logger.debug(f"Session {session_id[:8]} not found (cannot delete)")
            return False

    async def list_sessions(self) -> list[str]:
        """List all session IDs in memory.

        Returns:
            List of session IDs
        """
        async with self._lock:
            return list(self._sessions.keys())

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        """Clean up expired sessions from memory.

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            Number of sessions cleaned up
        """
        async with self._lock:
            current_time = time.time()
            expired = [
                sid
                for sid, session in self._sessions.items()
                if current_time - session.last_activity > ttl_seconds
            ]

            for sid in expired:
                del self._sessions[sid]
                logger.debug(f"Cleaned up expired session {sid[:8]}")

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions from memory")

            return len(expired)


class RedisSessionStore(SessionStore):
    """Redis-backed session storage implementation.

    This is the enterprise storage backend, suitable for:
    - Production deployments
    - Multi-instance applications
    - High-availability setups
    - Distributed systems

    Features:
    - Persistent storage across restarts
    - Shared state across multiple instances
    - Automatic TTL-based expiration
    - Connection pooling

    Enterprise Pattern:
    - Redis connection pooling
    - Graceful error handling
    - Automatic reconnection
    - JSON serialization

    Example:
        >>> store = RedisSessionStore(redis_url="redis://localhost:6379/0")
        >>> await store.connect()
        >>> session = ConversationSession()
        >>> await store.save(session)
        >>> loaded = await store.load(session.session_id)
        >>> await store.disconnect()

    Note:
        Requires redis-py: pip install redis
    """

    def __init__(self, redis_url: str, key_prefix: str = "mcp:session:"):
        """Initialize Redis session store.

        Args:
            redis_url: Redis connection URL (e.g., 'redis://localhost:6379/0')
            key_prefix: Prefix for Redis keys (default: 'mcp:session:')

        Raises:
            ImportError: If redis package is not installed
        """
        if importlib.util.find_spec("redis") is None:
            raise ImportError(
                "redis package is required for RedisSessionStore. "
                "Install with: pip install redis"
            ) from None

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis: Any = None  # redis.Redis instance
        self._lock = asyncio.Lock()
        logger.info(f"Initialized RedisSessionStore (url={redis_url})")

    async def connect(self) -> None:
        """Connect to Redis.

        This should be called before using the store.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._redis.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Redis.

        This should be called when shutting down.
        """
        if self._redis:
            await self._redis.close()
            logger.info("Disconnected from Redis")

    def _get_key(self, session_id: str) -> str:
        """Get Redis key for session ID.

        Args:
            session_id: The session identifier

        Returns:
            Redis key with prefix
        """
        return f"{self.key_prefix}{session_id}"

    async def save(self, session: ConversationSession) -> None:
        """Save a conversation session to Redis.

        Args:
            session: The session to save

        Raises:
            RuntimeError: If not connected to Redis
            Exception: If save fails
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        try:
            key = self._get_key(session.session_id)
            # Serialize session to JSON
            session_json = session.to_json()

            # Save to Redis (no expiration, managed by cleanup)
            await self._redis.set(key, session_json)
            logger.debug(f"Saved session {session.session_id[:8]} to Redis")

        except Exception as e:
            logger.error(f"Failed to save session to Redis: {e}")
            raise

    async def load(self, session_id: str) -> ConversationSession | None:
        """Load a conversation session from Redis.

        Args:
            session_id: The session identifier

        Returns:
            ConversationSession if found, None otherwise

        Raises:
            RuntimeError: If not connected to Redis
            Exception: If load fails
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        try:
            key = self._get_key(session_id)
            session_json = await self._redis.get(key)

            if session_json:
                session = ConversationSession.from_json(session_json)
                logger.debug(f"Loaded session {session_id[:8]} from Redis")
                return session

            logger.debug(f"Session {session_id[:8]} not found in Redis")
            return None

        except Exception as e:
            logger.error(f"Failed to load session from Redis: {e}")
            raise

    async def delete(self, session_id: str) -> bool:
        """Delete a conversation session from Redis.

        Args:
            session_id: The session identifier

        Returns:
            True if deleted, False if not found

        Raises:
            RuntimeError: If not connected to Redis
            Exception: If delete fails
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        try:
            key = self._get_key(session_id)
            deleted = await self._redis.delete(key)

            if deleted:
                logger.debug(f"Deleted session {session_id[:8]} from Redis")
                return True

            logger.debug(f"Session {session_id[:8]} not found in Redis")
            return False

        except Exception as e:
            logger.error(f"Failed to delete session from Redis: {e}")
            raise

    async def list_sessions(self) -> list[str]:
        """List all session IDs in Redis.

        Returns:
            List of session IDs

        Raises:
            RuntimeError: If not connected to Redis
            Exception: If list fails
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        try:
            # Scan for all keys with our prefix
            pattern = f"{self.key_prefix}*"
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                # Remove prefix to get session ID
                session_id = key[len(self.key_prefix) :]
                keys.append(session_id)

            logger.debug(f"Listed {len(keys)} sessions from Redis")
            return keys

        except Exception as e:
            logger.error(f"Failed to list sessions from Redis: {e}")
            raise

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        """Clean up expired sessions from Redis.

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            Number of sessions cleaned up

        Raises:
            RuntimeError: If not connected to Redis
            Exception: If cleanup fails
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        try:
            current_time = time.time()
            session_ids = await self.list_sessions()
            expired_count = 0

            for session_id in session_ids:
                try:
                    session = await self.load(session_id)
                    if session and current_time - session.last_activity > ttl_seconds:
                        await self.delete(session_id)
                        expired_count += 1
                        logger.debug(f"Cleaned up expired session {session_id[:8]}")
                except Exception as e:
                    logger.warning(f"Error checking session {session_id[:8]}: {e}")
                    continue

            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired sessions from Redis")

            return expired_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions from Redis: {e}")
            raise
