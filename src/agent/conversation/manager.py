"""Conversation manager for session lifecycle management.

This module provides the ConversationManager class, which orchestrates:
- Session creation and retrieval
- Conversation history management
- Automatic session cleanup
- Integration with storage backends

Enterprise Pattern:
- Centralized session management
- Background cleanup tasks
- Thread-safe operations
- Pluggable storage backends
"""

import asyncio
import logging
import uuid
from typing import Any

from ..config import agent_config
from .persistence import InMemorySessionStore, RedisSessionStore, SessionStore
from .session import ConversationSession

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation sessions with automatic cleanup.

    This is the central component for conversation management. It handles:
    - Session lifecycle (create, retrieve, delete)
    - Conversation history tracking
    - Automatic cleanup of expired sessions
    - Integration with storage backends (in-memory or Redis)

    Enterprise Pattern:
    - Factory pattern for storage backend creation
    - Background task for automatic cleanup
    - Thread-safe session access
    - Graceful error handling

    Example:
        >>> manager = ConversationManager()
        >>> await manager.initialize()
        >>> session = await manager.create_session()
        >>> await manager.chat(session.session_id, "Find patients with diabetes")
        >>> await manager.shutdown()

    Attributes:
        store: Session storage backend (in-memory or Redis)
        cleanup_task: Background task for session cleanup
    """

    def __init__(
        self,
        session_store: SessionStore | None = None,
        ttl_seconds: int | None = None,
        cleanup_interval: int | None = None,
        max_history_messages: int | None = None,
    ):
        """Initialize conversation manager.

        Args:
            session_store: Optional custom session store (auto-created if None)
            ttl_seconds: Session TTL in seconds (from config if None)
            cleanup_interval: Cleanup interval in seconds (from config if None)
            max_history_messages: Max messages in history (from config if None)

        Note:
            If session_store is None, it will be created during initialize()
            based on agent_config.session_persistence_type
        """
        self.store = session_store
        self.ttl_seconds = ttl_seconds or agent_config.session_ttl_seconds
        self.cleanup_interval = cleanup_interval or agent_config.session_cleanup_interval_seconds
        self.max_history_messages = (
            max_history_messages or agent_config.session_max_history_messages
        )

        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._shutdown_event.set()  # Initially not shutting down
        self._initialized = False

        logger.info(
            f"Initialized ConversationManager "
            f"(ttl={self.ttl_seconds}s, cleanup_interval={self.cleanup_interval}s, "
            f"max_history={self.max_history_messages})"
        )

    async def initialize(self) -> None:
        """Initialize the conversation manager.

        This creates the storage backend and starts the cleanup task.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.debug("ConversationManager already initialized")
            return

        try:
            logger.info("Initializing ConversationManager...")

            # Create storage backend if not provided
            if self.store is None:
                self.store = await self._create_storage_backend()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._initialized = True
            logger.info("ConversationManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ConversationManager: {e}")
            raise RuntimeError(f"Conversation manager initialization failed: {e}") from e

    async def _create_storage_backend(self) -> SessionStore:
        """Create storage backend based on configuration.

        Returns:
            SessionStore instance (InMemory or Redis)

        Raises:
            ValueError: If persistence type is invalid
        """
        persistence_type = agent_config.session_persistence_type

        if persistence_type == "in_memory":
            logger.info("Creating InMemorySessionStore")
            return InMemorySessionStore()

        elif persistence_type == "redis":
            redis_url = agent_config.session_redis_url
            if not redis_url:
                raise ValueError(
                    "session_redis_url is required for Redis persistence. "
                    "Set SESSION_REDIS_URL environment variable."
                )

            logger.info(f"Creating RedisSessionStore (url={redis_url})")
            store = RedisSessionStore(redis_url=redis_url)
            await store.connect()
            return store

        else:
            raise ValueError(
                f"Invalid session_persistence_type: {persistence_type}. "
                f"Must be 'in_memory' or 'redis'."
            )

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up expired sessions.

        This runs continuously until shutdown, cleaning up expired sessions
        at regular intervals based on cleanup_interval.
        """
        logger.info(
            f"Started session cleanup loop (interval: {self.cleanup_interval}s, "
            f"ttl: {self.ttl_seconds}s)"
        )

        while self._shutdown_event.is_set():
            try:
                # Wait for cleanup interval or shutdown
                await asyncio.sleep(self.cleanup_interval)

                # Check if shutdown was requested during sleep
                if not self._shutdown_event.is_set():
                    break

                # Cleanup expired sessions
                logger.debug("Running session cleanup...")
                cleaned_count = await self.store.cleanup_expired(self.ttl_seconds)

                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} expired sessions")

            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                # Continue running despite errors
                await asyncio.sleep(5)

        logger.info("Session cleanup loop stopped")

    async def create_session(
        self, session_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> ConversationSession:
        """Create a new conversation session.

        Args:
            session_id: Optional custom session ID (auto-generated if None)
            metadata: Optional session metadata (user_id, context, etc.)

        Returns:
            New ConversationSession instance

        Raises:
            RuntimeError: If manager not initialized

        Example:
            >>> session = await manager.create_session(
            ...     metadata={"user_id": "user123", "department": "clinical"}
            ... )
        """
        if not self._initialized:
            raise RuntimeError("ConversationManager not initialized. Call initialize() first.")

        if session_id is None:
            session_id = str(uuid.uuid4())

        # Check if session already exists
        existing = await self.store.load(session_id)
        if existing:
            logger.warning(f"Session {session_id[:8]} already exists, returning existing")
            return existing

        # Create new session
        session = ConversationSession(session_id=session_id, metadata=metadata or {})
        await self.store.save(session)

        logger.info(f"Created new conversation session: {session_id[:8]}")
        return session

    async def get_session(self, session_id: str) -> ConversationSession | None:
        """Get an existing conversation session.

        Args:
            session_id: Session identifier

        Returns:
            ConversationSession if found, None otherwise

        Raises:
            RuntimeError: If manager not initialized

        Example:
            >>> session = await manager.get_session(session_id)
            >>> if session:
            ...     print(f"Found session with {len(session.messages)} messages")
        """
        if not self._initialized:
            raise RuntimeError("ConversationManager not initialized. Call initialize() first.")

        session = await self.store.load(session_id)

        if session:
            # Check if session has expired
            if session.is_expired(self.ttl_seconds):
                logger.info(f"Session {session_id[:8]} expired, removing")
                await self.store.delete(session_id)
                return None

        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found

        Raises:
            RuntimeError: If manager not initialized

        Example:
            >>> deleted = await manager.delete_session(session_id)
            >>> if deleted:
            ...     print("Session deleted successfully")
        """
        if not self._initialized:
            raise RuntimeError("ConversationManager not initialized. Call initialize() first.")

        deleted = await self.store.delete(session_id)
        if deleted:
            logger.info(f"Deleted session {session_id[:8]}")
        return deleted

    async def list_sessions(self) -> list[str]:
        """List all session IDs.

        Returns:
            List of session IDs

        Raises:
            RuntimeError: If manager not initialized

        Example:
            >>> session_ids = await manager.list_sessions()
            >>> print(f"Found {len(session_ids)} active sessions")
        """
        if not self._initialized:
            raise RuntimeError("ConversationManager not initialized. Call initialize() first.")

        return await self.store.list_sessions()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to a session.

        Args:
            session_id: Session identifier
            role: Message role ('user', 'assistant', 'system', 'tool')
            content: Message content
            metadata: Optional metadata (tool calls, results, etc.)

        Raises:
            ValueError: If session not found
            RuntimeError: If manager not initialized

        Example:
            >>> await manager.add_message(
            ...     session_id,
            ...     "user",
            ...     "Find patients with diabetes"
            ... )
        """
        if not self._initialized:
            raise RuntimeError("ConversationManager not initialized. Call initialize() first.")

        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.add_message(role, content, metadata)
        await self.store.save(session)

    async def get_conversation_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get summary of a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with conversation statistics, or None if not found

        Raises:
            RuntimeError: If manager not initialized

        Example:
            >>> summary = await manager.get_conversation_summary(session_id)
            >>> if summary:
            ...     print(f"Messages: {summary['message_count']}")
        """
        if not self._initialized:
            raise RuntimeError("ConversationManager not initialized. Call initialize() first.")

        session = await self.get_session(session_id)
        if not session:
            return None

        return session.get_conversation_summary()

    async def shutdown(self) -> None:
        """Shutdown the conversation manager.

        This stops the cleanup task and disconnects from storage backends.
        """
        if not self._initialized:
            logger.debug("ConversationManager not initialized, skipping shutdown")
            return

        try:
            logger.info("Shutting down ConversationManager...")

            # Signal shutdown to cleanup loop
            self._shutdown_event.clear()

            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Disconnect from Redis if applicable
            if isinstance(self.store, RedisSessionStore):
                await self.store.disconnect()

            self._initialized = False
            logger.info("ConversationManager shutdown complete")

        except Exception as e:
            logger.error(f"Error during ConversationManager shutdown: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
