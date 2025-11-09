#!/usr/bin/env python3
"""
MongoDB Connection Manager - Enterprise Edition

Provides singleton MongoDB connection management with connection pooling,
health checks, auto-reconnection, and resource lifecycle management.
Optimized for high-throughput ETL pipelines with connection reuse.

Features:
- Singleton pattern for connection reuse across pipeline stages
- Configurable connection pool sizes
- Automatic health checks and reconnection
- Context manager support for resource cleanup
- Thread-safe operations
- Connection metrics and monitoring
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "pydantic is required for connection management. Install with: pip install pydantic"
    ) from None

from pymongo import MongoClient
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    ServerSelectionTimeoutError,
)

logger = logging.getLogger(__name__)


class ConnectionConfig(BaseModel):
    """Configuration class for MongoDB connections."""

    host: str = Field(default="localhost", description="MongoDB host")
    port: int = Field(default=27017, description="MongoDB port")
    user: str = Field(default="admin", description="MongoDB username")
    password: str = Field(default="mongopass123", description="MongoDB password")
    db_name: str = Field(default="fhir_db", description="Default database name")
    auth_source: str = Field(default="admin", description="Authentication database")
    max_pool_size: int = Field(default=10, description="Maximum connection pool size", ge=1)
    min_pool_size: int = Field(default=2, description="Minimum connection pool size", ge=1)
    max_idle_time_ms: int = Field(
        default=30000, description="Maximum idle time for connections", ge=0
    )
    server_selection_timeout_ms: int = Field(
        default=30000, description="Server selection timeout", ge=0
    )
    connect_timeout_ms: int = Field(default=20000, description="Initial connection timeout", ge=0)
    socket_timeout_ms: int = Field(default=20000, description="Socket operation timeout", ge=0)
    retry_writes: bool = Field(default=True, description="Enable write retries")
    retry_reads: bool = Field(default=True, description="Enable read retries")
    heartbeat_frequency_ms: int = Field(default=10000, description="Heartbeat frequency", ge=0)
    max_reconnect_attempts: int = Field(
        default=3, description="Maximum reconnection attempts", ge=0
    )
    reconnect_interval_ms: int = Field(
        default=1000, description="Interval between reconnection attempts", ge=0
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False  # Allow mutation for compatibility

    def get_connection_string(self) -> str:
        """Generate MongoDB connection string."""
        return (
            f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/"
            f"{self.db_name}?authSource={self.auth_source}"
        )

    def get_client_kwargs(self) -> dict[str, Any]:
        """Get MongoDB client keyword arguments."""
        return {
            "maxPoolSize": self.max_pool_size,
            "minPoolSize": self.min_pool_size,
            "maxIdleTimeMS": self.max_idle_time_ms,
            "serverSelectionTimeoutMS": self.server_selection_timeout_ms,
            "connectTimeoutMS": self.connect_timeout_ms,
            "socketTimeoutMS": self.socket_timeout_ms,
            "retryWrites": self.retry_writes,
            "retryReads": self.retry_reads,
            "heartbeatFrequencyMS": self.heartbeat_frequency_ms,
        }


class ConnectionMetrics:
    """Connection metrics tracking."""

    def __init__(self):
        self.connections_created = 0
        self.connections_closed = 0
        self.connection_errors = 0
        self.health_check_count = 0
        self.health_check_failures = 0
        self.reconnect_attempts = 0
        self.reconnect_successes = 0

    def reset(self):
        """Reset all metrics."""
        self.__init__()

    def get_stats(self) -> dict[str, int]:
        """Get current metrics."""
        return {
            "connections_created": self.connections_created,
            "connections_closed": self.connections_closed,
            "connection_errors": self.connection_errors,
            "health_check_count": self.health_check_count,
            "health_check_failures": self.health_check_failures,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_successes": self.reconnect_successes,
        }


class MongoDBConnectionManager:
    """Singleton MongoDB connection manager with pooling and health checks."""

    _instance: Optional["MongoDBConnectionManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MongoDBConnectionManager":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the connection manager."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._client: MongoClient | None = None
        self._config: ConnectionConfig | None = None
        self._connected = False
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        self._lock = threading.RLock()
        self._metrics = ConnectionMetrics()

        logger.info("[INFO] MongoDB Connection Manager initialized")

    def configure(self, config: ConnectionConfig) -> None:
        """Configure the connection manager.

        Args:
            config: Connection configuration
        """
        with self._lock:
            if self._config is not None and self._client is not None:
                logger.warning("[WARNING] Connection manager already configured and connected")
                return

            self._config = config
            logger.info(f"[INFO] Connection manager configured for {config.host}:{config.port}")

    def connect(self) -> bool:
        """Establish connection to MongoDB.

        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            if self._connected and self._client:
                return True

            if not self._config:
                logger.error("[ERROR] Connection manager not configured")
                return False

            try:
                connection_string = self._config.get_connection_string()
                client_kwargs = self._config.get_client_kwargs()

                logger.info("[INFO] Connecting to MongoDB...")
                self._client = MongoClient(connection_string, **client_kwargs)

                # Test connection
                self._client.admin.command("ping")
                self._connected = True
                self._metrics.connections_created += 1
                self._last_health_check = time.time()

                logger.info("[SUCCESS] MongoDB connection established")
                return True

            except Exception as e:
                self._metrics.connection_errors += 1
                logger.error(f"[ERROR] Failed to connect to MongoDB: {e}")
                self._client = None
                self._connected = False
                return False

    def disconnect(self) -> None:
        """Close MongoDB connection."""
        with self._lock:
            if self._client:
                try:
                    self._client.close()
                    self._metrics.connections_closed += 1
                    logger.info("[INFO] MongoDB connection closed")
                except Exception as e:
                    logger.warning(f"[WARNING] Error closing MongoDB connection: {e}")
                finally:
                    self._client = None
                    self._connected = False

    def is_connected(self) -> bool:
        """Check if currently connected to MongoDB.

        Returns:
            True if connected, False otherwise
        """
        with self._lock:
            return self._connected and self._client is not None

    def health_check(self) -> bool:
        """Perform health check on MongoDB connection.

        Returns:
            True if healthy, False otherwise
        """
        with self._lock:
            self._metrics.health_check_count += 1
            current_time = time.time()

            # Skip health check if done recently
            if current_time - self._last_health_check < self._health_check_interval:
                return self._connected

            if not self._client:
                self._connected = False
                return False

            try:
                # Perform a lightweight health check
                self._client.admin.command("ping")
                self._last_health_check = current_time
                return True

            except (
                ConnectionFailure,
                ServerSelectionTimeoutError,
                NetworkTimeout,
                AutoReconnect,
            ) as e:
                logger.warning(f"[WARNING] Health check failed: {e}")
                self._metrics.health_check_failures += 1
                self._connected = False

                # Attempt reconnection
                if self._attempt_reconnect():
                    return True

                return False

            except Exception as e:
                logger.error(f"[ERROR] Unexpected error during health check: {e}")
                self._metrics.health_check_failures += 1
                self._connected = False
                return False

    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to MongoDB.

        Returns:
            True if reconnection successful, False otherwise
        """
        if not self._config:
            return False

        for attempt in range(self._config.max_reconnect_attempts):
            self._metrics.reconnect_attempts += 1
            logger.info(
                f"[INFO] Reconnection attempt {attempt + 1}/{self._config.max_reconnect_attempts}"
            )

            try:
                # Close existing connection if any
                if self._client:
                    try:
                        self._client.close()
                    except Exception:
                        pass

                # Create new connection
                connection_string = self._config.get_connection_string()
                client_kwargs = self._config.get_client_kwargs()

                self._client = MongoClient(connection_string, **client_kwargs)
                self._client.admin.command("ping")

                self._connected = True
                self._metrics.reconnect_successes += 1
                self._last_health_check = time.time()

                logger.info("[SUCCESS] Reconnection successful")
                return True

            except Exception as e:
                logger.warning(f"[WARNING] Reconnection attempt {attempt + 1} failed: {e}")

                if attempt < self._config.max_reconnect_attempts - 1:
                    time.sleep(self._config.reconnect_interval_ms / 1000)

        logger.error("[ERROR] All reconnection attempts failed")
        return False

    def get_client(self) -> MongoClient:
        """Get MongoDB client instance.

        Returns:
            MongoDB client instance

        Raises:
            RuntimeError: If not connected or connection manager not configured
        """
        with self._lock:
            if not self._config:
                raise RuntimeError("Connection manager not configured")

            if not self.health_check():
                raise RuntimeError("MongoDB connection is not healthy")

            return self._client

    def get_database(self, db_name: str | None = None) -> Any:
        """Get MongoDB database instance.

        Args:
            db_name: Database name (uses config default if None)

        Returns:
            MongoDB database instance

        Raises:
            RuntimeError: If not connected or connection manager not configured
        """
        client = self.get_client()
        db_name = db_name or self._config.db_name
        return client[db_name]

    def get_metrics(self) -> dict[str, int]:
        """Get connection metrics.

        Returns:
            Dictionary of connection metrics
        """
        return self._metrics.get_stats()

    @contextmanager
    def connection_context(self, db_name: str | None = None):
        """Context manager for database connections.

        Args:
            db_name: Database name (uses config default if None)

        Yields:
            MongoDB database instance

        Raises:
            RuntimeError: If connection fails
        """
        try:
            db = self.get_database(db_name)
            yield db
        except Exception as e:
            logger.error(f"[ERROR] Connection context error: {e}")
            raise
        finally:
            # Health check after use
            self.health_check()


# Global instance
_connection_manager = MongoDBConnectionManager()


def get_connection_manager() -> MongoDBConnectionManager:
    """Get the global connection manager instance.

    Returns:
        MongoDB connection manager instance
    """
    return _connection_manager


def configure_connection(config: ConnectionConfig) -> None:
    """Configure the global connection manager.

    Args:
        config: Connection configuration
    """
    _connection_manager.configure(config)


def connect() -> bool:
    """Connect using the global connection manager.

    Returns:
        True if connection successful, False otherwise
    """
    return _connection_manager.connect()


def disconnect() -> None:
    """Disconnect using the global connection manager."""
    _connection_manager.disconnect()


def is_connected() -> bool:
    """Check if connected using the global connection manager.

    Returns:
        True if connected, False otherwise
    """
    return _connection_manager.is_connected()


def get_client() -> MongoClient:
    """Get MongoDB client using the global connection manager.

    Returns:
        MongoDB client instance
    """
    return _connection_manager.get_client()


def get_database(db_name: str | None = None) -> Any:
    """Get MongoDB database using the global connection manager.

    Args:
        db_name: Database name

    Returns:
        MongoDB database instance
    """
    return _connection_manager.get_database(db_name)


def get_connection_metrics() -> dict[str, int]:
    """Get connection metrics from the global connection manager.

    Returns:
        Dictionary of connection metrics
    """
    return _connection_manager.get_metrics()


@contextmanager
def database_connection(db_name: str | None = None):
    """Context manager for database connections using the global manager.

    Args:
        db_name: Database name

    Yields:
        MongoDB database instance
    """
    with _connection_manager.connection_context(db_name) as db:
        yield db
