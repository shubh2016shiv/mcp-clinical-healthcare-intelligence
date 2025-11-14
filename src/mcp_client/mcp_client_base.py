"""Abstract base class for MCP client implementations.

This module defines the abstract interface for MCP clients, allowing
different transport implementations (STDIO, HTTP) while maintaining
a consistent API.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from llama_index.tools.mcp import BasicMCPClient

logger = logging.getLogger(__name__)


class MCPClientBase(ABC):
    """Abstract base class for MCP client implementations.

    This class defines the interface that all MCP client implementations
    must follow, ensuring consistent behavior across different transport
    mechanisms (STDIO, HTTP, SSE).

    Attributes:
        client: The underlying BasicMCPClient instance
        connection_timeout: Timeout for establishing connection
        request_timeout: Timeout for individual requests
    """

    def __init__(
        self,
        connection_timeout: int = 30,
        request_timeout: int = 60,
    ):
        """Initialize the MCP client base.

        Args:
            connection_timeout: Timeout for establishing connection (seconds)
            request_timeout: Timeout for individual requests (seconds)
        """
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.client: BasicMCPClient | None = None
        self._is_connected = False
        self._connection_lock = asyncio.Lock()  # Protect connection state
        self._connection_time: float | None = None  # Timestamp when connection was established

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to MCP server.

        This method should be implemented by subclasses to handle
        transport-specific connection logic.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server.

        This method should be implemented by subclasses to handle
        transport-specific disconnection logic.
        """
        pass

    @abstractmethod
    def _build_client(self) -> BasicMCPClient:
        """Build and return the BasicMCPClient instance.

        This method should be implemented by subclasses to create
        the appropriate BasicMCPClient for their transport mechanism.

        Returns:
            Configured BasicMCPClient instance
        """
        pass

    async def _is_connection_alive(self) -> bool:
        """Check if the current connection is alive.

        This method attempts to verify the connection is still active.
        It tries ping first, but falls back to a simple tool list check
        if ping is not available or fails.

        Returns:
            True if connection is alive, False otherwise
        """
        if not self._is_connected or self.client is None:
            return False

        try:
            # Try lightweight ping first
            try:
                await asyncio.wait_for(
                    self.client.send_ping(),
                    timeout=5.0,  # Short timeout for health check
                )
                return True
            except (AttributeError, RuntimeError, TypeError) as ping_error:
                # Ping failed - BasicMCPClient might not support ping in this state
                # or might require session context. Fall back to checking if we can
                # at least access the client object (connection exists)
                logger.debug(
                    f"Ping check failed ({type(ping_error).__name__}), "
                    "assuming connection is alive if client exists"
                )
                # If client exists and is connected, assume it's alive
                # The actual health check will verify with get_tools() if needed
                return self.client is not None
        except TimeoutError:
            logger.debug("Connection ping timeout, connection appears stale")
            return False
        except Exception as e:
            logger.debug(f"Connection check failed: {e}, connection appears stale")
            return False

    async def ensure_connected(self) -> None:
        """Ensure client is connected, connecting if necessary.

        This method provides a convenient way to ensure the client
        is ready before making requests. It will automatically reconnect
        if the current connection appears stale.

        IMPORTANT: This method does NOT check connection health if we just
        connected (within last 3 seconds). This prevents false positives
        immediately after establishing a connection.

        Raises:
            ConnectionError: If connection fails
        """
        async with self._connection_lock:
            # If not connected, connect
            if not self._is_connected:
                await self.connect()
                return

            # If connected, check if connection is still alive
            # BUT: Skip health check if connection was just established
            # (within last 3 seconds) to avoid false positives
            # This prevents reconnection attempts immediately after successful connection
            current_time = time.time()
            connection_age = current_time - (self._connection_time or 0)
            HEALTH_CHECK_GRACE_PERIOD = 3.0  # Don't check health for 3 seconds after connection

            if connection_age < HEALTH_CHECK_GRACE_PERIOD:
                # Connection was just established, skip health check
                logger.debug(
                    f"Skipping health check - connection established {connection_age:.2f}s ago "
                    f"(grace period: {HEALTH_CHECK_GRACE_PERIOD}s)"
                )
                return

            # Connection has been established for a while, check if it's still alive
            if not await self._is_connection_alive():
                logger.info("Detected stale connection, attempting reconnection...")
                try:
                    # Disconnect first to clean up
                    # Use safe cleanup that doesn't require context manager state
                    if self.client is not None:
                        try:
                            # Try to clean up using context manager protocol if available
                            # But handle gracefully if client isn't in context manager state
                            if hasattr(self.client, "_exit_stack"):
                                await self.client.__aexit__(None, None, None)
                            else:
                                # BasicMCPClient might not be in context manager state
                                # Just reset the client reference
                                logger.debug(
                                    "Client not in context manager state, skipping __aexit__"
                                )
                        except AttributeError:
                            # _exit_stack doesn't exist - client wasn't used as context manager
                            logger.debug(
                                "Client cleanup: no _exit_stack attribute (not in context manager)"
                            )
                        except Exception as cleanup_error:
                            logger.warning(f"Error during client cleanup: {cleanup_error}")
                    self.client = None
                    self._is_connected = False
                except Exception as e:
                    logger.warning(f"Error during stale connection cleanup: {e}")

                # Reconnect
                await self.connect()

    @property
    def is_connected(self) -> bool:
        """Check if client is connected.

        Returns:
            True if connected, False otherwise
        """
        return self._is_connected

    async def get_tools(self) -> list[Any]:
        """Get available tools from MCP server.

        Returns:
            List of available tools

        Raises:
            ConnectionError: If not connected
            RuntimeError: If tool retrieval fails
        """
        await self.ensure_connected()

        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            logger.info("Retrieving tools from MCP server...")
            from llama_index.tools.mcp import McpToolSpec

            mcp_tool_spec = McpToolSpec(client=self.client)
            tools = await mcp_tool_spec.to_tool_list_async()
            logger.info(f"Successfully retrieved {len(tools)} tools from MCP server")
            return tools
        except Exception as e:
            logger.error(f"Failed to retrieve tools from MCP server: {e}")
            raise RuntimeError(f"Tool retrieval failed: {e}") from e

    async def get_resources(self) -> list[Any]:
        """Get available resources from MCP server.

        Returns:
            List of available resources

        Raises:
            ConnectionError: If not connected
            RuntimeError: If resource retrieval fails
        """
        await self.ensure_connected()

        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            logger.info("Retrieving resources from MCP server...")
            from llama_index.tools.mcp import McpToolSpec

            mcp_tool_spec = McpToolSpec(client=self.client)
            resources = await mcp_tool_spec.to_resources_list_async()
            logger.info(f"Successfully retrieved {len(resources)} resources from MCP server")
            return resources
        except Exception as e:
            logger.error(f"Failed to retrieve resources from MCP server: {e}")
            raise RuntimeError(f"Resource retrieval failed: {e}") from e

    async def health_check(self) -> bool:
        """Perform a health check on the MCP server.

        This method attempts to verify the server is healthy by trying to
        retrieve tools. If ping is available and works, it uses that for
        a lighter check. Otherwise, it falls back to getting tools.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            await self.ensure_connected()

            if self.client is None:
                logger.warning("Client not initialized")
                return False

            # Try lightweight ping first, but fall back to get_tools if ping fails
            # This is because BasicMCPClient's send_ping() might require session state
            # that isn't always available
            try:
                # Attempt ping - this is faster if it works
                await self.client.send_ping()
                logger.debug("MCP server health check passed (ping)")
                return True
            except (AttributeError, RuntimeError) as ping_error:
                # Ping failed - likely because client isn't in session state
                # Fall back to getting tools as a health check
                logger.debug(
                    f"Ping failed ({ping_error}), falling back to get_tools() for health check"
                )
                # Use get_tools as a more reliable health check
                # This actually communicates with the server
                tools = await self.get_tools()
                logger.debug(f"MCP server health check passed (retrieved {len(tools)} tools)")
                return True

        except Exception as e:
            logger.error(f"MCP server health check failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
