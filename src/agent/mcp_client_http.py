"""HTTP-based MCP client implementation.

This module provides a client for connecting to MCP servers via HTTP,
suitable for remote server communication and production deployments.
"""

import asyncio
import logging
import time

from llama_index.tools.mcp import BasicMCPClient

from .mcp_client_base import MCPClientBase

logger = logging.getLogger(__name__)


class MCPClientHttp(MCPClientBase):
    """MCP client using HTTP transport with SSE support.

    This client connects to an MCP server via HTTP with Server-Sent Events (SSE),
    suitable for remote servers and production deployments.

    Attributes:
        host: MCP server host address
        port: MCP server port
        use_sse: Whether to use SSE transport (default: True)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        use_sse: bool = True,
        connection_timeout: int = 30,
        request_timeout: int = 60,
    ):
        """Initialize HTTP MCP client.

        Args:
            host: MCP server host (default: 127.0.0.1)
            port: MCP server port (default: 8000)
            use_sse: Use SSE transport (default: True)
            connection_timeout: Timeout for connection (seconds)
            request_timeout: Timeout for requests (seconds)

        Raises:
            ValueError: If host or port are invalid
        """
        super().__init__(connection_timeout, request_timeout)

        if not host:
            raise ValueError("Host cannot be empty")
        if not (1024 <= port <= 65535):
            raise ValueError(f"Port must be between 1024 and 65535, got {port}")

        self.host = host
        self.port = port
        self.use_sse = use_sse

        logger.info(f"Initialized HTTP MCP client for {host}:{port} " f"(SSE: {use_sse})")

    def _get_server_url(self) -> str:
        """Get the MCP server URL.

        Returns:
            The server URL
        """
        protocol = "https" if self.host != "127.0.0.1" else "http"
        endpoint = "/sse" if self.use_sse else "/mcp"
        return f"{protocol}://{self.host}:{self.port}{endpoint}"

    def _build_client(self) -> BasicMCPClient:
        """Build BasicMCPClient for HTTP transport.

        Returns:
            Configured BasicMCPClient for HTTP communication
        """
        server_url = self._get_server_url()
        logger.info(f"Building HTTP MCP client for: {server_url}")

        client = BasicMCPClient(server_url)

        logger.debug(f"HTTP client built for URL: {server_url}")
        return client

    async def connect(self) -> None:
        """Establish connection to MCP server via HTTP.

        This method initializes the BasicMCPClient and establishes
        communication with the HTTP server.

        Raises:
            ConnectionError: If connection fails
        """
        async with self._connection_lock:
            if self._is_connected:
                logger.debug("Already connected to MCP server")
                return

            try:
                server_url = self._get_server_url()
                logger.info(f"Connecting to MCP server via HTTP: {server_url}")

                # Build the client
                self.client = self._build_client()

                # Verify connection by attempting to get tools
                logger.debug("Verifying HTTP connection...")
                tools = await asyncio.wait_for(
                    self._verify_connection(),
                    timeout=self.connection_timeout,
                )

                logger.info(f"Successfully connected to MCP server (found {len(tools)} tools)")
                self._is_connected = True
                # Record connection time to prevent false positive health checks
                self._connection_time = time.time()

            except TimeoutError as e:
                logger.error(
                    f"Connection timeout after {self.connection_timeout}s. "
                    f"Ensure MCP server is running at {self._get_server_url()}"
                )
                raise ConnectionError(
                    f"Failed to connect to MCP server within {self.connection_timeout}s"
                ) from e
            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {e}")
                raise ConnectionError(f"MCP server connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to MCP server.

        This method cleans up the client connection.
        """
        async with self._connection_lock:
            if not self._is_connected:
                logger.debug("Not connected, skipping disconnect")
                return

            try:
                logger.info("Disconnecting from MCP server")
                # BasicMCPClient doesn't require explicit cleanup for HTTP
                self.client = None
                self._is_connected = False
                self._connection_time = None  # Clear connection timestamp
                logger.info("Successfully disconnected from MCP server")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                self._is_connected = False
                self._connection_time = None  # Clear connection timestamp

    async def _verify_connection(self) -> list:
        """Verify connection by attempting to retrieve tools.

        Returns:
            List of available tools

        Raises:
            RuntimeError: If verification fails
        """
        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            from llama_index.tools.mcp import McpToolSpec

            mcp_tool_spec = McpToolSpec(client=self.client)
            tools = await mcp_tool_spec.to_tool_list_async()
            return tools
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify MCP server connection: {e}") from e
