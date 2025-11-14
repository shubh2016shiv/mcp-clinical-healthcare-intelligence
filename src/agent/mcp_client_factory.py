"""Factory for creating MCP client instances.

This module provides a factory pattern implementation for creating
the appropriate MCP client based on configuration.
"""

import logging

from .config import MCPTransport, agent_config
from .mcp_client_base import MCPClientBase
from .mcp_client_http import MCPClientHttp
from .mcp_client_stdio import MCPClientStdio

logger = logging.getLogger(__name__)


class MCPClientFactory:
    """Factory for creating MCP client instances.

    This factory provides a centralized way to create MCP clients
    based on the configured transport mechanism.
    """

    @staticmethod
    def create_client(
        transport: MCPTransport | None = None,
        **kwargs,
    ) -> MCPClientBase:
        """Create an MCP client based on transport type.

        Args:
            transport: Transport type (STDIO or HTTP). If None, uses agent_config.
            **kwargs: Additional arguments to pass to the client constructor

        Returns:
            Configured MCPClientBase instance

        Raises:
            ValueError: If transport type is unknown
            FileNotFoundError: If STDIO server path doesn't exist

        Example:
            >>> client = MCPClientFactory.create_client(MCPTransport.STDIO)
            >>> await client.connect()
        """
        if transport is None:
            transport = agent_config.mcp_transport

        logger.info(f"Creating MCP client with transport: {transport.value}")

        if transport == MCPTransport.STDIO:
            return MCPClientFactory._create_stdio_client(**kwargs)
        elif transport == MCPTransport.HTTP:
            return MCPClientFactory._create_http_client(**kwargs)
        else:
            raise ValueError(f"Unknown transport type: {transport}")

    @staticmethod
    def _create_stdio_client(**kwargs) -> MCPClientStdio:
        """Create a STDIO MCP client.

        Args:
            **kwargs: Arguments for MCPClientStdio constructor
                - server_path: Path to MCP server script
                - server_args: Additional server arguments
                - connection_timeout: Connection timeout in seconds
                - request_timeout: Request timeout in seconds

        Returns:
            Configured MCPClientStdio instance
        """
        # Use config defaults if not provided
        server_path = kwargs.pop("server_path", agent_config.mcp_server_path)
        server_args = kwargs.pop("server_args", None)
        connection_timeout = kwargs.pop("connection_timeout", agent_config.mcp_connection_timeout)
        request_timeout = kwargs.pop("request_timeout", agent_config.mcp_request_timeout)

        logger.debug(
            f"Creating STDIO client: server_path={server_path}, "
            f"connection_timeout={connection_timeout}s, "
            f"request_timeout={request_timeout}s"
        )

        return MCPClientStdio(
            server_path=server_path,
            server_args=server_args,
            connection_timeout=connection_timeout,
            request_timeout=request_timeout,
        )

    @staticmethod
    def _create_http_client(**kwargs) -> MCPClientHttp:
        """Create an HTTP MCP client.

        Args:
            **kwargs: Arguments for MCPClientHttp constructor
                - host: Server host address
                - port: Server port
                - use_sse: Use SSE transport
                - connection_timeout: Connection timeout in seconds
                - request_timeout: Request timeout in seconds

        Returns:
            Configured MCPClientHttp instance
        """
        # Use config defaults if not provided
        host = kwargs.pop("host", agent_config.mcp_server_host)
        port = kwargs.pop("port", agent_config.mcp_server_port)
        use_sse = kwargs.pop("use_sse", True)
        connection_timeout = kwargs.pop("connection_timeout", agent_config.mcp_connection_timeout)
        request_timeout = kwargs.pop("request_timeout", agent_config.mcp_request_timeout)

        logger.debug(
            f"Creating HTTP client: host={host}, port={port}, use_sse={use_sse}, "
            f"connection_timeout={connection_timeout}s, "
            f"request_timeout={request_timeout}s"
        )

        return MCPClientHttp(
            host=host,
            port=port,
            use_sse=use_sse,
            connection_timeout=connection_timeout,
            request_timeout=request_timeout,
        )

    @staticmethod
    def create_from_config() -> MCPClientBase:
        """Create an MCP client from agent configuration.

        Returns:
            Configured MCPClientBase instance based on agent_config

        Example:
            >>> client = MCPClientFactory.create_from_config()
            >>> await client.connect()
        """
        logger.info("Creating MCP client from agent configuration")
        return MCPClientFactory.create_client(transport=agent_config.mcp_transport)
