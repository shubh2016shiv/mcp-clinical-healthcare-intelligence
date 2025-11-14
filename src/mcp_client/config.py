"""MCP Client configuration management.

This module provides configuration for MCP client connections,
loading settings from the centralized config manager.
"""

import logging
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class MCPTransport(str, Enum):
    """MCP server transport protocol.

    Attributes:
        STDIO: Standard input/output transport (default, local process)
        HTTP: HTTP transport with SSE support
    """

    STDIO = "stdio"
    HTTP = "http"


class MCPClientConfig(BaseSettings):
    """MCP client-specific configuration.

    This configuration manages MCP server connection settings
    used by MCP client implementations.

    Settings are loaded from:
    1. Environment variables (highest priority)
    2. .env file (if present)
    3. Class defaults (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================================
    # MCP Transport Configuration
    # ========================================================================

    mcp_transport: MCPTransport = Field(
        default=MCPTransport.STDIO,
        description="Transport protocol for MCP server communication (stdio or http)",
    )

    mcp_server_path: str = Field(
        default="src/mcp_server/server.py",
        description="Path to MCP server script (used for STDIO transport)",
    )

    mcp_server_host: str = Field(
        default="127.0.0.1",
        description="MCP server host (used for HTTP transport)",
    )

    mcp_server_port: int = Field(
        default=8000,
        description="MCP server port (used for HTTP transport)",
        ge=1024,
        le=65535,
    )

    mcp_connection_timeout: int = Field(
        default=30,
        description="Timeout for MCP server connection in seconds",
        ge=5,
        le=300,
    )

    mcp_request_timeout: int = Field(
        default=60,
        description="Timeout for individual MCP tool requests in seconds",
        ge=10,
        le=600,
    )

    def get_mcp_server_url(self) -> str:
        """Get the MCP server URL for HTTP transport.

        Returns:
            The MCP server URL

        Example:
            >>> config = MCPClientConfig()
            >>> config.get_mcp_server_url()
            'http://127.0.0.1:8000/sse'
        """
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}/sse"

    def validate_configuration(self) -> None:
        """Validate MCP client configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        logger.info("Validating MCP client configuration...")

        # Validate transport-specific settings
        if self.mcp_transport == MCPTransport.STDIO:
            if not self.mcp_server_path:
                raise ValueError("mcp_server_path is required for STDIO transport")
        elif self.mcp_transport == MCPTransport.HTTP:
            if not self.mcp_server_host or self.mcp_server_port <= 0:
                raise ValueError(
                    "mcp_server_host and mcp_server_port are required for HTTP transport"
                )

        # Validate timeout settings
        if self.mcp_connection_timeout >= self.mcp_request_timeout:
            logger.warning(
                f"mcp_connection_timeout ({self.mcp_connection_timeout}s) >= "
                f"mcp_request_timeout ({self.mcp_request_timeout}s). "
                f"Consider adjusting these values."
            )

        logger.info("MCP client configuration validation passed")


# Global MCP client config instance
mcp_client_config = MCPClientConfig()
