"""Agent configuration management.

This module provides configuration for the MCP agent orchestrator,
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


class AgentConfig(BaseSettings):
    """Agent-specific configuration.

    This configuration extends the centralized settings with agent-specific
    parameters for MCP integration and LLM orchestration.

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
    # MCP Agent Configuration
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

    # ========================================================================
    # Agent Behavior Configuration
    # ========================================================================

    agent_max_iterations: int = Field(
        default=10,
        description="Maximum number of agent reasoning iterations",
        ge=1,
        le=50,
    )

    agent_verbose: bool = Field(
        default=True,
        description="Enable verbose logging for agent operations",
    )

    agent_system_prompt: str = Field(
        default=(
            "You are a healthcare data analysis assistant. Use the available MCP tools to query "
            "patient information, analyze medical conditions, review medication histories, "
            "and search drug information. Always provide context and explanations for your findings. "
            "Be mindful of patient privacy and data sensitivity."
        ),
        description="System prompt for the agent",
    )

    # ========================================================================
    # LLM Configuration
    # ========================================================================

    agent_llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for agent reasoning",
    )

    agent_llm_temperature: float = Field(
        default=0.7,
        description="Temperature for LLM sampling (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    agent_llm_max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens for LLM response (None for default)",
        ge=100,
    )

    # ========================================================================
    # Tool Loading Configuration
    # ========================================================================

    agent_load_all_tools: bool = Field(
        default=True,
        description="Load all available MCP tools on startup",
    )

    agent_tool_cache_enabled: bool = Field(
        default=True,
        description="Cache tool definitions to avoid repeated MCP calls",
    )

    agent_tool_cache_ttl: int = Field(
        default=3600,
        description="Tool cache TTL in seconds (1 hour default)",
        ge=60,
        le=86400,
    )

    # ========================================================================
    # Agent Type and Memory Configuration
    # ========================================================================

    agent_type: str = Field(
        default="function",
        description="Agent type: 'function' for stateless, 'react' for conversational with memory",
    )

    agent_memory_enabled: bool = Field(
        default=True,
        description="Enable memory for conversational agents (ReActAgent)",
    )

    agent_memory_token_limit: int = Field(
        default=4096,
        description="Token limit for agent's conversation memory",
        ge=1024,
        le=32000,
    )

    agent_max_iterations: int = Field(
        default=10,
        description="Maximum iterations for ReActAgent reasoning loop",
        ge=1,
        le=50,
    )

    # ========================================================================
    # Session Management Configuration
    # ========================================================================

    session_persistence_type: str = Field(
        default="in_memory",
        description="Session persistence: 'in_memory' (default) or 'redis' (enterprise)",
    )

    session_redis_url: str | None = Field(
        default=None,
        description="Redis URL for session persistence (e.g., 'redis://localhost:6379/0')",
    )

    session_ttl_seconds: int = Field(
        default=3600,
        description="Session time-to-live in seconds (1 hour default)",
        ge=60,
        le=86400,
    )

    session_cleanup_interval_seconds: int = Field(
        default=600,
        description="Interval for cleaning up expired sessions (10 minutes default)",
        ge=60,
        le=3600,
    )

    session_max_history_messages: int = Field(
        default=50,
        description="Maximum messages to keep in session history",
        ge=5,
        le=200,
    )

    # ========================================================================
    # Concurrency Configuration
    # ========================================================================

    agent_max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent MCP tool requests",
        ge=1,
        le=100,
    )

    agent_request_queue_size: int = Field(
        default=100,
        description="Maximum size of request queue",
        ge=10,
        le=1000,
    )

    def get_mcp_server_url(self) -> str:
        """Get the MCP server URL for HTTP transport.

        Returns:
            The MCP server URL

        Example:
            >>> config = AgentConfig()
            >>> config.get_mcp_server_url()
            'http://127.0.0.1:8000/sse'
        """
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}/sse"

    def validate_configuration(self) -> None:
        """Validate agent configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        logger.info("Validating agent configuration...")

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

        logger.info("Agent configuration validation passed")


# Global agent config instance
agent_config = AgentConfig()
