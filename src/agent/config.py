"""Agent configuration management.

This module provides configuration for the MCP agent orchestrator,
loading settings from the centralized config manager.
"""

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.mcp_client.config import MCPTransport, mcp_client_config

logger = logging.getLogger(__name__)


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

    # ========================================================================
    # Session Management Configuration (LlamaIndex ChatStore)
    # ========================================================================

    session_persistence_type: str = Field(
        default="in_memory",
        description="Session persistence: 'in_memory' (SimpleChatStore) or 'redis' (RedisChatStore)",
    )

    session_redis_url: str | None = Field(
        default=None,
        description="Redis URL for session persistence (e.g., 'redis://localhost:6379/0')",
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

    def validate_configuration(self) -> None:
        """Validate agent configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        logger.info("Validating agent configuration...")

        # Validate agent-specific settings
        if self.agent_type.lower() not in ["function", "react"]:
            raise ValueError(f"agent_type must be 'function' or 'react', got '{self.agent_type}'")

        if self.session_persistence_type.lower() not in ["in_memory", "redis"]:
            raise ValueError(
                f"session_persistence_type must be 'in_memory' or 'redis', "
                f"got '{self.session_persistence_type}'"
            )

        # Validate Redis settings if Redis persistence is enabled
        if self.session_persistence_type.lower() == "redis" and not self.session_redis_url:
            logger.warning(
                "session_persistence_type is 'redis' but session_redis_url is not set. "
                "Falling back to in-memory storage."
            )

        logger.info("Agent configuration validation passed")

    @property
    def mcp_transport(self) -> MCPTransport:
        """Get MCP transport from MCP client config."""
        return mcp_client_config.mcp_transport

    @property
    def mcp_server_path(self) -> str:
        """Get MCP server path from MCP client config."""
        return mcp_client_config.mcp_server_path

    @property
    def mcp_server_host(self) -> str:
        """Get MCP server host from MCP client config."""
        return mcp_client_config.mcp_server_host

    @property
    def mcp_server_port(self) -> int:
        """Get MCP server port from MCP client config."""
        return mcp_client_config.mcp_server_port

    @property
    def mcp_connection_timeout(self) -> int:
        """Get MCP connection timeout from MCP client config."""
        return mcp_client_config.mcp_connection_timeout

    @property
    def mcp_request_timeout(self) -> int:
        """Get MCP request timeout from MCP client config."""
        return mcp_client_config.mcp_request_timeout


# Global agent config instance
agent_config = AgentConfig()
