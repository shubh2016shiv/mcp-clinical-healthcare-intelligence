"""Centralized configuration management using Pydantic Settings.

This module provides type-safe configuration with environment variable support,
validation, and clear defaults for all system components. It serves as the single
source of truth for application settings across all modules.

Configuration can be overridden via environment variables (e.g., MONGODB_URI)
and is validated at startup to ensure all required settings are correctly configured.

Example:
    Loading and validating settings:
    >>> from src.config.settings import settings
    >>> settings.validate_configuration()
    >>> print(settings.mongodb_database)
    'text_to_mongodb_demo'
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from src.mcp_server.security.config import SecurityConfig

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Enumeration of supported Large Language Model providers.

    This enum defines the available LLM backends that can be used for
    query complexity analysis and natural language processing.

    Attributes:
        OPENAI: OpenAI GPT models via OpenAI API
        ANTHROPIC: Anthropic Claude models via Anthropic API
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ToolLoadingStrategy(str, Enum):
    """Enumeration of tool loading strategies for the MCP agent.

    Different strategies optimize for different use cases:
    - ALL: Load all available tools (comprehensive but slower)
    - HIERARCHICAL: Load tools by tier (default, balanced approach)
    - ADAPTIVE: Load tools based on query analysis (most efficient)

    Attributes:
        ALL: Load all tools regardless of complexity
        HIERARCHICAL: Load tools in tiers based on anticipated complexity
        ADAPTIVE: Dynamically load tools based on query analysis
    """

    ALL = "all"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class Settings(BaseSettings):
    """Application settings loaded from environment variables with validation.

    This class centralizes all configuration for the Text-to-MongoDB application.
    It provides type-safe access to settings with automatic validation, defaults,
    and environment variable support.

    Settings are loaded from:
    1. Environment variables (highest priority)
    2. .env file (if present)
    3. Class defaults (lowest priority)

    All settings can be overridden via environment variables using uppercase names
    (e.g., MONGODB_URI=mongodb://custom:27017).

    Attributes:
        MongoDB Configuration settings for connection and pooling
        LLM Configuration settings for model selection and API keys
        Agent Configuration settings for behavior and limits
        Tool Loading Configuration for tool discovery strategy
        Safety Configuration for operation restrictions
        MCP Server Configuration for server endpoints
        Logging Configuration for log levels
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================================
    # MongoDB Configuration
    # ========================================================================

    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description=(
            "MongoDB connection URI with protocol, host, port, and authentication. "
            "Format: mongodb://[username:password@]host[:port][/database][?options]"
            "If mongodb_user and mongodb_password are provided, they will be used to construct the URI."
        ),
    )

    mongodb_user: str | None = Field(
        default=None,
        description="MongoDB username for authentication. If provided, will be used to construct mongodb_uri.",
    )

    mongodb_password: str | None = Field(
        default=None,
        description="MongoDB password for authentication. If provided, will be used to construct mongodb_uri.",
    )

    mongodb_auth_source: str = Field(
        default="admin",
        description="MongoDB authentication database (authSource). Defaults to 'admin'.",
    )

    mongodb_database: str = Field(
        default="text_to_mongodb_demo",
        description="Name of the MongoDB database to use for all operations",
    )

    mongodb_timeout: int = Field(
        default=30,
        description="Query timeout in seconds. Prevents long-running queries",
        ge=1,
        le=300,
    )

    mongodb_min_pool_size: int = Field(
        default=10,
        description="Minimum number of connections to maintain in the connection pool",
        ge=1,
    )

    mongodb_max_pool_size: int = Field(
        default=50,
        description="Maximum number of connections allowed in the connection pool",
        ge=1,
    )

    # ========================================================================
    # LLM Provider Configuration
    # ========================================================================

    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Primary LLM provider for query analysis and complexity scoring",
    )

    openai_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI services. Required if llm_provider is 'openai'",
    )

    anthropic_api_key: str | None = Field(
        default=None,
        description="API key for Anthropic Claude services. Required if llm_provider is 'anthropic'",
    )

    # ========================================================================
    # Model Selection
    # ========================================================================

    default_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "Model used for simple queries with low complexity scores. "
            "Should be a fast, cost-effective model"
        ),
    )

    complex_model: str = Field(
        default="gpt-4o",
        description=(
            "Model used for complex queries with high complexity scores. "
            "Should be a more capable model"
        ),
    )

    complexity_threshold: float = Field(
        default=0.7,
        description=(
            "Threshold score (0.0-1.0) that determines when to use complex_model. "
            "Scores >= threshold route to complex_model"
        ),
        ge=0.0,
        le=1.0,
    )

    # ========================================================================
    # Agent Configuration
    # ========================================================================

    max_iterations: int = Field(
        default=10,
        description=(
            "Maximum number of reasoning iterations for the agent. "
            "Prevents infinite loops and excessive processing"
        ),
        ge=1,
        le=50,
    )

    agent_verbose: bool = Field(
        default=True,
        description="Enable verbose output for agent reasoning steps and tool calls",
    )

    result_limit: int = Field(
        default=100,
        description=(
            "Maximum number of documents returned from queries. "
            "Protects against memory issues from large result sets"
        ),
        ge=1,
        le=10000,
    )

    # ========================================================================
    # Tool Loading Strategy
    # ========================================================================

    tool_loading_strategy: ToolLoadingStrategy = Field(
        default=ToolLoadingStrategy.HIERARCHICAL,
        description=(
            "Strategy for loading MCP tools. "
            "Options: all (load all), hierarchical (load by tier), adaptive (dynamic)"
        ),
    )

    # ========================================================================
    # Safety Configuration
    # ========================================================================

    read_only_mode: bool = Field(
        default=True,
        description=(
            "Enforce read-only mode to prevent destructive operations. "
            "Blocks insert, update, delete, $out, and $merge operations"
        ),
    )

    # ========================================================================
    # MCP Server Configuration
    # ========================================================================

    mcp_server_host: str = Field(
        default="127.0.0.1",
        description="Host address for the MCP server. Use 0.0.0.0 to listen on all interfaces",
    )

    mcp_server_port: int = Field(
        default=8000,
        description="Port number for the MCP server",
        ge=1024,
        le=65535,
    )

    # ========================================================================
    # Security Configuration
    # ========================================================================

    security_enabled: bool = Field(
        default=True,
        description="Enable the security layer for PHI protection and HIPAA compliance",
    )

    security_api_key: str | None = Field(
        default=None,
        description="API key for authentication (for development/single-user mode)",
    )

    security_default_role: str = Field(
        default="read_only",
        description="Default user role when security context not provided",
    )

    security_audit_log_path: str = Field(
        default="audit.log",
        description="Path for HIPAA-compliant audit logging",
    )

    security_session_timeout: int = Field(
        default=30,
        description="Session timeout in minutes",
        ge=5,
        le=480,
    )

    security_rate_limit: int = Field(
        default=60,
        description="Rate limit: requests per minute per client",
        ge=10,
        le=1000,
    )

    security_max_query_results: int = Field(
        default=100,
        description="Maximum query results to prevent bulk PHI extraction",
        ge=1,
        le=500,
    )

    security_audit_retention_days: int = Field(
        default=2555,  # 7 years
        description="Audit log retention in days (HIPAA requirement: 7 years)",
        ge=365,
        le=3650,
    )

    # ========================================================================
    # Redis Cache Configuration
    # ========================================================================

    redis_enabled: bool = Field(
        default=True,
        description="Enable Redis caching layer for performance optimization",
    )

    redis_host: str = Field(
        default="localhost",
        description="Redis server host address",
    )

    redis_port: int = Field(
        default=6379,
        description="Redis server port",
        ge=1024,
        le=65535,
    )

    redis_password: str | None = Field(
        default=None,
        description="Redis server password (if authentication required)",
    )

    redis_ssl: bool = Field(
        default=False,
        description="Use SSL/TLS for Redis connection",
    )

    redis_max_connections: int = Field(
        default=20,
        description="Maximum connections in Redis connection pool",
        ge=1,
        le=100,
    )

    redis_socket_timeout: int = Field(
        default=5,
        description="Redis socket timeout in seconds",
        ge=1,
        le=60,
    )

    # Cache TTL Settings (in seconds)
    redis_ttl_rbac_decisions: int = Field(
        default=3600,
        description="TTL for RBAC decision cache (seconds, 1 hour)",
        ge=60,
        le=86400,
    )

    redis_ttl_tool_prompts: int = Field(
        default=86400,
        description="TTL for tool prompt cache (seconds, 24 hours)",
        ge=3600,
        le=604800,
    )

    redis_ttl_field_schemas: int = Field(
        default=3600,
        description="TTL for field schema cache (seconds, 1 hour)",
        ge=60,
        le=86400,
    )

    redis_ttl_session_data: int = Field(
        default=1800,
        description="TTL for session data cache (seconds, 30 minutes)",
        ge=300,
        le=3600,
    )

    redis_ttl_agg_results: int = Field(
        default=1800,
        description="TTL for aggregation result cache (seconds, 30 minutes)",
        ge=300,
        le=3600,
    )

    # ========================================================================
    # Logging Configuration
    # ========================================================================

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level for application logs. DEBUG provides most detail",
    )

    # ========================================================================
    # Field Validators
    # ========================================================================

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def validate_api_keys(cls, value: str | None) -> str | None:
        """Validate API key format if provided.

        API keys should start with 'sk-' prefix as per OpenAI and Anthropic
        standards. This validator ensures keys follow expected format.

        Args:
            value: The API key to validate (can be None)

        Returns:
            The validated API key unchanged

        Raises:
            ValueError: If API key is provided but doesn't start with 'sk-'
        """
        if value is not None and not value.startswith("sk-"):
            raise ValueError("API key must start with 'sk-' prefix")
        return value

    @field_validator("mongodb_max_pool_size")
    @classmethod
    def validate_pool_sizes(cls, max_size: int, info) -> int:
        """Validate that max pool size is greater than min pool size.

        This validator ensures connection pool configuration is valid and
        min_pool_size is not greater than max_pool_size.

        Args:
            max_size: The maximum pool size to validate
            info: Validation context containing data

        Returns:
            The validated max pool size unchanged

        Raises:
            ValueError: If max_pool_size <= min_pool_size
        """
        if "mongodb_min_pool_size" in info.data:
            min_size = info.data["mongodb_min_pool_size"]
            if min_size > max_size:
                raise ValueError(
                    f"mongodb_min_pool_size ({min_size}) cannot exceed "
                    f"mongodb_max_pool_size ({max_size})"
                )
        return max_size

    # ========================================================================
    # Helper Properties
    # ========================================================================

    @property
    def mongodb_connection_string(self) -> str:
        """Get the formatted MongoDB connection string with credentials if provided.

        If mongodb_user and mongodb_password are set, constructs a URI with credentials.
        Otherwise, returns the mongodb_uri as-is.

        Returns:
            The MongoDB URI with credentials if user/password are provided, otherwise the base URI

        Example:
            >>> settings.mongodb_user = "admin"
            >>> settings.mongodb_password = "pass123"
            >>> settings.mongodb_connection_string
            'mongodb://admin:pass123@localhost:27017/?authSource=admin'
        """
        # If credentials are provided, construct URI with authentication
        if self.mongodb_user and self.mongodb_password:
            # Parse the base URI to extract host and port
            base_uri = self.mongodb_uri
            protocol = "mongodb://"

            # Remove protocol if present
            if base_uri.startswith("mongodb://"):
                uri_without_protocol = base_uri[10:]  # Remove "mongodb://"
            elif base_uri.startswith("mongodb+srv://"):
                protocol = "mongodb+srv://"
                uri_without_protocol = base_uri[14:]  # Remove "mongodb+srv://"
            else:
                uri_without_protocol = base_uri

            # Extract host:port (and any existing path/query)
            if "/" in uri_without_protocol:
                host_port, path_query = uri_without_protocol.split("/", 1)
            else:
                host_port = uri_without_protocol
                path_query = ""

            # Construct URI with credentials
            from urllib.parse import quote_plus

            username = quote_plus(self.mongodb_user)
            password = quote_plus(self.mongodb_password)

            # Build query parameters
            query_params = f"authSource={self.mongodb_auth_source}"
            if path_query:
                if "?" in path_query:
                    # Merge with existing query params
                    query_params = f"{path_query}&{query_params}"
                else:
                    # Path without query, add query
                    query_params = f"{path_query}?{query_params}"
            else:
                # No path, add query
                query_params = f"?{query_params}"

            return f"{protocol}{username}:{password}@{host_port}/{query_params}"

        # No credentials, return URI as-is
        return self.mongodb_uri

    @property
    def mcp_server_url(self) -> str:
        """Get the formatted MCP server URL.

        Constructs the full URL from configured host and port.

        Returns:
            The MCP server URL

        Example:
            >>> settings.mcp_server_url
            'http://127.0.0.1:8000'
        """
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}"

    @property
    def security_config(self) -> "SecurityConfig":
        """Get the security configuration object.

        Rationale: Provides a properly configured SecurityConfig object
        based on the settings. This ensures all security components use
        consistent configuration derived from environment variables.

        Returns:
            SecurityConfig: Configured security settings object
        """
        from src.mcp_server.security.config import SecurityConfig

        return SecurityConfig(
            api_key_min_length=32,  # Fixed for security
            session_timeout_minutes=self.security_session_timeout,
            max_failed_auth_attempts=5,  # Fixed for security
            lockout_duration_minutes=15,  # Fixed for security
            rate_limit_requests_per_minute=self.security_rate_limit,
            rate_limit_burst=10,  # Fixed burst allowance
            max_query_results=self.security_max_query_results,
            max_query_depth=3,  # Fixed for security
            enable_audit_logging=True,  # Always enabled for HIPAA
            audit_retention_days=self.security_audit_retention_days,
            enable_pii_redaction=False,  # Can be configured later if needed
            # allowed_query_fields uses default from SecurityConfig
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def get_api_key(self) -> str:
        """Get the appropriate API key based on configured provider.

        This method returns the API key for the currently configured LLM provider.
        It provides a convenient way to get the right key without checking
        the provider separately.

        Returns:
            The API key string for the configured provider

        Raises:
            ValueError: If API key is not configured for the provider or
                       provider is unknown

        Example:
            >>> settings.llm_provider = LLMProvider.OPENAI
            >>> api_key = settings.get_api_key()
        """
        if self.llm_provider == LLMProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured in environment variables")
            return self.openai_api_key

        if self.llm_provider == LLMProvider.ANTHROPIC:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured in environment variables")
            return self.anthropic_api_key

        raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def validate_configuration(self) -> None:
        """Validate complete application configuration at startup.

        Performs comprehensive validation of all settings to ensure the
        application can start correctly. Should be called early in the
        application initialization.

        Raises:
            ValueError: If configuration is invalid with descriptive message
                       indicating what needs to be fixed

        Example:
            >>> from src.config.settings import settings
            >>> try:
            ...     settings.validate_configuration()
            ... except ValueError as e:
            ...     print(f"Configuration error: {e}")
        """
        logger.info("Validating application configuration...")

        # Validate API key is present for configured provider
        try:
            self.get_api_key()
        except ValueError as e:
            logger.error(f"API key validation failed: {e}")
            raise ValueError(f"Invalid LLM configuration: {e}") from e

        # Validate pool size configuration
        if self.mongodb_min_pool_size > self.mongodb_max_pool_size:
            error_msg = (
                f"Invalid MongoDB pool configuration: "
                f"min_pool_size ({self.mongodb_min_pool_size}) > "
                f"max_pool_size ({self.mongodb_max_pool_size})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate complexity threshold
        if not 0.0 <= self.complexity_threshold <= 1.0:
            error_msg = (
                f"Invalid complexity_threshold: {self.complexity_threshold}. "
                f"Must be between 0.0 and 1.0"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")

    def print_config(self) -> None:
        """Print current configuration in a formatted table (excluding sensitive data).

        This method is useful for debugging and verifying configuration at startup.
        Sensitive data like API keys are masked to prevent accidental exposure.

        Example:
            >>> from src.config.settings import settings
            >>> settings.print_config()
        """
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Application Configuration", show_header=True)
            table.add_column("Setting", style="cyan", no_wrap=False)
            table.add_column("Value", style="magenta", no_wrap=False)

            # Mask sensitive connection strings
            mongodb_uri_display = self.mongodb_uri
            if "@" in mongodb_uri_display:
                # Hide credentials in connection string
                parts = mongodb_uri_display.split("@")
                mongodb_uri_display = "mongodb://***:***@" + parts[1]

            # Non-sensitive settings to display
            config_items = {
                "MongoDB URI": mongodb_uri_display,
                "Database": self.mongodb_database,
                "Timeout": f"{self.mongodb_timeout}s",
                "Connection Pool": f"{self.mongodb_min_pool_size}-{self.mongodb_max_pool_size}",
                "LLM Provider": self.llm_provider.value,
                "Default Model": self.default_model,
                "Complex Model": self.complex_model,
                "Complexity Threshold": f"{self.complexity_threshold:.2f}",
                "Max Iterations": str(self.max_iterations),
                "Result Limit": str(self.result_limit),
                "Tool Loading Strategy": self.tool_loading_strategy.value,
                "Read-Only Mode": "✓ Enabled" if self.read_only_mode else "✗ Disabled",
                "MCP Server URL": self.mcp_server_url,
                "Log Level": self.log_level,
            }

            for key, value in config_items.items():
                table.add_row(key, str(value))

            console.print(table)

        except ImportError:
            # Fall back to simple print if rich is not available
            logger.info("Configuration (Rich library not available for formatting):")
            for key, value in config_items.items():  # noqa: F821
                logger.info(f"  {key}: {value}")


# Global settings instance - initialized once at module import
settings = Settings()


def print_config() -> None:
    """Convenience function to print current configuration.

    This is a module-level wrapper around settings.print_config() for
    easier access from command-line interfaces and initialization scripts.

    Example:
        >>> from src.config.settings import print_config
        >>> print_config()
    """
    settings.print_config()


if __name__ == "__main__":
    # Script: Validate and display configuration
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        logger.info("Starting configuration validation...")
        settings.validate_configuration()
        print_config()
        logger.info("✓ Configuration is valid and ready for use")
        sys.exit(0)

    except ValueError as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"✗ Unexpected error during configuration: {e}", exc_info=True)
        sys.exit(1)
