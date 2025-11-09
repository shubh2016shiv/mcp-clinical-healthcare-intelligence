#!/usr/bin/env python3
"""
Configuration Management - Enterprise Edition

Centralized configuration management with environment-based settings,
validation, and runtime configuration loading.

Features:
- Environment-based configuration (dev/staging/prod)
- Configuration validation on startup
- Type-safe configuration classes
- Environment variable integration
- Configuration inheritance and overrides
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    raise ImportError(
        "pydantic is required for configuration management. Install with: pip install pydantic"
    ) from None

from healthcare_data_pipeline.connection_manager import ConnectionConfig
from healthcare_data_pipeline.retry_handler import RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    structured: bool = Field(default=False, description="Enable structured JSON logging")
    correlation_id_enabled: bool = Field(default=True, description="Enable correlation IDs")
    file_path: str | None = Field(default=None, description="Log file path")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {value}. Must be one of {valid_levels}")
        return value.upper()


class PipelineConfig(BaseModel):
    """Pipeline-specific configuration."""

    batch_size: int = Field(
        default=500, description="Batch size for bulk operations", ge=1, le=10000
    )
    max_workers: int = Field(default=4, description="Maximum worker threads/processes", ge=1, le=32)
    enable_parallel_ingestion: bool = Field(
        default=True, description="Enable parallel file ingestion"
    )
    enable_parallel_transformation: bool = Field(
        default=True, description="Enable parallel transformation"
    )
    checkpoint_enabled: bool = Field(default=True, description="Enable checkpointing for resume")
    dlq_enabled: bool = Field(default=True, description="Enable dead letter queue")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    data_quality_enabled: bool = Field(default=True, description="Enable data quality validation")


class GeminiConfig(BaseModel):
    """Gemini AI configuration."""

    api_key: str | None = Field(default=None, description="Gemini API key")
    model: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    max_retries: int = Field(default=3, description="Maximum API retries", ge=0, le=10)
    timeout: float = Field(default=30.0, description="API timeout in seconds", gt=0)
    enabled: bool = Field(default=True, description="Enable Gemini enrichment")


class ApplicationConfig(BaseModel):
    """Main application configuration."""

    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent, description="Project root path"
    )

    # Database configuration
    mongodb: ConnectionConfig = Field(
        default_factory=ConnectionConfig, description="MongoDB connection config"
    )

    # Retry configuration
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")

    # Component configurations
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    pipeline: PipelineConfig = Field(
        default_factory=PipelineConfig, description="Pipeline configuration"
    )
    gemini: GeminiConfig = Field(
        default_factory=GeminiConfig, description="Gemini AI configuration"
    )

    @field_validator("project_root")
    @classmethod
    def validate_project_root(cls, value: Path) -> Path:
        """Validate project root exists."""
        if not value.exists():
            raise ValueError(f"Project root path does not exist: {value}")
        return value

    def get_database_name(self) -> str:
        """Get database name with environment suffix for non-production."""
        base_name = self.mongodb.db_name
        if self.environment != Environment.PRODUCTION:
            return f"{base_name}_{self.environment.value}"
        return base_name


class ConfigManager:
    """Centralized configuration manager with environment loading."""

    _instance: Optional["ConfigManager"] = None
    _config: ApplicationConfig | None = None

    def __new__(cls) -> "ConfigManager":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize config manager."""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._config = None
        self._env_vars_loaded = False

    def load_from_env_file(self, env_file: str | None = None) -> dict[str, str]:
        """Load environment variables from .env file.

        Args:
            env_file: Path to .env file (auto-discovers if None)

        Returns:
            Dictionary of loaded environment variables
        """
        env_vars = {}

        # Find .env file
        if env_file is None:
            project_root = Path(__file__).parent.parent.resolve()
            possible_files = [
                project_root / ".env",
                project_root / ".env.local",
                project_root / f".env.{os.environ.get('ENVIRONMENT', 'development')}",
            ]

            for env_path in possible_files:
                if env_path.exists():
                    env_file = str(env_path.resolve())
                    break

        if env_file and os.path.exists(env_file):
            try:
                from dotenv import load_dotenv

                # Use override=True to ensure .env values take precedence over existing env vars
                load_dotenv(env_file, override=True)
                logger.info(f"[INFO] Loaded environment variables from {env_file}")

                # Read and store loaded vars
                with open(env_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            env_vars[key] = value.strip().strip("\"'")

            except ImportError:
                logger.warning("[WARNING] python-dotenv not installed, skipping .env file loading")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to load .env file: {e}")
        else:
            # Fallback: Try auto-discovery (load_dotenv searches current dir and parents)
            try:
                from dotenv import load_dotenv

                result = load_dotenv(override=True)
                if result:
                    logger.info(
                        "[INFO] Loaded environment variables from auto-discovered .env file"
                    )
            except ImportError:
                logger.warning("[WARNING] python-dotenv not installed, skipping .env file loading")
            except Exception as e:
                logger.debug(f"[DEBUG] Auto-discovery of .env file failed: {e}")

        self._env_vars_loaded = True
        return env_vars

    def load_from_environment(self) -> ApplicationConfig:
        """Load configuration from environment variables and defaults.

        Returns:
            Application configuration instance

        Raises:
            ValidationError: If configuration is invalid
        """
        if not self._env_vars_loaded:
            self.load_from_env_file()

        # Get environment
        env_str = os.environ.get("ENVIRONMENT", "development").lower()
        try:
            environment = Environment(env_str)
        except ValueError:
            logger.warning(f"[WARNING] Invalid ENVIRONMENT '{env_str}', defaulting to development")
            environment = Environment.DEVELOPMENT

        # MongoDB configuration
        mongodb_config = ConnectionConfig(
            host=os.environ.get("MONGODB_HOST", "localhost"),
            port=int(os.environ.get("MONGODB_PORT", "27017")),
            user=os.environ.get("MONGODB_USER", "admin"),
            password=os.environ.get("MONGODB_PASSWORD", "mongopass123"),
            db_name=os.environ.get("MONGODB_DATABASE", "text_to_mongo_db"),
            auth_source=os.environ.get("MONGODB_AUTH_SOURCE", "admin"),
            max_pool_size=int(os.environ.get("MONGODB_MAX_POOL_SIZE", "10")),
            min_pool_size=int(os.environ.get("MONGODB_MIN_POOL_SIZE", "2")),
            max_idle_time_ms=int(os.environ.get("MONGODB_MAX_IDLE_TIME_MS", "30000")),
            server_selection_timeout_ms=int(
                os.environ.get("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "30000")
            ),
            connect_timeout_ms=int(os.environ.get("MONGODB_CONNECT_TIMEOUT_MS", "20000")),
            socket_timeout_ms=int(os.environ.get("MONGODB_SOCKET_TIMEOUT_MS", "20000")),
            retry_writes=os.environ.get("MONGODB_RETRY_WRITES", "true").lower() == "true",
            retry_reads=os.environ.get("MONGODB_RETRY_READS", "true").lower() == "true",
            heartbeat_frequency_ms=int(os.environ.get("MONGODB_HEARTBEAT_FREQUENCY_MS", "10000")),
        )

        # Retry configuration
        retry_config = RetryConfig(
            max_attempts=int(os.environ.get("RETRY_MAX_ATTEMPTS", "3")),
            base_delay=float(os.environ.get("RETRY_BASE_DELAY", "1.0")),
            max_delay=float(os.environ.get("RETRY_MAX_DELAY", "60.0")),
            backoff_factor=float(os.environ.get("RETRY_BACKOFF_FACTOR", "2.0")),
            strategy=RetryStrategy(os.environ.get("RETRY_STRATEGY", "exponential_jitter")),
            jitter_factor=float(os.environ.get("RETRY_JITTER_FACTOR", "0.1")),
            circuit_breaker_enabled=os.environ.get("CIRCUIT_BREAKER_ENABLED", "true").lower()
            == "true",
            circuit_breaker_threshold=int(os.environ.get("CIRCUIT_BREAKER_THRESHOLD", "5")),
            circuit_breaker_timeout=float(os.environ.get("CIRCUIT_BREAKER_TIMEOUT", "60.0")),
        )

        # Logging configuration
        logging_config = LoggingConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            structured=os.environ.get("LOG_STRUCTURED", "false").lower() == "true",
            correlation_id_enabled=os.environ.get("LOG_CORRELATION_ID", "true").lower() == "true",
            file_path=os.environ.get("LOG_FILE_PATH"),
        )

        # Pipeline configuration
        pipeline_config = PipelineConfig(
            batch_size=int(os.environ.get("PIPELINE_BATCH_SIZE", "500")),
            max_workers=int(os.environ.get("PIPELINE_MAX_WORKERS", "4")),
            enable_parallel_ingestion=os.environ.get("PIPELINE_PARALLEL_INGESTION", "true").lower()
            == "true",
            enable_parallel_transformation=os.environ.get(
                "PIPELINE_PARALLEL_TRANSFORMATION", "true"
            ).lower()
            == "true",
            checkpoint_enabled=os.environ.get("PIPELINE_CHECKPOINT_ENABLED", "true").lower()
            == "true",
            dlq_enabled=os.environ.get("PIPELINE_DLQ_ENABLED", "true").lower() == "true",
            metrics_enabled=os.environ.get("PIPELINE_METRICS_ENABLED", "true").lower() == "true",
            data_quality_enabled=os.environ.get("PIPELINE_DATA_QUALITY_ENABLED", "true").lower()
            == "true",
        )

        # Gemini configuration
        gemini_config = GeminiConfig(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            max_retries=int(os.environ.get("GEMINI_MAX_RETRIES", "3")),
            timeout=float(os.environ.get("GEMINI_TIMEOUT", "30.0")),
            enabled=os.environ.get("GEMINI_ENABLED", "true").lower() == "true",
        )

        # Create application config
        config = ApplicationConfig(
            environment=environment,
            mongodb=mongodb_config,
            retry=retry_config,
            logging=logging_config,
            pipeline=pipeline_config,
            gemini=gemini_config,
        )

        self._config = config
        logger.info(f"[INFO] Configuration loaded for environment: {environment.value}")
        return config

    def get_config(self) -> ApplicationConfig:
        """Get the current application configuration.

        Returns:
            Application configuration instance

        Raises:
            RuntimeError: If configuration not loaded
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_from_environment() first.")
        return self._config

    def validate_config(self) -> list[str]:
        """Validate the current configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self._config is None:
            errors.append("Configuration not loaded")
            return errors

        config = self._config

        # Validate MongoDB configuration
        # Only enforce password validation in non-development environments
        if config.environment != Environment.DEVELOPMENT:
            if not config.mongodb.password or config.mongodb.password == "mongopass123":
                errors.append(
                    "MongoDB password not set or using default value (not allowed in production/staging)"
                )

        # Validate Gemini configuration
        if config.gemini.enabled and not config.gemini.api_key:
            errors.append("Gemini enabled but API key not provided")

        # Validate pipeline configuration
        if config.pipeline.batch_size < 1:
            errors.append("Pipeline batch size must be positive")

        if config.pipeline.max_workers < 1:
            errors.append("Pipeline max workers must be positive")

        return errors

    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        if self._config is None:
            print("Configuration not loaded")
            return

        config = self._config
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Environment: {config.environment.value}")
        print(f"Project Root: {config.project_root}")
        print()
        print("MongoDB:")
        print(f"  Host: {config.mongodb.host}:{config.mongodb.port}")
        print(f"  Database: {config.get_database_name()}")
        print(f"  User: {config.mongodb.user}")
        print(f"  Pool Size: {config.mongodb.max_pool_size}")
        print()
        print("Pipeline:")
        print(f"  Batch Size: {config.pipeline.batch_size}")
        print(f"  Max Workers: {config.pipeline.max_workers}")
        print(f"  Parallel Ingestion: {config.pipeline.enable_parallel_ingestion}")
        print(f"  Parallel Transformation: {config.pipeline.enable_parallel_transformation}")
        print()
        print("Gemini AI:")
        print(f"  Enabled: {config.gemini.enabled}")
        print(f"  Model: {config.gemini.model}")
        print(f"  Has API Key: {'Yes' if config.gemini.api_key else 'No'}")
        print()
        print("Retry Configuration:")
        print(f"  Max Attempts: {config.retry.max_attempts}")
        print(f"  Circuit Breaker: {config.retry.circuit_breaker_enabled}")
        print("=" * 60 + "\n")


# Global instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance.

    Returns:
        Configuration manager instance
    """
    return _config_manager


def load_config() -> ApplicationConfig:
    """Load and return application configuration.

    Returns:
        Application configuration instance
    """
    return _config_manager.load_from_environment()


def get_config() -> ApplicationConfig:
    """Get the current application configuration.

    Returns:
        Application configuration instance
    """
    return _config_manager.get_config()


def validate_config() -> list[str]:
    """Validate the current configuration.

    Returns:
        List of validation error messages (empty if valid)
    """
    return _config_manager.validate_config()


def print_config_summary() -> None:
    """Print configuration summary."""
    _config_manager.print_config_summary()


def create_env_template(output_path: str | None = None) -> str:
    """Create a .env template file.

    Args:
        output_path: Path to write the template file (prints to stdout if None)

    Returns:
        Template content as string
    """
    template = """# Healthcare Data Pipeline Environment Configuration
# Copy this file to .env and customize for your environment

# Environment Settings
ENVIRONMENT=development

# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=mongopass123
MONGODB_DATABASE=text_to_mongo_db
MONGODB_AUTH_SOURCE=admin

# MongoDB Connection Pool Settings
MONGODB_MAX_POOL_SIZE=10
MONGODB_MIN_POOL_SIZE=2
MONGODB_MAX_IDLE_TIME_MS=30000
MONGODB_SERVER_SELECTION_TIMEOUT_MS=30000
MONGODB_CONNECT_TIMEOUT_MS=20000
MONGODB_SOCKET_TIMEOUT_MS=20000
MONGODB_RETRY_WRITES=true
MONGODB_RETRY_READS=true
MONGODB_HEARTBEAT_FREQUENCY_MS=10000

# Retry Configuration
RETRY_MAX_ATTEMPTS=3
RETRY_BASE_DELAY=1.0
RETRY_MAX_DELAY=60.0
RETRY_BACKOFF_FACTOR=2.0
RETRY_STRATEGY=exponential_jitter
RETRY_JITTER_FACTOR=0.1
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60.0

# Logging Configuration
LOG_LEVEL=INFO
LOG_STRUCTURED=false
LOG_CORRELATION_ID=true
LOG_FILE_PATH=

# Pipeline Configuration
PIPELINE_BATCH_SIZE=500
PIPELINE_MAX_WORKERS=4
PIPELINE_PARALLEL_INGESTION=true
PIPELINE_PARALLEL_TRANSFORMATION=true
PIPELINE_CHECKPOINT_ENABLED=true
PIPELINE_DLQ_ENABLED=true
PIPELINE_METRICS_ENABLED=true
PIPELINE_DATA_QUALITY_ENABLED=true

# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MAX_RETRIES=3
GEMINI_TIMEOUT=30.0
GEMINI_ENABLED=true
"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(template)
        logger.info(f"[INFO] Environment template written to {output_path}")
    else:
        print(template)

    return template
