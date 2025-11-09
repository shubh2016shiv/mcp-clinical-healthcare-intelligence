#!/usr/bin/env python3
"""
Structured Logging - Enterprise Edition

Advanced logging system with JSON formatting, correlation IDs, and contextual
information for enterprise-grade observability.

Features:
- JSON-formatted log output for machine parsing
- Correlation IDs for request tracing
- Contextual logging with automatic metadata
- Multiple output destinations (console, file)
- Log level filtering and formatting
- Thread-safe operations
"""

import json
import logging
import sys
import threading
import uuid
from datetime import datetime
from typing import Any, Optional


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds correlation ID to all log records."""

    def __init__(self, correlation_id: str | None = None):
        """Initialize filter with correlation ID.

        Args:
            correlation_id: Correlation ID to use (generates new one if None)
        """
        super().__init__()
        self.correlation_id = correlation_id or self.generate_correlation_id()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID.

        Returns:
            New correlation ID string
        """
        return str(uuid.uuid4())

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record.

        Args:
            record: Log record to filter

        Returns:
            Always True to include all records
        """
        if not hasattr(record, "correlation_id"):
            record.correlation_id = self.correlation_id
        return True


class ContextFilter(logging.Filter):
    """Logging filter that adds contextual information to log records."""

    def __init__(self):
        """Initialize context filter."""
        super().__init__()
        self.context_data = threading.local()

    def set_context(self, **kwargs) -> None:
        """Set contextual data for current thread.

        Args:
            **kwargs: Context key-value pairs
        """
        for key, value in kwargs.items():
            setattr(self.context_data, key, value)

    def clear_context(self) -> None:
        """Clear contextual data for current thread."""
        self.context_data.__dict__.clear()

    def get_context(self) -> dict[str, Any]:
        """Get current thread's context data.

        Returns:
            Dictionary of context data
        """
        return self.context_data.__dict__.copy()

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context data to log record.

        Args:
            record: Log record to filter

        Returns:
            Always True to include all records
        """
        # Add context data
        context = self.get_context()
        for key, value in context.items():
            if not hasattr(record, key):
                setattr(record, key, value)

        # Add standard metadata
        if not hasattr(record, "timestamp"):
            record.timestamp = datetime.utcnow().isoformat() + "Z"
        if not hasattr(record, "thread_id"):
            record.thread_id = threading.get_ident()
        if not hasattr(record, "thread_name"):
            record.thread_name = threading.current_thread().name

        return True


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, include_extra: bool = True):
        """Initialize JSON formatter.

        Args:
            include_extra: Include extra fields in JSON output
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        # Base log data
        log_data = {
            "timestamp": getattr(record, "timestamp", datetime.utcnow().isoformat() + "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "thread_id": getattr(record, "thread_id", None),
            "thread_name": getattr(record, "thread_name", None),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add context data
        context_attrs = [
            "pipeline_stage",
            "collection_name",
            "operation_type",
            "record_count",
            "duration_ms",
            "error_type",
            "component",
            "operation",
            "status",
        ]

        for attr in context_attrs:
            value = getattr(record, attr, None)
            if value is not None:
                log_data[attr] = value

        # Add any extra fields if requested
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                ] + list(log_data.keys()):
                    log_data[key] = value

        return json.dumps(log_data, default=str, separators=(",", ":"))


class ConsoleFormatter(logging.Formatter):
    """Enhanced console formatter with colors and structured output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Get color for level
        color = self.COLORS.get(record.levelname, self.RESET)

        # Format timestamp
        timestamp = getattr(record, "timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

        # Build components
        parts = [
            f"{color}{self.BOLD}{record.levelname}{self.RESET}",
            f"{timestamp}",
            f"{record.name}",
        ]

        # Add correlation ID if present
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            parts.append(f"CID:{correlation_id[:8]}")

        # Add context info
        context_parts = []
        if hasattr(record, "pipeline_stage"):
            context_parts.append(f"stage:{record.pipeline_stage}")
        if hasattr(record, "collection_name"):
            context_parts.append(f"collection:{record.collection_name}")
        if hasattr(record, "operation_type"):
            context_parts.append(f"op:{record.operation_type}")
        if hasattr(record, "record_count"):
            context_parts.append(f"count:{record.record_count}")

        if context_parts:
            parts.append(f"[{', '.join(context_parts)}]")

        # Add message
        parts.append(record.getMessage())

        return " | ".join(parts)


class StructuredLogger:
    """Structured logger with correlation ID and context support."""

    def __init__(self, name: str, correlation_id: str | None = None):
        """Initialize structured logger.

        Args:
            name: Logger name
            correlation_id: Correlation ID for this logger instance
        """
        self.logger = logging.getLogger(name)
        self.correlation_id = correlation_id
        self.context_filter = ContextFilter()

        # Add filters if not already present
        if not any(isinstance(f, CorrelationIdFilter) for f in self.logger.filters):
            correlation_filter = CorrelationIdFilter(correlation_id)
            self.logger.addFilter(correlation_filter)

        if not any(isinstance(f, ContextFilter) for f in self.logger.filters):
            self.logger.addFilter(self.context_filter)

    def set_context(self, **kwargs) -> None:
        """Set contextual data for this logger instance."""
        self.context_filter.set_context(**kwargs)

    def clear_context(self) -> None:
        """Clear contextual data for this logger instance."""
        self.context_filter.clear_context()

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional context."""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with optional context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log message with additional context."""
        # Set context for this log entry
        if kwargs:
            self.set_context(**kwargs)

        # Log the message
        self.logger.log(level, message)

        # Clear context after logging
        if kwargs:
            self.clear_context()


class LogManager:
    """Centralized logging configuration and management."""

    _instance: Optional["LogManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LogManager":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize log manager."""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._configured = False
        self._handlers: dict[str, logging.Handler] = {}

    def configure(
        self,
        level: str = "INFO",
        structured: bool = False,
        correlation_id_enabled: bool = True,
        console_enabled: bool = True,
        file_path: str | None = None,
        file_max_bytes: int = 10 * 1024 * 1024,  # 10MB
        file_backup_count: int = 5,
    ) -> None:
        """Configure logging system.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            structured: Use JSON structured logging
            correlation_id_enabled: Enable correlation ID tracking
            console_enabled: Enable console logging
            file_path: File path for file logging (None disables)
            file_max_bytes: Maximum file size before rotation
            file_backup_count: Number of backup files to keep
        """
        with self._lock:
            if self._configured:
                return

            # Clear existing handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Set log level
            numeric_level = getattr(logging, level.upper(), logging.INFO)
            root_logger.setLevel(numeric_level)

            # Add correlation ID filter to root logger
            if correlation_id_enabled:
                if not any(isinstance(f, CorrelationIdFilter) for f in root_logger.filters):
                    correlation_filter = CorrelationIdFilter()
                    root_logger.addFilter(correlation_filter)

            # Add context filter to root logger
            if not any(isinstance(f, ContextFilter) for f in root_logger.filters):
                context_filter = ContextFilter()
                root_logger.addFilter(context_filter)

            # Configure console handler
            if console_enabled:
                console_handler = logging.StreamHandler(sys.stdout)
                if structured:
                    formatter = JsonFormatter()
                else:
                    formatter = ConsoleFormatter()
                console_handler.setFormatter(formatter)
                console_handler.setLevel(numeric_level)
                root_logger.addHandler(console_handler)
                self._handlers["console"] = console_handler

            # Configure file handler
            if file_path:
                try:
                    from logging.handlers import RotatingFileHandler

                    file_handler = RotatingFileHandler(
                        file_path,
                        maxBytes=file_max_bytes,
                        backupCount=file_backup_count,
                        encoding="utf-8",
                    )
                    file_handler.setFormatter(JsonFormatter())  # Always JSON for files
                    file_handler.setLevel(numeric_level)
                    root_logger.addHandler(file_handler)
                    self._handlers["file"] = file_handler

                except Exception as e:
                    print(f"Failed to configure file logging: {e}")

            self._configured = True

    def get_logger(self, name: str, correlation_id: str | None = None) -> StructuredLogger:
        """Get a structured logger instance.

        Args:
            name: Logger name
            correlation_id: Correlation ID for this logger

        Returns:
            Structured logger instance
        """
        return StructuredLogger(name, correlation_id)

    def set_global_context(self, **kwargs) -> None:
        """Set global context data for all loggers."""
        root_logger = logging.getLogger()
        for filter_obj in root_logger.filters:
            if isinstance(filter_obj, ContextFilter):
                filter_obj.set_context(**kwargs)
                break

    def clear_global_context(self) -> None:
        """Clear global context data."""
        root_logger = logging.getLogger()
        for filter_obj in root_logger.filters:
            if isinstance(filter_obj, ContextFilter):
                filter_obj.clear_context()
                break

    def flush_handlers(self) -> None:
        """Flush all log handlers."""
        for handler in self._handlers.values():
            try:
                handler.flush()
            except Exception:
                pass


# Global instance and convenience functions
_log_manager = LogManager()


def configure_logging(
    level: str = "INFO",
    structured: bool = False,
    correlation_id_enabled: bool = True,
    console_enabled: bool = True,
    file_path: str | None = None,
) -> None:
    """Configure the logging system.

    Args:
        level: Log level
        structured: Use structured JSON logging
        correlation_id_enabled: Enable correlation IDs
        console_enabled: Enable console logging
        file_path: File path for logging (None disables file logging)
    """
    _log_manager.configure(
        level=level,
        structured=structured,
        correlation_id_enabled=correlation_id_enabled,
        console_enabled=console_enabled,
        file_path=file_path,
    )


def get_logger(name: str, correlation_id: str | None = None) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name
        correlation_id: Correlation ID

    Returns:
        Structured logger instance
    """
    return _log_manager.get_logger(name, correlation_id)


def set_global_context(**kwargs) -> None:
    """Set global logging context."""
    _log_manager.set_global_context(**kwargs)


def clear_global_context() -> None:
    """Clear global logging context."""
    _log_manager.clear_global_context()


def flush_logs() -> None:
    """Flush all log handlers."""
    _log_manager.flush_handlers()


# Context manager for scoped logging
class LoggingContext:
    """Context manager for scoped logging with automatic context management."""

    def __init__(self, **context):
        """Initialize logging context.

        Args:
            **context: Context key-value pairs
        """
        self.context = context
        self.previous_context = {}

    def __enter__(self):
        """Enter context and set logging context."""
        # Store previous context
        root_logger = logging.getLogger()
        for filter_obj in root_logger.filters:
            if isinstance(filter_obj, ContextFilter):
                self.previous_context = filter_obj.get_context()
                filter_obj.set_context(**self.context)
                break
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous logging context."""
        root_logger = logging.getLogger()
        for filter_obj in root_logger.filters:
            if isinstance(filter_obj, ContextFilter):
                filter_obj.clear_context()
                filter_obj.set_context(**self.previous_context)
                break


# Convenience context manager
def log_context(**kwargs):
    """Context manager for logging context.

    Args:
        **kwargs: Context key-value pairs

    Example:
        with log_context(pipeline_stage="ingestion", collection="patients"):
            logger.info("Processing patients")
    """
    return LoggingContext(**kwargs)
