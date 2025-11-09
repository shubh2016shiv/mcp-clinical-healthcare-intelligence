#!/usr/bin/env python3
"""
Retry Handler - Enterprise Edition

Comprehensive retry framework with exponential backoff, circuit breaker pattern,
and configurable retry policies for database operations and API calls.

Features:
- Exponential backoff with jitter
- Circuit breaker pattern for fault tolerance
- Configurable retry policies per operation type
- Metrics collection and monitoring
- Thread-safe operations
"""

import functools
import logging
import random
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "pydantic is required for retry handler. Install with: pip install pydantic"
    ) from None

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry strategy types."""

    IMMEDIATE = "immediate"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(default=3, description="Maximum number of retry attempts", ge=1)
    base_delay: float = Field(default=1.0, description="Base delay between retries (seconds)", ge=0)
    max_delay: float = Field(
        default=60.0, description="Maximum delay between retries (seconds)", ge=0
    )
    backoff_factor: float = Field(default=2.0, description="Exponential backoff multiplier", gt=0)
    strategy: RetryStrategy = Field(
        default=RetryStrategy.EXPONENTIAL_JITTER, description="Retry delay strategy"
    )
    jitter_factor: float = Field(
        default=0.1, description="Random jitter factor (0.0 to 1.0)", ge=0, le=1
    )
    retryable_exceptions: list[str] | None = Field(
        default=None,
        description="List of exception type names to retry on (e.g., 'ConnectionError', 'TimeoutError')",
    )
    circuit_breaker_enabled: bool = Field(
        default=True, description="Enable circuit breaker pattern"
    )
    circuit_breaker_threshold: int = Field(
        default=5, description="Failures before opening circuit", ge=1
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, description="Time before attempting half-open (seconds)", ge=0
    )
    circuit_breaker_half_open_max_calls: int = Field(
        default=3, description="Max calls in half-open state", ge=1
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False  # Allow mutation for compatibility
        arbitrary_types_allowed = True  # Allow exception types

    def __init__(self, **data):
        """Initialize retry configuration with default retryable exceptions."""
        super().__init__(**data)
        # Set default retryable exceptions if not provided
        if self.retryable_exceptions is None:
            # Store as strings for Pydantic compatibility
            self.retryable_exceptions = ["ConnectionError", "TimeoutError", "OSError"]

    def get_retryable_exception_types(self) -> list[type[Exception]]:
        """Get actual exception types from string names.

        Returns:
            List of exception types to retry on
        """
        if self.retryable_exceptions is None:
            return [ConnectionError, TimeoutError, OSError]

        # Build exception type map dynamically
        exception_map = {
            "ConnectionError": ConnectionError,
            "TimeoutError": TimeoutError,
            "OSError": OSError,
            "Exception": Exception,
        }

        result = []
        for exc_name in self.retryable_exceptions:
            exc_type = exception_map.get(exc_name)
            if exc_type:
                result.append(exc_type)

        return result if result else [ConnectionError, TimeoutError, OSError]


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: RetryConfig):
        """Initialize circuit breaker.

        Args:
            config: Retry configuration
        """
        self.config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_call_count = 0
        self._lock = threading.RLock()

    def _should_attempt_call(self) -> bool:
        """Determine if a call should be attempted based on circuit state.

        Returns:
            True if call should be attempted, False if circuit is open
        """
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True

            if self._state == CircuitBreakerState.OPEN:
                # Check if timeout has passed to move to half-open
                if time.time() - self._last_failure_time >= self.config.circuit_breaker_timeout:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_call_count = 0
                    logger.info("[INFO] Circuit breaker moving to HALF_OPEN state")
                    return True
                return False

            if self._state == CircuitBreakerState.HALF_OPEN:
                return self._half_open_call_count < self.config.circuit_breaker_half_open_max_calls

            return False

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                logger.info("[INFO] Circuit breaker moving to CLOSED state")
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_call_count += 1
                if self._half_open_call_count >= self.config.circuit_breaker_half_open_max_calls:
                    self._state = CircuitBreakerState.OPEN
                    logger.warning("[WARNING] Circuit breaker moving to OPEN state")
            elif self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.config.circuit_breaker_threshold:
                    self._state = CircuitBreakerState.OPEN
                    logger.warning("[WARNING] Circuit breaker moving to OPEN state")

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state.

        Returns:
            Current circuit breaker state
        """
        with self._lock:
            return self._state

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats
        """
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time,
                "half_open_call_count": self._half_open_call_count,
            }


class RetryHandler:
    """Retry handler with exponential backoff and circuit breaker."""

    def __init__(self, config: RetryConfig):
        """Initialize retry handler.

        Args:
            config: Retry configuration
        """
        self.config = config
        self.circuit_breaker = CircuitBreaker(config) if config.circuit_breaker_enabled else None
        self._metrics = RetryMetrics()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0

        if self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        else:  # EXPONENTIAL or EXPONENTIAL_JITTER
            delay = self.config.base_delay * (self.config.backoff_factor**attempt)

        # Cap at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if configured
        if self.config.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            jitter = delay * self.config.jitter_factor * random.uniform(-1, 1)
            delay += jitter

        return max(0, delay)

    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if an exception is retryable.

        Args:
            exception: Exception to check

        Returns:
            True if exception is retryable, False otherwise
        """
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.config.get_retryable_exception_types()
        )

    def execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            Exception: Last exception if all retries exhausted
        """
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker._should_attempt_call():
            self._metrics.circuit_breaker_rejections += 1
            raise RuntimeError("Circuit breaker is OPEN - operation rejected")

        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                self._metrics.total_attempts += 1
                result = func(*args, **kwargs)
                self._metrics.successful_calls += 1

                # Record success for circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker._record_success()

                return result

            except Exception as e:
                last_exception = e
                self._metrics.failed_attempts += 1

                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.debug(f"[DEBUG] Non-retryable exception: {type(e).__name__}: {e}")
                    break

                # Record failure for circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker._record_failure()

                # Check if this was the last attempt
                if attempt == self.config.max_attempts - 1:
                    logger.warning(
                        f"[WARNING] All {self.config.max_attempts} attempts exhausted for {func.__name__}"
                    )
                    break

                # Calculate and apply delay
                delay = self._calculate_delay(attempt)
                if delay > 0:
                    logger.debug(
                        f"[DEBUG] Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{self.config.max_attempts})"
                    )
                    time.sleep(delay)

        # All retries exhausted
        if self.circuit_breaker:
            self.circuit_breaker._record_failure()

        raise last_exception

    def get_metrics(self) -> dict[str, Any]:
        """Get retry handler metrics.

        Returns:
            Dictionary with retry metrics
        """
        metrics = {
            "total_attempts": self._metrics.total_attempts,
            "successful_calls": self._metrics.successful_calls,
            "failed_attempts": self._metrics.failed_attempts,
            "circuit_breaker_rejections": self._metrics.circuit_breaker_rejections,
        }

        if self.circuit_breaker:
            metrics["circuit_breaker"] = self.circuit_breaker.get_stats()

        return metrics


class RetryMetrics:
    """Metrics for retry operations."""

    def __init__(self):
        self.total_attempts = 0
        self.successful_calls = 0
        self.failed_attempts = 0
        self.circuit_breaker_rejections = 0


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER,
    jitter_factor: float = 0.1,
    retryable_exceptions: list[type[Exception]] | None = None,
    circuit_breaker_enabled: bool = True,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_timeout: float = 60.0,
    circuit_breaker_half_open_max_calls: int = 3,
) -> Callable:
    """Decorator for functions that should be retried on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Exponential backoff multiplier
        strategy: Retry delay strategy
        jitter_factor: Random jitter factor (0.0 to 1.0)
        retryable_exceptions: List of exception types to retry on (will be converted to strings)
        circuit_breaker_enabled: Enable circuit breaker pattern
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Time before attempting half-open (seconds)
        circuit_breaker_half_open_max_calls: Max calls in half-open state

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Convert exception types to string names for Pydantic compatibility
        retryable_exception_names = None
        if retryable_exceptions:
            retryable_exception_names = [exc.__name__ for exc in retryable_exceptions]

        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            strategy=strategy,
            jitter_factor=jitter_factor,
            retryable_exceptions=retryable_exception_names,
            circuit_breaker_enabled=circuit_breaker_enabled,
            circuit_breaker_threshold=circuit_breaker_threshold,
            circuit_breaker_timeout=circuit_breaker_timeout,
            circuit_breaker_half_open_max_calls=circuit_breaker_half_open_max_calls,
        )

        retry_handler = RetryHandler(config)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_handler.execute_with_retry(func, *args, **kwargs)

        # Store retry handler on the function for introspection
        wrapper._retry_handler = retry_handler
        return wrapper

    return decorator


# Pre-configured retry handlers for common use cases

mongodb_retry_config = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    backoff_factor=2.0,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    retryable_exceptions=[
        "ConnectionError",
        "TimeoutError",
        "OSError",
        "Exception",  # Catch-all for MongoDB errors (will be refined)
    ],
    circuit_breaker_enabled=True,
    circuit_breaker_threshold=10,
    circuit_breaker_timeout=120.0,
)

api_retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    retryable_exceptions=[
        "ConnectionError",
        "TimeoutError",
        "OSError",
        "Exception",  # HTTP errors
    ],
    circuit_breaker_enabled=True,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=300.0,
)

# Global retry handlers
_mongodb_retry_handler = RetryHandler(mongodb_retry_config)
_api_retry_handler = RetryHandler(api_retry_config)


def mongodb_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for MongoDB operations with optimized retry settings."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        return _mongodb_retry_handler.execute_with_retry(func, *args, **kwargs)

    return wrapper


def api_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for API calls with optimized retry settings."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        return _api_retry_handler.execute_with_retry(func, *args, **kwargs)

    return wrapper


def get_retry_metrics(operation_type: str = "mongodb") -> dict[str, Any]:
    """Get retry metrics for a specific operation type.

    Args:
        operation_type: Type of operation ("mongodb" or "api")

    Returns:
        Dictionary with retry metrics
    """
    if operation_type == "mongodb":
        return _mongodb_retry_handler.get_metrics()
    elif operation_type == "api":
        return _api_retry_handler.get_metrics()
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")
