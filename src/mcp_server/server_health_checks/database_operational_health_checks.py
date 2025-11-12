"""Operational server_health_checks checks for database connections, timeouts, and performance."""

import logging
from collections.abc import Callable
from typing import Any

from src.mcp_server.server_health_checks.health_check_framework import CheckCategory, CheckSeverity

logger = logging.getLogger(__name__)


class OperationalIntegrityChecker:
    """Performs operational server_health_checks checks for database operations and performance."""

    def run_operational_checks(self) -> list[tuple[str, CheckCategory, CheckSeverity, Callable]]:
        """Define and return all operational-related checks."""
        return [
            (
                "Database Query Timeout",
                CheckCategory.PERFORMANCE,
                CheckSeverity.HIGH,
                self._check_query_timeout,
            ),
            (
                "Connection Pool Configuration",
                CheckCategory.CONNECTION,
                CheckSeverity.HIGH,
                self._check_connection_pool,
            ),
            (
                "Motor Connection Configuration",
                CheckCategory.CONNECTION,
                CheckSeverity.MEDIUM,
                self._check_motor_connection,
            ),
        ]

    def _check_query_timeout(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check that database queries have proper timeout configuration."""
        try:
            from src.config.settings import settings

            # Check MongoDB timeout settings
            timeout_configured = (
                hasattr(settings, "mongodb_timeout") and settings.mongodb_timeout > 0
            )
            timeout_reasonable = (
                timeout_configured and settings.mongodb_timeout <= 300
            )  # Max 5 minutes

            passed = timeout_configured and timeout_reasonable
            message = (
                f"Database query timeout {'properly configured' if passed else 'misconfigured'}"
            )
            recommendations = []

            if not timeout_configured:
                recommendations.append(
                    "Configure mongodb_timeout in settings (recommended: 30-60 seconds)"
                )
            elif not timeout_reasonable:
                recommendations.append(
                    f"mongodb_timeout ({settings.mongodb_timeout}s) is too high, consider reducing to 30-60s"
                )

            details = {
                "mongodb_timeout": getattr(settings, "mongodb_timeout", "Not configured"),
                "timeout_reasonable": timeout_reasonable,
            }

            return passed, message, details, recommendations

        except Exception as e:
            return (
                False,
                f"Error checking query timeout: {e}",
                {"error": str(e)},
                ["Review settings configuration"],
            )

    def _check_connection_pool(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check MongoDB connection pool configuration."""
        try:
            from src.config.settings import settings

            min_pool = getattr(settings, "mongodb_min_pool_size", 0)
            max_pool = getattr(settings, "mongodb_max_pool_size", 100)

            passed = True
            message = "Connection pool properly configured"
            recommendations = []

            # Check min_pool_size
            if min_pool < 0:
                passed = False
                message = "Invalid min_pool_size (negative value)"
                recommendations.append("Set mongodb_min_pool_size >= 0")
            elif min_pool == 0:
                message += " (warning: no minimum connections)"

            # Check max_pool_size
            if max_pool <= 0:
                passed = False
                message = "Invalid max_pool_size (must be positive)"
                recommendations.append("Set mongodb_max_pool_size > 0")
            elif max_pool > 1000:
                message += " (warning: very high max_pool_size)"
                recommendations.append(
                    "Consider reducing mongodb_max_pool_size for resource management"
                )

            # Check min <= max
            if min_pool > max_pool:
                passed = False
                message = "min_pool_size > max_pool_size"
                recommendations.append("Ensure mongodb_min_pool_size <= mongodb_max_pool_size")

            details = {
                "min_pool_size": min_pool,
                "max_pool_size": max_pool,
                "pool_ratio": max_pool / max(1, min_pool) if min_pool > 0 else float("inf"),
            }

            return passed, message, details, recommendations

        except Exception as e:
            return (
                False,
                f"Error checking connection pool: {e}",
                {"error": str(e)},
                ["Review MongoDB connection settings"],
            )

    def _check_motor_connection(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check that Motor connection is properly configured for async operations."""
        try:
            from src.config.settings import settings
            from src.mcp_server.database import database

            # Check if database is initialized (should not be during validation)
            try:
                database.get_database()
                initialized = True
            except RuntimeError:
                initialized = False

            # Check that Motor functions are available
            has_initialize = hasattr(database, "initialize")
            has_get_database = hasattr(database, "get_database")
            has_health_check = hasattr(database, "health_check")

            passed = has_initialize and has_get_database and has_health_check
            message = f"Motor connection {'properly configured' if passed else 'misconfigured'}"
            recommendations = []

            if not has_initialize:
                passed = False
                recommendations.append("database.initialize function missing")
            if not has_get_database:
                passed = False
                recommendations.append("database.get_database function missing")
            if not has_health_check:
                passed = False
                recommendations.append("database.health_check function missing")

            # Motor handles connection pooling and async operations natively
            # No need for separate thread pools or executors

            details = {
                "motor_available": True,  # If we got here, motor is available
                "database_initialized": initialized,
                "has_initialize": has_initialize,
                "has_get_database": has_get_database,
                "has_health_check": has_health_check,
                "mongodb_timeout": getattr(settings, "mongodb_timeout", None),
            }

            return passed, message, details, recommendations

        except ImportError:
            return (
                False,
                "Motor not available",
                {"motor_import_failed": True},
                ["Install motor package: pip install motor"],
            )
        except Exception as e:
            return (
                False,
                f"Error checking Motor connection: {e}",
                {"error": str(e)},
                ["Review database module configuration"],
            )
