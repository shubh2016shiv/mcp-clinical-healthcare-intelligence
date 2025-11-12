"""Connection server_health_checks checks for MCP server.

This module provides specialized checks for database connections,
network connectivity, connection pooling, and related infrastructure.
"""

import time
from typing import Any

from .health_check_framework import CheckCategory, CheckSeverity


class ConnectionIntegrityChecker:
    """Specialized checker for connection-related server_health_checks."""

    def __init__(self):
        self.database_module = None
        self._load_database_module()

    def _load_database_module(self):
        """Load database module safely."""
        try:
            from src.mcp_server.database import database

            self.database_module = database
        except Exception:
            self.database_module = None

    def check_connection_pooling(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check connection pooling configuration and health."""
        if not self.database_module:
            return (
                False,
                "Database module not available",
                {"module_available": False},
                ["Initialize database module before running checks"],
            )

        try:
            from src.config.settings import settings

            # Check pool settings (Motor uses these in initialize function)
            min_pool = getattr(settings, "mongodb_min_pool_size", 10)
            max_pool = getattr(settings, "mongodb_max_pool_size", 50)

            pool_config_valid = min_pool >= 0 and max_pool > min_pool and max_pool <= 100

            # Motor handles connection pooling automatically
            # We can't easily test pooling without initializing the database
            # But we can check that the configuration is reasonable

            passed = pool_config_valid

            details = {
                "min_pool_size": min_pool,
                "max_pool_size": max_pool,
                "pool_config_valid": pool_config_valid,
                "motor_handles_pooling": True,  # Motor manages pooling internally
            }

            recommendations = []
            if not pool_config_valid:
                recommendations.append(
                    "Fix connection pool configuration (10 <= min_pool_size < max_pool_size <= 100)"
                )
                if max_pool > 100:
                    recommendations.append(
                        "Consider reducing max_pool_size for resource management"
                    )

            return (
                passed,
                f"Connection pooling {'OK' if passed else 'has configuration issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Connection pooling check failed: {type(e).__name__}",
                {"error": str(e)},
                ["Check connection pool settings", "Verify Motor configuration"],
            )

    def check_connection_resilience(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check connection resilience and recovery mechanisms."""
        if not self.database_module:
            return (
                False,
                "Database module not available",
                {"module_available": False},
                ["Initialize database module before running checks"],
            )

        try:
            import asyncio

            from src.config.settings import settings

            # Initialize database for testing
            mongodb_uri = getattr(
                settings, "mongodb_connection_string", "mongodb://localhost:27017"
            )
            database_name = getattr(settings, "mongodb_database", "healthcare")

            self.database_module.initialize(mongodb_uri, database_name)

            # Test health check (Motor handles reconnection automatically)
            health_ok = asyncio.run(self.database_module.health_check())

            # Motor provides automatic reconnection, so we mainly check health
            passed = health_ok

            details = {
                "motor_auto_reconnection": True,  # Motor handles this automatically
                "health_check_passed": health_ok,
                "database_initialized": True,
                "resilience_test_passed": passed,
            }

            recommendations = []
            if not health_ok:
                recommendations.append(
                    "Database health check failed - check MongoDB server and network"
                )
                recommendations.append("Verify mongodb_connection_string in settings")
                recommendations.append("Check MongoDB server logs")

            return (
                passed,
                f"Connection resilience {'OK' if passed else 'has issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Connection resilience check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check Motor database module",
                    "Verify network stability",
                    "Review MongoDB connection settings",
                ],
            )

    def check_connection_performance(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check connection performance and latency."""
        if not self.database_module:
            return (
                False,
                "Database module not available",
                {"module_available": False},
                ["Initialize database module before running checks"],
            )

        try:
            import asyncio

            # Test connection latency using Motor health check
            latencies = []

            async def test_latency():
                for _i in range(3):  # Test 3 times (fewer for async)
                    start_time = time.time()
                    health_ok = await self.database_module.health_check()
                    end_time = time.time()

                    if health_ok:
                        latency = (end_time - start_time) * 1000  # Convert to milliseconds
                        latencies.append(latency)
                    else:
                        break

            asyncio.run(test_latency())

            if not latencies:
                return (
                    False,
                    "Connection performance test failed - no successful health checks",
                    {"tests_run": 0, "successful_tests": 0},
                    ["Fix connection issues before testing performance"],
                )

            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)

            # Performance thresholds (Motor should be fast)
            max_acceptable_latency = 500  # 500ms for Motor
            passed = avg_latency < max_acceptable_latency

            details = {
                "tests_run": 3,
                "successful_tests": len(latencies),
                "avg_latency_ms": round(avg_latency, 2),
                "max_latency_ms": round(max_latency, 2),
                "min_latency_ms": round(min_latency, 2),
                "latency_threshold_ms": max_acceptable_latency,
                "motor_async_performance": True,
            }

            recommendations = []
            if not passed:
                recommendations.append(
                    f"Average latency ({avg_latency:.2f}ms) exceeds threshold ({max_acceptable_latency}ms)"
                )
                recommendations.append("Check network latency to MongoDB server")
                recommendations.append("Consider serverSelectionTimeoutMS settings")

            return (
                passed,
                f"Connection performance {'good' if passed else 'slow'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Connection performance check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check Motor database module",
                    "Verify network performance",
                    "Review Motor timeout configurations",
                ],
            )

    def run_connection_checks(self) -> list[tuple[str, CheckCategory, CheckSeverity, callable]]:
        """Return all connection checks to be executed."""
        return [
            (
                "Connection Pooling",
                CheckCategory.CONNECTION,
                CheckSeverity.HIGH,
                self.check_connection_pooling,
            ),
            (
                "Connection Resilience",
                CheckCategory.CONNECTION,
                CheckSeverity.HIGH,
                self.check_connection_resilience,
            ),
            (
                "Connection Performance",
                CheckCategory.PERFORMANCE,
                CheckSeverity.MEDIUM,
                self.check_connection_performance,
            ),
        ]
