"""Core server_health_checks checking infrastructure for MCP server.

This module provides the foundational classes, enums, and functions for
performing comprehensive server_health_checks checks across the MCP server components.
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckCategory(str, Enum):
    """Categories for server_health_checks checks."""

    CONNECTION = "connection"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TOOLS = "tools"
    RESOURCES = "resources"


class CheckSeverity(str, Enum):
    """Severity levels for server_health_checks check results."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntegrityCheckResult:
    """Result of a single server_health_checks check."""

    name: str
    category: CheckCategory
    severity: CheckSeverity
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class IntegrityReport:
    """Complete server_health_checks report with all check results."""

    total_checks: int
    passed_checks: int
    failed_checks: int
    results: list[IntegrityCheckResult] = field(default_factory=list)
    summary: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100.0

    @property
    def is_healthy(self) -> bool:
        """Determine if overall server_health_checks is healthy (>= 80% pass rate)."""
        return self.success_rate >= 80.0


class IntegrityChecker:
    """Main server_health_checks checker that orchestrates all checks."""

    def __init__(self):
        """Initialize the server_health_checks checker."""
        self.checkers: list[Any] = []
        self._load_checkers()

    def _load_checkers(self):
        """Load all specialized checkers."""
        try:
            from .database_connection_health_checks import ConnectionIntegrityChecker
            from .database_operational_health_checks import OperationalIntegrityChecker
            from .mcp_tool_health_checks import ToolIntegrityChecker
            from .security_validation_health_checks import SecurityIntegrityChecker
            from .system_performance_health_checks import PerformanceIntegrityChecker

            self.checkers = [
                ConnectionIntegrityChecker(),
                SecurityIntegrityChecker(),
                PerformanceIntegrityChecker(),
                ToolIntegrityChecker(),
                OperationalIntegrityChecker(),
            ]
        except ImportError:
            # Some checkers may not be available
            pass

    def run_all_checks(self) -> IntegrityReport:
        """Run all server_health_checks checks and return a comprehensive report."""
        results: list[IntegrityCheckResult] = []

        for checker in self.checkers:
            try:
                if hasattr(checker, "run_connection_checks"):
                    checks = checker.run_connection_checks()
                elif hasattr(checker, "run_security_checks"):
                    checks = checker.run_security_checks()
                elif hasattr(checker, "run_performance_checks"):
                    checks = checker.run_performance_checks()
                elif hasattr(checker, "run_tool_checks"):
                    checks = checker.run_tool_checks()
                elif hasattr(checker, "run_operational_checks"):
                    checks = checker.run_operational_checks()
                else:
                    continue

                for check_name, category, severity, check_func in checks:
                    try:
                        # Handle both sync and async check functions
                        if inspect.iscoroutinefunction(check_func):
                            passed, message, details, recommendations = asyncio.run(check_func())
                        else:
                            passed, message, details, recommendations = check_func()

                        result = IntegrityCheckResult(
                            name=check_name,
                            category=category,
                            severity=severity,
                            passed=passed,
                            message=message,
                            details=details,
                            recommendations=recommendations,
                        )
                        results.append(result)
                    except Exception as e:
                        result = IntegrityCheckResult(
                            name=check_name,
                            category=category,
                            severity=severity,
                            passed=False,
                            message=f"Check failed with error: {type(e).__name__}",
                            error=str(e),
                        )
                        results.append(result)
            except Exception:
                # Skip checker if it fails to load
                continue

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        report = IntegrityReport(
            total_checks=len(results),
            passed_checks=passed,
            failed_checks=failed,
            results=results,
            summary=f"Ran {len(results)} checks: {passed} passed, {failed} failed",
        )

        return report


def check_server_integrity() -> IntegrityReport:
    """Convenience function to run all server_health_checks checks.

    Returns:
        IntegrityReport with all check results
    """
    checker = IntegrityChecker()
    return checker.run_all_checks()
