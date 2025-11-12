"""Performance server_health_checks checks for MCP server.

This module provides specialized checks for performance monitoring,
resource usage, response times, and system health metrics.
"""

import threading
import time
from typing import Any

import psutil

from .health_check_framework import CheckCategory, CheckSeverity


class PerformanceIntegrityChecker:
    """Specialized checker for performance-related server_health_checks."""

    def __init__(self):
        self.baseline_metrics = {}
        self._lock = threading.Lock()

    def check_memory_usage(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check memory usage and detect potential leaks."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            # Get memory metrics
            rss_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            vms_mb = memory_info.vms / 1024 / 1024  # Convert to MB

            # Get system memory
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent

            # Memory thresholds
            max_reasonable_memory_mb = 500  # 500MB max for MCP server
            max_system_memory_percent = 90  # 90% system memory usage

            memory_reasonable = rss_mb < max_reasonable_memory_mb
            system_memory_ok = system_memory_percent < max_system_memory_percent

            passed = memory_reasonable and system_memory_ok

            details = {
                "process_memory_mb": round(rss_mb, 2),
                "virtual_memory_mb": round(vms_mb, 2),
                "system_memory_percent": round(system_memory_percent, 2),
                "memory_reasonable": memory_reasonable,
                "system_memory_ok": system_memory_ok,
                "max_reasonable_memory_mb": max_reasonable_memory_mb,
            }

            recommendations = []
            if not memory_reasonable:
                recommendations.append(".2f")
                recommendations.append("Monitor for memory leaks")
                recommendations.append("Consider optimizing data structures")
            if not system_memory_ok:
                recommendations.append(".1f")
                recommendations.append("Check system memory usage")
                recommendations.append(
                    "Consider increasing system memory or optimizing application"
                )

            return passed, f"Memory usage {'OK' if passed else 'high'}", details, recommendations

        except Exception as e:
            return (
                False,
                f"Memory usage check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Install psutil for memory monitoring",
                    "Check system resource access",
                    "Review memory monitoring setup",
                ],
            )

    def check_cpu_usage(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check CPU usage and thread health."""
        try:
            process = psutil.Process()

            # Get CPU usage (sample for 1 second)
            cpu_percent = process.cpu_percent(interval=1.0)

            # Get thread information
            threads = process.threads()
            thread_count = len(threads)

            # CPU thresholds
            max_reasonable_cpu_percent = 80  # 80% CPU usage
            max_reasonable_threads = 50  # Max 50 threads

            cpu_reasonable = cpu_percent < max_reasonable_cpu_percent
            threads_reasonable = thread_count < max_reasonable_threads

            passed = cpu_reasonable and threads_reasonable

            details = {
                "cpu_percent": round(cpu_percent, 2),
                "thread_count": thread_count,
                "cpu_reasonable": cpu_reasonable,
                "threads_reasonable": threads_reasonable,
                "max_cpu_percent": max_reasonable_cpu_percent,
                "max_threads": max_reasonable_threads,
            }

            recommendations = []
            if not cpu_reasonable:
                recommendations.append(".1f")
                recommendations.append("Monitor for CPU-intensive operations")
                recommendations.append("Consider optimizing algorithms")
            if not threads_reasonable:
                recommendations.append(
                    f"High thread count ({thread_count}) - check for thread leaks"
                )
                recommendations.append("Review concurrent operation management")

            return passed, f"CPU usage {'OK' if passed else 'high'}", details, recommendations

        except Exception as e:
            return (
                False,
                f"CPU usage check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Install psutil for CPU monitoring",
                    "Check system resource access",
                    "Review performance monitoring setup",
                ],
            )

    async def check_database_performance(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check database query performance."""
        try:
            from src.mcp_server.database import database

            # Test if database is initialized
            try:
                db = database.get_database()
            except RuntimeError:
                return (
                    False,
                    "Database not initialized",
                    {"initialized": False},
                    ["Initialize database before performance testing"],
                )

            # Test basic query performance
            query_times = []

            for _i in range(3):
                start_time = time.time()
                try:
                    # Simple collection list operation (async)
                    await db.list_collection_names()
                    end_time = time.time()
                    query_time = (end_time - start_time) * 1000  # Convert to ms
                    query_times.append(query_time)
                except Exception:
                    query_times.append(float("inf"))

            # Calculate metrics
            valid_times = [t for t in query_times if t != float("inf")]
            if valid_times:
                avg_query_time = sum(valid_times) / len(valid_times)
                max_query_time = max(valid_times)
                min_query_time = min(valid_times)

                # Performance thresholds
                max_acceptable_query_time = 500  # 500ms max for basic queries
                passed = avg_query_time < max_acceptable_query_time
            else:
                avg_query_time = float("inf")
                max_query_time = float("inf")
                min_query_time = float("inf")
                passed = False

            details = {
                "queries_tested": len(query_times),
                "successful_queries": len(valid_times),
                "avg_query_time_ms": round(avg_query_time, 2)
                if avg_query_time != float("inf")
                else "N/A",
                "max_query_time_ms": round(max_query_time, 2)
                if max_query_time != float("inf")
                else "N/A",
                "min_query_time_ms": round(min_query_time, 2)
                if min_query_time != float("inf")
                else "N/A",
                "performance_acceptable": passed,
                "max_acceptable_time_ms": max_acceptable_query_time,
            }

            recommendations = []
            if not passed:
                recommendations.append(".2f")
                recommendations.append("Check database server performance")
                recommendations.append("Review query optimization")
                recommendations.append("Consider database indexing")
            if len(valid_times) < len(query_times):
                recommendations.append("Some database queries failed - check connectivity")

            return (
                passed,
                f"Database performance {'good' if passed else 'slow'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Database performance check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check database connection",
                    "Verify MongoDB server status",
                    "Review query patterns",
                ],
            )

    def check_response_time_baseline(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Establish and check response time baselines."""
        try:
            # Test basic tool import and instantiation time
            start_time = time.time()
            from src.mcp_server.tools._healthcare.patient_tools import PatientTools

            import_time = (time.time() - start_time) * 1000

            start_time = time.time()
            PatientTools()
            instantiation_time = (time.time() - start_time) * 1000

            # Performance thresholds
            max_import_time = 2000  # 2 seconds max for imports
            max_instantiation_time = 500  # 0.5 seconds max for instantiation

            import_fast = import_time < max_import_time
            instantiation_fast = instantiation_time < max_instantiation_time

            passed = import_fast and instantiation_fast

            details = {
                "import_time_ms": round(import_time, 2),
                "instantiation_time_ms": round(instantiation_time, 2),
                "import_fast": import_fast,
                "instantiation_fast": instantiation_fast,
                "max_import_time_ms": max_import_time,
                "max_instantiation_time_ms": max_instantiation_time,
            }

            recommendations = []
            if not import_fast:
                recommendations.append(".2f")
                recommendations.append("Optimize module imports")
                recommendations.append("Check for circular imports")
            if not instantiation_fast:
                recommendations.append(".2f")
                recommendations.append("Optimize tool initialization")
                recommendations.append("Review dependency loading")

            return passed, f"Response time {'good' if passed else 'slow'}", details, recommendations

        except Exception as e:
            return (
                False,
                f"Response time check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check tool implementations",
                    "Review import dependencies",
                    "Optimize initialization code",
                ],
            )

    def check_resource_limits(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check system resource limits and usage."""
        try:
            # Check file descriptor limits (Unix-like systems)
            try:
                import resource

                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                file_descriptors_available = soft_limit
            except (ImportError, AttributeError):
                # Windows or resource module not available
                file_descriptors_available = "N/A"

            # Check disk space
            disk_usage = psutil.disk_usage("/")
            disk_free_gb = disk_usage.free / (1024**3)
            disk_percent = disk_usage.percent

            # Check network connections
            network_connections = len(psutil.net_connections())

            # Resource thresholds
            min_disk_free_gb = 1.0  # 1GB minimum free space
            max_disk_usage_percent = 95  # 95% max disk usage
            max_network_connections = 1000  # Max reasonable connections

            disk_space_ok = disk_free_gb > min_disk_free_gb
            disk_usage_ok = disk_percent < max_disk_usage_percent
            connections_reasonable = network_connections < max_network_connections

            passed = disk_space_ok and disk_usage_ok

            details = {
                "file_descriptors_available": file_descriptors_available,
                "disk_free_gb": round(disk_free_gb, 2),
                "disk_usage_percent": round(disk_percent, 2),
                "network_connections": network_connections,
                "disk_space_ok": disk_space_ok,
                "disk_usage_ok": disk_usage_ok,
                "connections_reasonable": connections_reasonable,
                "min_disk_free_gb": min_disk_free_gb,
                "max_disk_usage_percent": max_disk_usage_percent,
            }

            recommendations = []
            if not disk_space_ok:
                recommendations.append(".2f")
                recommendations.append("Free up disk space or add more storage")
            if not disk_usage_ok:
                recommendations.append(".1f")
                recommendations.append("Monitor disk usage trends")
            if not connections_reasonable:
                recommendations.append(
                    f"High network connections ({network_connections}) - check for connection leaks"
                )

            return (
                passed,
                f"Resource limits {'OK' if passed else 'exceeded'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Resource limits check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Install psutil for resource monitoring",
                    "Check system resource access",
                    "Review resource monitoring setup",
                ],
            )

    def run_performance_checks(self) -> list[tuple[str, CheckCategory, CheckSeverity, callable]]:
        """Return all performance checks to be executed."""
        return [
            (
                "Memory Usage",
                CheckCategory.RESOURCES,
                CheckSeverity.MEDIUM,
                self.check_memory_usage,
            ),
            ("CPU Usage", CheckCategory.PERFORMANCE, CheckSeverity.MEDIUM, self.check_cpu_usage),
            (
                "Database Performance",
                CheckCategory.PERFORMANCE,
                CheckSeverity.HIGH,
                self.check_database_performance,
            ),
            (
                "Response Time Baseline",
                CheckCategory.PERFORMANCE,
                CheckSeverity.MEDIUM,
                self.check_response_time_baseline,
            ),
            (
                "Resource Limits",
                CheckCategory.RESOURCES,
                CheckSeverity.MEDIUM,
                self.check_resource_limits,
            ),
        ]
