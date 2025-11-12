#!/usr/bin/env python3
"""Integrity validation script for MCP server.

This script performs comprehensive integrity checks on the MCP server
components before allowing the server to start.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_critical_imports():
    """Check that all critical imports work."""
    print("Checking critical imports...")
    try:
        import fastmcp  # noqa: F401
        import pydantic  # noqa: F401
        import pymongo  # noqa: F401

        from src.config.settings import settings  # noqa: F401
        from src.mcp_server.database.connection import get_connection_manager  # noqa: F401
        from src.mcp_server.tools.models import ErrorResponse  # noqa: F401
        from src.mcp_server.tools.utils import handle_mongo_errors  # noqa: F401

        print("[PASS] All critical imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def check_configuration():
    """Check that configuration is valid."""
    print("Checking configuration...")
    try:
        from src.config.settings import settings

        has_mongo_uri = bool(getattr(settings, "mongodb_connection_string", ""))
        has_database = bool(getattr(settings, "mongodb_database", ""))

        if has_mongo_uri and has_database:
            print("[PASS] Configuration valid")
            return True
        else:
            print("[FAIL] Configuration incomplete")
            if not has_mongo_uri:
                print("  Missing: mongodb_connection_string")
            if not has_database:
                print("  Missing: mongodb_database")
            return False
    except Exception as e:
        print(f"[FAIL] Configuration check failed: {e}")
        return False


def check_database_connection():
    """Check that database connection works."""
    print("Checking database connection...")
    try:
        from src.mcp_server.database.connection import get_connection_manager

        manager = get_connection_manager()

        if not manager.is_connected():
            success = manager.connect()
            if not success:
                print("[FAIL] Failed to establish database connection")
                return False

        if not manager.health_check():
            print("[FAIL] Database health check failed")
            return False

        print("[PASS] Database connection healthy")
        return True

    except Exception as e:
        print(f"[FAIL] Database connection check failed: {e}")
        return False


def check_connection_manager():
    """Check connection manager improvements."""
    print("Checking connection manager...")
    try:
        from src.mcp_server.database.connection import get_connection_manager

        # Test singleton pattern
        manager1 = get_connection_manager()
        manager2 = get_connection_manager()
        assert manager1 is manager2, "Singleton pattern not working"

        # Test methods exist
        assert hasattr(manager1, "connect"), "connect method missing"
        assert hasattr(manager1, "disconnect"), "disconnect method missing"
        assert hasattr(manager1, "health_check"), "health_check method missing"
        assert hasattr(manager1, "is_connected"), "is_connected method missing"

        print("[PASS] Connection manager working correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Connection manager check failed: {e}")
        return False


def check_base_tool():
    """Check BaseTool improvements."""
    print("Checking BaseTool...")
    try:
        from src.mcp_server.tools.base_tool import BaseTool

        # Test class methods exist
        assert hasattr(BaseTool, "get_shared_database"), "get_shared_database method missing"
        assert hasattr(BaseTool, "get_connection_manager"), "get_connection_manager method missing"
        assert hasattr(BaseTool, "health_check"), "health_check method missing"

        print("[PASS] BaseTool improvements working")
        return True
    except Exception as e:
        print(f"[FAIL] BaseTool check failed: {e}")
        return False


def check_error_handling():
    """Check error handling improvements."""
    print("Checking error handling...")
    try:
        from src.mcp_server.tools.models import ErrorResponse
        from src.mcp_server.tools.utils import handle_errors_sync, handle_mongo_errors

        # Test ErrorResponse creation
        err = ErrorResponse(error="test", operation="test_op")
        err_dict = err.model_dump()
        assert "error" in err_dict and "operation" in err_dict, "ErrorResponse serialization failed"

        # Test decorators are callable
        assert callable(handle_mongo_errors), "handle_mongo_errors not callable"
        assert callable(handle_errors_sync), "handle_errors_sync not callable"

        print("[PASS] Error handling improvements working")
        return True
    except Exception as e:
        print(f"[FAIL] Error handling check failed: {e}")
        return False


def check_timeout_handling():
    """Check timeout handling in database operations."""
    print("Checking timeout handling...")
    try:
        from src.config.settings import settings

        # Check MongoDB timeout is configured
        timeout = getattr(settings, "mongodb_timeout", None)
        if timeout is None or timeout <= 0:
            print("[FAIL] MongoDB timeout not configured")
            return False

        if timeout > 300:  # 5 minutes
            print(f"[WARN] MongoDB timeout ({timeout}s) is very high")
        elif timeout < 5:  # 5 seconds
            print(f"[WARN] MongoDB timeout ({timeout}s) is very low")

        # Test connection manager timeout handling
        from src.mcp_server.database.connection import get_connection_manager

        manager = get_connection_manager()

        # Check if timeout attributes exist
        has_timeout_attrs = hasattr(manager, "MAX_RETRIES") and hasattr(manager, "RETRY_DELAY")

        if not has_timeout_attrs:
            print("[FAIL] Connection manager missing timeout/retry attributes")
            return False

        print("[PASS] Timeout handling properly configured")
        return True
    except Exception as e:
        print(f"[FAIL] Timeout check failed: {e}")
        return False


def check_operational_integrity():
    """Check operational integrity including connection pooling."""
    print("Checking operational integrity...")
    try:
        from src.mcp_server.integrity import OperationalIntegrityChecker

        checker = OperationalIntegrityChecker()
        all_checks = checker.run_operational_checks()

        passed = 0
        total = len(all_checks)

        for check_name, _category, _severity, check_func in all_checks:
            try:
                result_passed, message, details, recommendations = check_func()
                if result_passed:
                    passed += 1
                    print(f"  [PASS] {check_name}")
                else:
                    print(f"  [FAIL] {check_name}: {message}")
                    if recommendations:
                        print(f"    -> {recommendations[0]}")
            except Exception as e:
                print(f"  [ERROR] {check_name}: {e}")

        success_rate = passed / total if total > 0 else 0
        if success_rate >= 0.8:  # 80% pass rate
            print(f"[PASS] Operational integrity: {passed}/{total} checks passed")
            return True
        else:
            print(f"[FAIL] Operational integrity: Only {passed}/{total} checks passed")
            return False

    except ImportError:
        print("[FAIL] Operational integrity checker not available")
        return False
    except Exception as e:
        print(f"[FAIL] Operational integrity check failed: {e}")
        return False


def run_integrity_checks():
    """Run all integrity checks."""
    print("=" * 60)
    print("MCP SERVER INTEGRITY VALIDATION")
    print("=" * 60)

    checks = [
        check_critical_imports,
        check_configuration,
        check_database_connection,
        check_connection_manager,
        check_base_tool,
        check_error_handling,
        check_timeout_handling,
        check_operational_integrity,
    ]

    passed = 0
    total = len(checks)

    for check_func in checks:
        if check_func():
            passed += 1
        print()

    print("=" * 60)
    print(f"VALIDATION SUMMARY: {passed}/{total} checks passed")
    if passed == total:
        print("Status: ALL CHECKS PASSED - Server can start safely")
        return True
    else:
        print("Status: CHECKS FAILED - Server startup blocked")
        print("Please fix the failed checks before starting the server.")
        return False


if __name__ == "__main__":
    success = run_integrity_checks()
    sys.exit(0 if success else 1)
