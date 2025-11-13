"""Authorization Decorators and Audit System

This module provides decorators and utilities for enforcing authentication
and audit logging in the healthcare MCP server to ensure HIPAA compliance.

Rationale for Authorization Decorators:
- Declarative Security: Authentication declared at function level
- Consistent Enforcement: Same security checks across all tools
- Audit Integration: Automatic logging of all access attempts
- Read-Only Safeguard: MCP servers are read-only by design

Permission Constants Rationale:
- Centralized permissions prevent typos and inconsistencies
- Clear naming helps developers understand access requirements
"""

import logging
from collections.abc import Callable
from functools import wraps

from .audit import AuditLogger
from .authentication import SecurityContext, UserRole, get_security_context

# Permission Constants
# Rationale: Centralized permission definitions prevent typos and ensure
# consistent access control across the application. Each permission
# corresponds to HIPAA-required access controls.
#
# SECURITY BEST PRACTICE: MCP servers are read-only by design. No write
# permissions exist because MCP tools should never modify data. This reduces
# attack surface and ensures data server_health_checks. All data modifications must
# occur through separate, secured APIs with proper authorization.

PHI_PERMISSIONS = {
    "read_phi": "Read protected health information (read-only)",
    "read_financial": "Read financial/insurance information (read-only)",
    "read_aggregated_data": "Read aggregated/de-identified data only (read-only)",
    "read_deidentified_data": "Read de-identified health data (read-only)",
    # Note: No write permissions - MCP servers are read-only by design
}

ADMIN_PERMISSIONS = {
    "view_audit_logs": "View audit logs and security events (read-only)",
    "system_admin": "System administration functions (read-only access)",
    # Note: User management and write operations are handled outside MCP
}

# Combined permission dictionary for easy lookup
ALL_PERMISSIONS = {**PHI_PERMISSIONS, **ADMIN_PERMISSIONS}

# Module-level logger for authorization operations
logger = logging.getLogger(__name__)


def require_auth():
    """Decorator to enforce authentication and audit logging on MCP tools

    Usage:
        @require_auth()
        async def search_patients(...):
            # Tool implementation

    Rationale: Declarative security ensures authentication is checked on every
    tool invocation and all access is audited for HIPAA compliance.

    Security Best Practice: MCP servers are read-only by design. No write
    operations are permitted through MCP tools, ensuring data integrity.

    Security Benefits:
    - Authentication: Every request requires valid security context
    - Audit trail: All access attempts logged for compliance
    - Read-only: No write operations possible by design

    Returns:
        Decorated function with security enforcement
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            """Wrapper function that enforces security before executing tool"""

            # Initialize security components
            from src.config.settings import settings

            config = settings.security_config
            auth_manager = None  # Will be initialized from security manager
            audit_logger = AuditLogger(config)

            try:
                # Extract security context
                # In production, this would come from FastMCP request context
                # For now, use the development helper function
                try:
                    context = get_security_context()
                except Exception as e:
                    error_msg = f"Failed to retrieve security context: {type(e).__name__}"
                    logger.error(f"{error_msg}: {e}", exc_info=True)
                    audit_logger.log_phi_access(
                        context=SecurityContext(
                            user_id="system_error",
                            role=UserRole.READ_ONLY,
                            session_id="none",
                            ip_address="unknown",
                        ),
                        operation=func.__name__,
                        resource_type="authentication_failed",
                        resource_ids=[],
                        query_params={},
                        result_count=0,
                        success=False,
                        error=error_msg,
                    )
                    raise PermissionError(error_msg) from e

                if not context:
                    error_msg = "Authentication required - no security context"
                    audit_logger.log_phi_access(
                        context=SecurityContext(
                            user_id="anonymous",
                            role=UserRole.READ_ONLY,
                            session_id="none",
                            ip_address="unknown",
                        ),
                        operation=func.__name__,
                        resource_type="authentication_failed",
                        resource_ids=[],
                        query_params={},
                        result_count=0,
                        success=False,
                        error=error_msg,
                    )
                    raise PermissionError(error_msg)

                # Validate security context
                if not isinstance(context, SecurityContext):
                    error_msg = f"Invalid security context type: {type(context).__name__}"
                    logger.error(error_msg)
                    raise PermissionError(error_msg)

                # Execute the tool function
                logger.debug(f"Executing secured tool: {func.__name__} for user {context.user_id}")

                result = await func(*args, **kwargs)

                # Log successful access
                audit_logger.log_phi_access(
                    context=context,
                    operation=func.__name__,
                    resource_type=get_resource_type_from_function(func),
                    resource_ids=get_resource_ids_from_result(result),
                    query_params=get_safe_params(kwargs),
                    result_count=get_result_count(result),
                    success=True,
                )

                return result

            except PermissionError:
                # Re-raise permission errors (already logged above)
                raise
            except Exception as e:
                # Log unexpected errors
                try:
                    audit_logger.log_phi_access(
                        context=context
                        if "context" in locals()
                        else SecurityContext(
                            user_id="unknown",
                            role="unknown",
                            session_id="unknown",
                            ip_address="unknown",
                        ),
                        operation=func.__name__,
                        resource_type="error",
                        resource_ids=[],
                        query_params=get_safe_params(kwargs),
                        result_count=0,
                        success=False,
                        error=str(e),
                    )
                except Exception as audit_error:
                    logger.error(f"Failed to log audit event: {audit_error}")

                raise

        return wrapper

    return decorator


def require_role(required_role: str):
    """Decorator to enforce specific role requirements

    Usage:
        @require_role("clinician")
        async def update_patient_record(...):
            # Only clinicians can modify patient records

    Rationale: Role-based decorators provide simpler access control
    for common scenarios while still enforcing proper authorization.

    Args:
        required_role: Required user role

    Returns:
        Decorated function with role enforcement
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = get_security_context()

            if context.role.value != required_role:
                raise PermissionError(f"Required role: {required_role}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def check_permission(context: SecurityContext, permission: str) -> bool:
    """Check if security context has specific permission

    Rationale: Utility function for permission checking outside decorators.
    Useful for conditional logic within tool implementations.

    Args:
        context: Security context to check
        permission: Permission to verify

    Returns:
        bool: True if permission granted
    """
    return context.has_permission(permission)


def get_resource_type_from_function(func: Callable) -> str:
    """Extract resource type from function name

    Rationale: Automatic resource type detection for audit logging.
    Helps categorize audit events without manual specification.

    Args:
        func: Function being executed

    Returns:
        str: Resource type category
    """
    func_name = func.__name__.lower()

    # Map function names to resource types
    if "patient" in func_name:
        return "patient"
    elif "condition" in func_name or "diagnos" in func_name:
        return "condition"
    elif "medication" in func_name or "drug" in func_name:
        return "medication"
    elif "financial" in func_name or "billing" in func_name or "claim" in func_name:
        return "financial"
    elif "audit" in func_name or "log" in func_name:
        return "audit"
    else:
        return "unknown"


def get_resource_ids_from_result(result) -> list:
    """Extract resource IDs from function result

    Rationale: Automatic ID extraction for comprehensive audit trails.
    Enables tracking of specific resources accessed in each operation.

    Args:
        result: Function result to analyze

    Returns:
        List of resource IDs found in result
    """
    ids = []

    try:
        if isinstance(result, dict):
            # Look for common ID fields
            for key, value in result.items():
                if "id" in key.lower() and isinstance(value, str | int):
                    ids.append(str(value))
                elif isinstance(value, list):
                    # Check list items for IDs
                    for item in value[:5]:  # Limit to prevent huge logs
                        if isinstance(item, dict) and "id" in item:
                            ids.append(str(item["id"]))

        elif isinstance(result, list):
            # Check list items
            for item in result[:5]:  # Limit processing
                if isinstance(item, dict) and "id" in item:
                    ids.append(str(item["id"]))

    except Exception:
        # If extraction fails, return empty list rather than failing
        pass

    return ids[:10]  # Limit for log size


def get_result_count(result) -> int:
    """Get result count for audit logging

    Rationale: Track data volume in audit logs for HIPAA compliance.
    Helps detect unusual access patterns.

    Args:
        result: Function result

    Returns:
        int: Number of results returned
    """
    try:
        if isinstance(result, dict):
            count = result.get("count", 0)
            if count > 0:
                return count
            # Check for list fields
            for value in result.values():
                if isinstance(value, list):
                    return len(value)
        elif isinstance(result, list):
            return len(result)
        return 1  # Single result
    except Exception:
        return 0


def get_safe_params(kwargs: dict) -> dict:
    """Get safe parameters for audit logging

    Rationale: Remove sensitive data from audit logs while preserving
    useful information for security analysis.

    Args:
        kwargs: Function keyword arguments

    Returns:
        dict: Sanitized parameters for logging
    """
    safe_params = {}
    sensitive_keys = {"password", "token", "key", "secret", "session_id"}

    for key, value in kwargs.items():
        if key.lower() in sensitive_keys:
            safe_params[key] = "[REDACTED]"
        elif isinstance(value, str | int | float | bool):
            safe_params[key] = value
        else:
            safe_params[key] = f"<{type(value).__name__}>"

    return safe_params
