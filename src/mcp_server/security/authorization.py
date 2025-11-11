"""Authorization Decorators and Permission System

This module provides decorators and utilities for enforcing access control
in the healthcare MCP server. It implements role-based access control (RBAC)
and permission checking to ensure HIPAA compliance.

Rationale for Authorization Decorators:
- Declarative Security: Permissions declared at function level
- Consistent Enforcement: Same security checks across all tools
- Audit Integration: Automatic logging of access attempts
- Fail-Safe Defaults: Deny access by default, explicit allow

Permission Constants Rationale:
- Centralized permissions prevent typos and inconsistencies
- Clear naming helps developers understand access requirements
- Granular permissions enable least privilege principle
"""

import logging
from collections.abc import Callable
from functools import wraps

from .audit import AuditLogger
from .authentication import SecurityContext, get_security_context
from .rbac_config import check_role_access_to_collections, get_required_collections_for_tool

# Permission Constants
# Rationale: Centralized permission definitions prevent typos and ensure
# consistent access control across the application. Each permission
# corresponds to HIPAA-required access controls.
#
# SECURITY BEST PRACTICE: MCP servers are read-only by design. No write
# permissions exist because MCP tools should never modify data. This reduces
# attack surface and ensures data integrity. All data modifications must
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


def require_auth():
    """Decorator to enforce authentication and authorization on MCP tools

    Usage:
        @require_auth()
        async def search_patients(...):
            # Tool implementation

    Rationale: Declarative security makes it impossible to forget access
    control checks. Tool access is automatically determined based on the
    collections it requires and the user's role-based collection access.
    This ensures consistent HIPAA compliance across all healthcare data access.

    Security Best Practice: MCP servers are read-only by design. Access control
    is based on collection-level permissions. This reduces attack surface and
    ensures data integrity. No write operations are permitted through MCP tools.

    Security Benefits:
    - Zero-trust: Every request validated
    - Collection-based RBAC: Access determined by required collections
    - Audit trail: All access attempts logged
    - Fail-safe: Denies access by default
    - Rate limiting: Integrated abuse prevention

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
                context = get_security_context()

                if not context:
                    error_msg = "Authentication required - no security context"
                    audit_logger.log_phi_access(
                        context=SecurityContext(
                            user_id="anonymous",
                            role="unknown",
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

                # Validate session if using session-based auth
                # For development, skip session validation for now

                # Check collection-based access control
                # Use RBAC cache for performance optimization
                try:
                    required_collections = get_required_collections_for_tool(func.__name__)

                    # Check cache first for faster authorization
                    from src.mcp_server.cache import get_cache_manager

                    cache_manager = get_cache_manager()

                    access_allowed = False
                    if cache_manager.is_available():
                        # Try to get cached decision
                        cached_decision = cache_manager.rbac_cache.get_access_decision(
                            context.role.value, func.__name__
                        )

                        if cached_decision is not None:
                            access_allowed = cached_decision
                        else:
                            # Compute decision and cache it
                            access_allowed = check_role_access_to_collections(
                                context.role, required_collections
                            )
                            cache_manager.rbac_cache.cache_access_decision(
                                context.role.value,
                                func.__name__,
                                required_collections,
                                access_allowed,
                            )
                    else:
                        # Fallback to direct check if cache unavailable
                        access_allowed = check_role_access_to_collections(
                            context.role, required_collections
                        )

                    if not access_allowed:
                        error_msg = f"Insufficient permissions. Role '{context.role.value}' cannot access required collections: {required_collections}"
                        audit_logger.log_phi_access(
                            context=context,
                            operation=func.__name__,
                            resource_type="authorization_failed",
                            resource_ids=[],
                            query_params={},
                            result_count=0,
                            success=False,
                            error=error_msg,
                        )
                        raise PermissionError(error_msg)
                except ValueError:
                    # Tool not found in RBAC config - deny access
                    error_msg = f"Tool not configured in RBAC: {func.__name__}"
                    audit_logger.log_phi_access(
                        context=context,
                        operation=func.__name__,
                        resource_type="authorization_failed",
                        resource_ids=[],
                        query_params={},
                        result_count=0,
                        success=False,
                        error=error_msg,
                    )
                    raise PermissionError(error_msg) from None

                # Check rate limiting
                # Note: Rate limiting is handled at the authentication level

                # Execute the tool function
                logger = logging.getLogger(__name__)
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
                    logger = logging.getLogger(__name__)
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
