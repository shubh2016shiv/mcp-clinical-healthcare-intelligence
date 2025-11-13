"""Healthcare MCP Server Security Layer

This module implements enterprise-grade security controls for healthcare MCP servers
handling Protected Health Information (PHI) and FHIR data in MongoDB.

Security Principles Applied:
1. Defense in Depth: Multiple layers of security controls
2. Least Privilege: Minimal access rights for operations (read-only)
3. Zero Trust: Verify every request, trust nothing by default
4. Audit Everything: Comprehensive logging of PHI access
5. Fail Secure: Safe defaults that prevent data exposure
6. Read-Only Design: No write operations reduce attack surface

HIPAA/Healthcare-Specific Considerations:
- PHI access requires authentication and authorization
- All PHI access must be audited (HIPAA Security Rule)
- Data must be encrypted in transit and at rest
- Minimum necessary principle: return only required data
- Breach notification requirements mandate tracking
- Read-only operations: MCP servers never modify data

Security Best Practice: MCP servers are read-only by design. This reduces
attack surface, ensures data server_health_checks, and follows the principle of least
privilege. All data modifications must occur through separate, secured APIs.

This module provides:
- Authentication & Authorization (RBAC, read-only session management)
- Input validation & sanitization (NoSQL injection prevention)
- Audit logging (HIPAA compliance for read operations)
- Data minimization (minimum necessary principle)
- Rate limiting (prevents abuse)
- FastMCP integration (middleware support for read-only tools)
"""

from .audit import AuditLogger
from .authentication import AuthenticationManager, SecurityContext, UserRole, get_security_context
from .authorization import require_auth
from .config import SecurityConfig
from .data_minimization import DataMinimizer
from .middleware import SecurityContextMiddleware
from .validation import InputValidator

__all__ = [
    "SecurityConfig",
    "UserRole",
    "SecurityContext",
    "AuthenticationManager",
    "InputValidator",
    "AuditLogger",
    "DataMinimizer",
    "SecurityContextMiddleware",
    "require_auth",
    "get_security_context",
]


# Global security manager instance
_security_manager = None


def initialize_security() -> "SecurityManager":
    """Initialize the security layer with all components.

    Rationale: Centralized initialization ensures all security components
    are properly configured and connected at server startup. This prevents
    security gaps from misconfiguration or missing components.

    Returns:
        SecurityManager: Configured security manager instance
    """
    global _security_manager

    if _security_manager is not None:
        return _security_manager

    # Import here to avoid circular imports
    from src.config.settings import settings

    # Initialize security manager with all components
    _security_manager = SecurityManager(settings.security_config)

    return _security_manager


def get_security_manager() -> "SecurityManager":
    """Get the global security manager instance.

    Rationale: Singleton pattern ensures consistent security state across
    the application. All security operations use the same configured instance.

    Returns:
        SecurityManager: Global security manager instance

    Raises:
        RuntimeError: If security has not been initialized
    """
    if _security_manager is None:
        raise RuntimeError("Security not initialized. Call initialize_security() first.")
    return _security_manager


class SecurityManager:
    """Central security manager coordinating all security components.

    Rationale: Single point of coordination for all security operations.
    Ensures consistent application of security policies across all components.
    """

    def __init__(self, config: SecurityConfig):
        """Initialize security manager with configuration.

        Args:
            config: Security configuration with HIPAA-compliant defaults
        """
        self.config = config
        self.auth_manager = AuthenticationManager(config)
        self.validator = InputValidator(config)
        self.audit_logger = AuditLogger(config)
        self.data_minimizer = DataMinimizer()

        # Initialize cache manager for Redis-backed caching
        try:
            from src.mcp_server.cache import get_cache_manager, initialize_cache

            initialize_cache()
            self.cache_manager = get_cache_manager()
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Cache manager initialization failed: {e}")
            self.cache_manager = None

    def validate_request(
        self,
        context: SecurityContext,
        operation: str,
        query_params: dict = None,
        resource_ids: list = None,
    ) -> bool:
        """Validate a security request with all security layers.

        Rationale: Applies all security checks in sequence to ensure
        comprehensive protection against various attack vectors.

        Args:
            context: Authenticated security context
            operation: Operation being performed
            query_params: Query parameters to validate
            resource_ids: Resource IDs being accessed

        Returns:
            bool: True if all validations pass
        """
        # 1. Rate limiting check
        if not self.auth_manager.check_rate_limit(context.ip_address):
            self.audit_logger.log_phi_access(
                context=context,
                operation=operation,
                resource_type="rate_limited",
                resource_ids=[],
                query_params={},
                result_count=0,
                success=False,
                error="Rate limit exceeded",
            )
            return False

        # 2. Input validation
        if query_params:
            try:
                self.validator.validate_query_params(query_params)
            except ValueError as e:
                self.audit_logger.log_phi_access(
                    context=context,
                    operation=operation,
                    resource_type="validation_failed",
                    resource_ids=resource_ids or [],
                    query_params=query_params,
                    result_count=0,
                    success=False,
                    error=str(e),
                )
                return False

        return True

    def is_hipaa_compliant(self) -> bool:
        """Check if current security configuration meets HIPAA requirements.

        Rationale: Automated compliance checking prevents deployment
        with insecure configurations that could lead to HIPAA violations.

        Returns:
            bool: True if configuration is HIPAA compliant
        """
        return self.config.is_hipaa_compliant()

    def get_security_status(self) -> dict:
        """Get comprehensive security status for monitoring.

        Rationale: Provides visibility into security system health
        for monitoring and compliance reporting.

        Returns:
            dict: Security status information
        """
        return {
            "hipaa_compliant": self.is_hipaa_compliant(),
            "audit_enabled": self.config.enable_audit_logging,
            "security_enabled": True,  # If we get here, security is enabled
            "max_query_results": self.config.max_query_results,
            "audit_retention_days": self.config.audit_retention_days,
            "rate_limit_per_minute": self.config.rate_limit_requests_per_minute,
        }
