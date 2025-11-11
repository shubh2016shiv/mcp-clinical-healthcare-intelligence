"""FastMCP Middleware Integration for Security Context

This module provides middleware integration with FastMCP to inject security
context into MCP tool requests. It handles authentication, session management,
and security context propagation.

Rationale for Middleware Integration:
- FastMCP may not expose request context directly to tools
- Need centralized security context extraction and validation
- Enables consistent security enforcement across all tools
- Supports both development and production authentication modes

Integration Approaches:
- Environment variables (development/simple deployments)
- Request headers (production with API gateway)
- Context variables (async context propagation)
"""

import contextvars
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .authentication import SecurityContext
from .config import SecurityConfig

# Context variable for security context propagation
# Rationale: Context variables provide thread-safe, async-safe storage
# for request-scoped security context across the entire request lifecycle
security_context_var: contextvars.ContextVar[SecurityContext | None] = contextvars.ContextVar(
    "security_context", default=None
)


class SecurityContextMiddleware:
    """Middleware for injecting security context into FastMCP requests

    Rationale: FastMCP doesn't provide built-in request context management,
    so this middleware intercepts requests to establish security context
    before tool execution. This ensures all tools have access to authenticated
    user information for authorization and audit logging.

    The middleware:
    1. Extracts authentication information from request context
    2. Validates sessions and permissions
    3. Injects security context for tool access
    4. Handles authentication failures gracefully
    """

    def __init__(self, config: SecurityConfig):
        """Initialize security middleware

        Args:
            config: Security configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def __call__(self, request: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
        """Middleware execution for each request

        Rationale: This method is called for every MCP request, allowing
        security context to be established before tool execution.

        Args:
            request: FastMCP request object
            call_next: Next middleware in chain

        Returns:
            Response from next middleware or tool
        """
        try:
            # Extract security context from request
            security_context = await self._extract_security_context(request)

            # Set context for this request
            token = security_context_var.set(security_context)

            try:
                # Continue with request processing
                response = await call_next(request)
                return response
            finally:
                # Clean up context
                security_context_var.reset(token)

        except Exception as e:
            self.logger.error(f"Security middleware error: {e}")
            # Return error response instead of crashing
            return self._create_error_response(str(e))

    async def _extract_security_context(self, request: Any) -> SecurityContext:
        """Extract security context from FastMCP request

        Rationale: Different deployment scenarios require different
        authentication methods. This method tries multiple approaches
        to establish security context.

        Args:
            request: FastMCP request object

        Returns:
            SecurityContext for the request

        Raises:
            ValueError: If security context cannot be established
        """
        # Method 1: Check for API key in environment/request headers
        api_key = self._extract_api_key(request)
        if api_key:
            return await self._authenticate_with_api_key(api_key, request)

        # Method 2: Check for session token in headers
        session_token = self._extract_session_token(request)
        if session_token:
            return await self._validate_session_token(session_token, request)

        # Method 3: Development mode - use default context
        if not self.config.api_key_min_length:  # Development mode
            return self._get_development_context()

        # No authentication found
        raise ValueError("Authentication required - no valid credentials provided")

    def _extract_api_key(self, request: Any) -> str | None:
        """Extract API key from request

        Rationale: API keys provide simple authentication for MCP clients.
        Check multiple sources for flexibility in deployment.

        Args:
            request: FastMCP request object

        Returns:
            API key string or None
        """
        # Check environment variable (development)
        import os

        env_key = os.getenv("MCP_API_KEY")
        if env_key:
            return env_key

        # Check request headers (production)
        # Note: FastMCP request structure may vary
        if hasattr(request, "headers"):
            return request.headers.get("X-API-Key") or request.headers.get("Authorization")

        # Check request metadata
        if hasattr(request, "metadata"):
            return request.metadata.get("api_key")

        return None

    def _extract_session_token(self, request: Any) -> str | None:
        """Extract session token from request

        Rationale: Session tokens enable stateless authentication
        with proper session management and timeout.

        Args:
            request: FastMCP request object

        Returns:
            Session token or None
        """
        # Check request headers
        if hasattr(request, "headers"):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                return auth_header[7:]  # Remove 'Bearer ' prefix

        # Check session cookie (if supported)
        if hasattr(request, "cookies"):
            return request.cookies.get("session_token")

        return None

    async def _authenticate_with_api_key(self, api_key: str, request: Any) -> SecurityContext:
        """Authenticate using API key

        Rationale: API key authentication provides simple, stateless auth
        suitable for MCP clients and automated systems.

        Args:
            api_key: API key to validate
            request: Request for IP extraction

        Returns:
            SecurityContext for authenticated user

        Raises:
            ValueError: If API key is invalid
        """
        # In production, validate against database/API
        # For now, accept any key and map to default user
        from src.config.settings import settings

        from .authentication import UserRole

        # Simple validation - check key format
        if len(api_key) < self.config.api_key_min_length:
            raise ValueError("Invalid API key format")

        # Extract client IP for logging
        client_ip = self._extract_client_ip(request)

        # Create security context
        # In production, look up user/role from API key
        context = SecurityContext(
            user_id=f"api_key_{api_key[:8]}",  # Partial key for logging
            role=UserRole(settings.security_default_role),
            session_id=f"api_{api_key[:16]}",  # Use key as session ID
            ip_address=client_ip,
            permissions=[],  # Will be set based on role
        )

        self.logger.info(f"API key authentication successful for user {context.user_id}")
        return context

    async def _validate_session_token(self, session_token: str, request: Any) -> SecurityContext:
        """Validate session token

        Rationale: Session tokens provide stateful authentication
        with proper timeout and revocation capabilities.

        Args:
            session_token: Session token to validate
            request: Request for IP extraction

        Returns:
            SecurityContext for valid session

        Raises:
            ValueError: If session is invalid
        """
        from . import get_security_manager

        manager = get_security_manager()
        client_ip = self._extract_client_ip(request)

        context = manager.auth_manager.validate_session(session_token, client_ip)
        if not context:
            raise ValueError("Invalid or expired session")

        return context

    def _get_development_context(self) -> SecurityContext:
        """Get development mode security context

        Rationale: Development environments need simple authentication
        that doesn't require complex setup while still maintaining security.

        Returns:
            Default security context for development
        """
        from src.config.settings import settings

        from .authentication import UserRole

        return SecurityContext(
            user_id="dev_user",
            role=UserRole(settings.security_default_role),
            session_id="dev_session",
            ip_address="127.0.0.1",
            permissions=[],
        )

    def _extract_client_ip(self, request: Any) -> str:
        """Extract client IP address from request

        Rationale: IP address tracking is essential for security monitoring,
        audit logging, and detecting suspicious access patterns.

        Args:
            request: FastMCP request object

        Returns:
            Client IP address or default
        """
        # Check various sources for client IP
        if hasattr(request, "client_ip"):
            return request.client_ip

        if hasattr(request, "remote_addr"):
            return request.remote_addr

        if hasattr(request, "headers"):
            # Check forwarded headers (common in proxy setups)
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()

            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip

        # Default for development
        return "127.0.0.1"

    def _create_error_response(self, error_message: str) -> dict[str, Any]:
        """Create error response for authentication failures

        Rationale: Authentication failures should return structured errors
        rather than crashing the MCP server.

        Args:
            error_message: Error description

        Returns:
            Error response dictionary
        """
        return {
            "error": "Authentication failed",
            "message": error_message,
            "type": "authentication_error",
        }


def get_current_security_context() -> SecurityContext | None:
    """Get the current request's security context

    Rationale: Provides access to security context from anywhere in the
    request processing pipeline. Uses context variables for thread safety.

    Returns:
        Current security context or None
    """
    return security_context_var.get()


def require_security_context(func: Callable) -> Callable:
    """Decorator to ensure security context is available

    Rationale: Some functions require authenticated context.
    This decorator fails fast if no context is available.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with context validation
    """

    async def wrapper(*args, **kwargs):
        context = get_current_security_context()
        if not context:
            raise ValueError("Security context required but not available")
        return await func(*args, **kwargs)

    return wrapper
