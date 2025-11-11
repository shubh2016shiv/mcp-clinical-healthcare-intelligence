"""Security-based field projection manager for data minimization at query level.

This module provides utilities for applying RBAC-based field projections directly
at the MongoDB query level, minimizing data transfer and reducing processing overhead.

Key Features:
    - Role-based field filtering at query level
    - Projection cache for performance
    - Support for nested field filtering
    - Integration with security context
    - Comprehensive logging for audit trails

Example:
    >>> from src.mcp_server.security.projection_manager import get_projection_manager
    >>> manager = get_projection_manager()
    >>> projection = manager.get_query_projection('viewer', 'patient')
    >>> results = collection.find(query, projection)
"""

import logging
from typing import Optional

from .authentication import UserRole

logger = logging.getLogger(__name__)


class ProjectionManager:
    """Manages RBAC-based field projections for query optimization.

    This class provides efficient projection generation based on user roles
    to minimize data transfer at the MongoDB query level, reducing both
    bandwidth and processing overhead.

    Implements singleton pattern - use get_projection_manager() to retrieve instance.
    """

    _instance: Optional["ProjectionManager"] = None
    _lock = None

    def __new__(cls) -> "ProjectionManager":
        """Implement singleton pattern.

        Returns:
            The single ProjectionManager instance for the application
        """
        if cls._instance is None:
            if cls._lock is None:
                import threading

                cls._lock = threading.Lock()

            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the projection manager.

        Sets up caching and state management.
        """
        if getattr(self, "_initialized", False):
            return

        self._projection_cache: dict[tuple[str, str], dict[str, int] | None] = {}
        self._initialized = True

        logger.debug("ProjectionManager initialized with caching support")

    def get_query_projection(self, role: str, collection_type: str) -> dict[str, int] | None:
        """Get MongoDB query projection for a specific role and collection.

        Returns a MongoDB projection dictionary based on the user's role and
        the collection type being queried. Results are cached for performance.

        Args:
            role: User role (e.g., 'viewer', 'analyst', 'admin')
            collection_type: Type of collection (e.g., 'patient', 'medication')

        Returns:
            MongoDB projection dict (1=include, 0=exclude), or None if no projection needed

        Example:
            >>> manager = get_projection_manager()
            >>> projection = manager.get_query_projection('viewer', 'patient')
            >>> # projection = {'patient_id': 1, 'name': 1, '_id': 1}
        """
        # Check cache first
        cache_key = (role, collection_type)
        if cache_key in self._projection_cache:
            logger.debug(f"Using cached projection for role={role}, collection={collection_type}")
            return self._projection_cache[cache_key]

        # Get allowed fields from RBAC config
        allowed_fields = self._get_allowed_fields(role, collection_type)

        # Generate projection
        projection = self._build_projection(allowed_fields)

        # Cache result
        self._projection_cache[cache_key] = projection

        logger.debug(
            f"Generated projection for role={role}, collection={collection_type}: "
            f"{len(allowed_fields) if allowed_fields else 0} fields allowed"
        )

        return projection

    def _get_allowed_fields(self, role: str, collection_type: str) -> list[str] | None:
        """Get allowed fields for a role from DataMinimizer.

        Args:
            role: User role (string value)
            collection_type: Collection type (for future use, currently not used)

        Returns:
            List of allowed field names, or None if all fields allowed
        """
        try:
            # Import here to avoid circular dependencies
            # Import from current package (security module)
            from . import get_security_manager

            # Get security manager and data minimizer
            security_manager = get_security_manager()
            data_minimizer = security_manager.data_minimizer

            # Convert role string to UserRole enum
            try:
                user_role = UserRole(role) if isinstance(role, str) else role
            except (ValueError, TypeError):
                logger.warning(f"Invalid role: {role}")
                return None

            # Get allowed fields from data minimizer
            allowed_fields_set = data_minimizer.get_allowed_fields(user_role)

            if allowed_fields_set is None:
                # None means all fields allowed (e.g., ADMIN role)
                return None

            # Convert set to list for projection building
            return list(allowed_fields_set)

        except Exception as e:
            logger.warning(f"Failed to get allowed fields for role={role}: {e}")
            return None

    def _build_projection(self, allowed_fields: list[str] | None) -> dict[str, int] | None:
        """Build MongoDB projection from allowed fields list.

        Args:
            allowed_fields: List of allowed field names, or None

        Returns:
            MongoDB projection dict, or None if no fields specified
        """
        if not allowed_fields:
            return None

        projection = {}

        # Include specified fields
        for field in allowed_fields:
            if field != "_id":  # Handle _id separately
                projection[field] = 1

        # Always include _id unless explicitly excluded
        if "_id" not in [f for f in allowed_fields if f.startswith("_id")]:
            projection["_id"] = 1

        return projection if projection else None

    def clear_cache(self) -> None:
        """Clear the projection cache.

        Useful after RBAC configuration changes.

        Example:
            >>> manager = get_projection_manager()
            >>> manager.clear_cache()
        """
        self._projection_cache.clear()
        logger.debug("Projection cache cleared")

    def get_cache_stats(self) -> dict[str, int]:
        """Get projection cache statistics.

        Returns:
            Dictionary with cache statistics

        Example:
            >>> manager = get_projection_manager()
            >>> stats = manager.get_cache_stats()
            >>> print(f"Cached projections: {stats['size']}")
        """
        return {"size": len(self._projection_cache), "entries": list(self._projection_cache.keys())}


# Global singleton instance
_projection_manager: ProjectionManager | None = None


def get_projection_manager() -> ProjectionManager:
    """Get the global singleton ProjectionManager instance.

    Creates the projection manager on first call, subsequent calls return
    the same instance.

    Returns:
        The global ProjectionManager singleton instance

    Example:
        >>> from src.mcp_server.security.projection_manager import get_projection_manager
        >>> manager = get_projection_manager()
        >>> projection = manager.get_query_projection('viewer', 'patient')
    """
    global _projection_manager
    if _projection_manager is None:
        _projection_manager = ProjectionManager()
    return _projection_manager
