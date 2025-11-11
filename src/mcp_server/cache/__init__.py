"""Redis caching layer for healthcare MCP server performance optimization.

This module provides enterprise-grade caching infrastructure using Redis to optimize
critical performance bottlenecks in security validation, session management, tool
prompts, data minimization, and analytics queries.

Core Components:
- rbac_cache: RBAC decision caching
- session_cache: Distributed session store
- prompt_cache: Tool prompt distributed caching
- field_schema_cache: Data minimization schema caching
- aggregation_cache: Analytics query result caching
- cache_manager: Centralized cache coordination

Design Principles:
- Graceful degradation: System works without Redis but slower
- Automatic fallback: On cache miss or Redis error, fall back to direct operations
- Security-first: No PHI cached, only metadata and decisions
- Compliance: TTL-based automatic expiration for HIPAA compliance

Example:
    Initialize and use the cache manager:
    >>> from src.mcp_server.cache import get_cache_manager
    >>> cache_manager = get_cache_manager()
    >>> if cache_manager.is_available():
    ...     # Use cached operations
    ...     pass
"""

from .cache_manager import CacheManager, get_cache_manager, initialize_cache

__all__ = [
    "CacheManager",
    "get_cache_manager",
    "initialize_cache",
]
