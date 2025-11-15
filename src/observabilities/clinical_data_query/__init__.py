"""Clinical data query observability module.

This module provides specialized observability for MongoDB queries executed
on clinical data collections. Designed for HIPAA compliance and healthcare
query performance monitoring.

Key Features:
- MongoDB query tracing without PHI storage
- Automatic linkage to agent execution traces
- Security compliance monitoring
- Query performance analysis
- HIPAA-compliant audit trails

Main Components:
- MongoDBQueryTrace: Data model for query metadata
- MongoDBQueryTracer: Core tracer for query execution
- MongoDBQueryContext: Context manager for automatic tracing
- ClinicalQueryManager: High-level query management utilities
"""

from .query_context import ClinicalQueryManager, MongoDBQueryContext
from .query_models import MongoDBQueryTrace, QueryType
from .query_tracer import MongoDBQueryTracer, get_query_tracer

__all__ = [
    # Data models
    "MongoDBQueryTrace",
    "QueryType",
    # Core tracer
    "MongoDBQueryTracer",
    "get_query_tracer",
    # Context manager
    "MongoDBQueryContext",
    # High-level utilities
    "ClinicalQueryManager",
]
