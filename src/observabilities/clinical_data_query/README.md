# Clinical Data Query Observability

This module provides specialized observability for MongoDB queries executed on clinical data collections. Designed for HIPAA compliance, security auditing, and healthcare query performance monitoring.

## Overview

The clinical data query observability module captures MongoDB query execution metadata without storing actual patient data (PHI - Protected Health Information). It provides:

- **Query Tracing**: Capture MongoDB query structure and execution metrics
- **HIPAA Compliance**: No PHI storage, only query metadata and patterns
- **Security Auditing**: Track query validation and security compliance
- **Performance Monitoring**: Analyze query execution times and patterns
- **Agent Integration**: Link clinical queries to agent execution traces

## Architecture

```
clinical_data_query/
├── __init__.py              # Public API exports
├── README.md                # This documentation
├── query_models.py          # MongoDBQueryTrace, QueryType (data models)
├── query_tracer.py          # MongoDBQueryTracer (core tracing logic)
└── query_context.py         # MongoDBQueryContext, ClinicalQueryManager
```

## Key Components

### MongoDBQueryTrace
Data model for capturing query metadata without PHI:
- Query structure (filter, projection, limits)
- Execution metrics (timing, success/failure)
- Security compliance flags
- Linkage to agent execution traces

### MongoDBQueryTracer
Core tracer for managing query execution:
- Start/finalize query traces
- Link queries to agent traces
- Performance statistics collection
- Memory management (LRU eviction)

### MongoDBQueryContext
Context manager for automatic query tracing:
- Minimal code changes required
- Automatic trace ID discovery
- Exception handling and cleanup
- HIPAA compliance enforcement

## Usage Patterns

### Basic Context Manager Usage

```python
from src.observabilities.clinical_data_query import MongoDBQueryContext, QueryType

# Wrap MongoDB query execution
with MongoDBQueryContext(
    collection_name="conditions",
    query_filter={"code.coding.display": {"$regex": "diabetes"}},
    query_type=QueryType.FIND,
    limit=20
) as ctx:
    # Execute query
    results = await collection.find(query).to_list(20)

    # Set result count (important for metrics)
    ctx.set_result(len(results))
# Trace automatically finalized
```

### Manual Tracer Usage

```python
from src.observabilities.clinical_data_query import get_query_tracer, QueryType

tracer = get_query_tracer()

# Start trace
query_trace = tracer.start_query_trace(
    collection_name="patients",
    query_filter={"age": {"$gt": 65}},
    query_type=QueryType.FIND,
    limit=100
)

try:
    # Execute query
    results = await collection.find(query_filter).to_list(100)

    # Finalize with success
    tracer.finalize_query_trace(
        query_id=query_trace.query_id,
        success=True,
        result_count=len(results),
        execution_time_ms=45.2
    )
except Exception as e:
    # Finalize with error
    tracer.finalize_query_trace(
        query_id=query_trace.query_id,
        success=False,
        error=str(e),
        execution_time_ms=12.5
    )
```

### Clinical Query Manager

```python
from src.observabilities.clinical_data_query import ClinicalQueryManager

manager = ClinicalQueryManager()

# High-level query tracing
with manager.trace_query("medications", medication_query) as ctx:
    results = await execute_medication_query()
    ctx.set_result(len(results))

# Get performance reports
performance = manager.get_query_performance()
security = manager.get_security_summary()
```

## Integration with Agent Execution

### Automatic Trace Linking

Clinical queries are automatically linked to agent execution traces:

```python
# Agent execution trace
with TraceContext("session_123", "Find diabetic patients") as agent_trace:
    # Tool calls within this context automatically link MongoDB queries
    with MongoDBQueryContext("conditions", diabetes_query) as query_ctx:
        # This MongoDB query is linked to agent_trace.trace_id
        results = await execute_query()
```

### Manual Linking

```python
# Link specific MongoDB queries to agent traces
query_trace = tracer.start_query_trace(
    collection_name="patients",
    query_filter=patient_query,
    trace_id="specific-agent-trace-id"  # Manual linking
)
```

## HIPAA Compliance

### No PHI Storage
- Only query structure metadata is stored
- No patient names, IDs, or clinical data
- Query filters are sanitized for logging

### Security Tracking
- Query validation status tracking
- Security check results logging
- PHI collection identification
- Audit trail generation

### Compliance Reports

```python
from src.observabilities.clinical_data_query import ClinicalQueryManager

manager = ClinicalQueryManager()
security_report = manager.get_security_summary()

# Example output:
{
    "total_queries": 150,
    "phi_queries": 120,      # Queries on PHI collections
    "validated_queries": 148, # Queries that passed validation
    "security_passed": 145,   # Queries that passed security checks
    "validation_rate": 98.7,
    "security_pass_rate": 96.7
}
```

## Performance Monitoring

### Query Performance Analysis

```python
from src.observabilities.clinical_data_query import get_query_tracer

tracer = get_query_tracer()

# Global statistics
stats = tracer.get_query_statistics()
# {
#   "total_queries": 1000,
#   "avg_execution_time_ms": 45.2,
#   "query_type_distribution": {"find": 800, "aggregate": 150, "count": 50}
# }

# Performance report
report = tracer.get_performance_report()
# {
#   "total_queries": 1000,
#   "successful_queries": 980,
#   "avg_execution_time_ms": 45.2,
#   "p95_execution_time_ms": 120.5,
#   "query_types": {
#     "find": {"count": 800, "avg_time_ms": 35.1},
#     "aggregate": {"count": 150, "avg_time_ms": 85.3}
#   }
# }
```

### Query Complexity Scoring

Each query gets a complexity score (0-10) for optimization:

```python
trace = MongoDBQueryTrace(...)
score = trace.get_query_complexity_score()
# Higher scores indicate complex queries needing optimization
```

## Integration Point

### Modified query_execution.py

The `execute_mongodb_query` function is wrapped with tracing:

```python
# In src/mcp_server/tools/_core/query_execution.py

async def execute_mongodb_query(...):
    # ... validation code ...

    # Wrap with clinical query tracing
    with MongoDBQueryContext(
        collection_name=collection_name,
        query_filter=parsed_query,
        query_type=QueryType(query_type),
        projection=projection_dict,
        limit=result_limit,
    ) as query_ctx:
        # Existing execution code unchanged
        start_time = time.time()

        try:
            if query_type == "find":
                results = await collection.find(...).to_list(limit)
                count = len(results)
                query_ctx.set_result(count)  # Record result count
            # ... other query types ...
        except Exception as e:
            # Error automatically captured by context
            raise
```

## Security Features

### Query Validation Tracking
- Pre-execution validation status
- Security check results
- Query depth validation
- Dangerous operation blocking

### Audit Trail Generation
- Query execution logs
- Security compliance reports
- Performance anomaly detection
- HIPAA compliance monitoring

## Best Practices

### 1. Always Use Context Managers
```python
# ✅ Good: Automatic cleanup
with MongoDBQueryContext("patients", query) as ctx:
    results = await execute_query()
    ctx.set_result(len(results))

# ❌ Bad: Manual cleanup required
trace = tracer.start_query_trace(...)
try:
    results = await execute_query()
finally:
    tracer.finalize_query_trace(...)
```

### 2. Set Result Counts
```python
# Always set result count for accurate metrics
with MongoDBQueryContext("conditions", query) as ctx:
    results = await collection.find(query).to_list(50)
    ctx.set_result(len(results))  # Important!
```

### 3. Use Security Validation
```python
with MongoDBQueryContext("patients", query, security_validated=True) as ctx:
    # Query passed security validation
    results = await execute_secure_query()
    ctx.set_result(len(results))
```

### 4. Monitor Performance
```python
# Regular performance monitoring
tracer = get_query_tracer()
report = tracer.get_performance_report()

if report["avg_execution_time_ms"] > 100:
    logger.warning("Slow clinical queries detected")
```

## Performance Characteristics

- **Overhead**: ~0.2ms per query trace
- **Memory**: ~1KB per trace
- **Storage**: Separate from agent traces
- **HIPAA**: Zero PHI storage

## Error Handling

The context manager automatically handles errors:

```python
# Successful query
with MongoDBQueryContext("patients", query) as ctx:
    results = await collection.find(query).to_list(10)
    ctx.set_result(len(results))
# Trace marked as successful

# Failed query
with MongoDBQueryContext("patients", query) as ctx:
    raise ValueError("Invalid query")
# Trace automatically marked as failed with error details
```

## Testing

```python
# Unit test example
def test_query_tracing():
    tracer = get_query_tracer()

    with MongoDBQueryContext("test_collection", {"test": "query"}) as ctx:
        # Simulate query execution
        ctx.set_result(5)

    # Verify trace was created
    trace = tracer.get_query_trace(ctx.query_id)
    assert trace is not None
    assert trace.success == True
    assert trace.result_count == 5
```

## Troubleshooting

### Trace Not Found
```python
# Check if trace_id discovery is working
from src.observabilities.integrations.context_managers import ExecutionContext

trace_id = ExecutionContext.get_current_trace_id()
if trace_id is None:
    logger.warning("No active agent trace - MongoDB queries won't be linked")
```

### Missing Result Counts
```python
# Always call set_result() in context
with MongoDBQueryContext("collection", query) as ctx:
    results = await execute_query()
    ctx.set_result(len(results))  # Required for metrics
```

### Performance Issues
```python
# Check query complexity
trace = MongoDBQueryTrace(...)
if trace.get_query_complexity_score() > 7:
    logger.warning(f"Complex query detected: {trace.collection_name}")
```

## Future Enhancements

- Query optimization recommendations
- Automated performance alerts
- Enhanced security compliance reporting
- Query pattern analysis for audit trails
- Integration with healthcare analytics platforms