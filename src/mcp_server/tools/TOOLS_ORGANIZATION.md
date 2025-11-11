# MCP Tools Organization Guide

## Overview

The MCP tools have been reorganized into a modular, domain-driven structure that separates concerns and enables logical scaling. This document explains the new organization and how to use the tools.

## Directory Structure

```
tools/
├── _core/                    # Tier 1: Core MongoDB operations
│   ├── database_introspection.py    # List collections, analyze schemas
│   ├── query_validation.py          # Validate queries for safety
│   ├── query_execution.py           # Execute queries with logging
│   └── result_serialization.py      # Convert BSON to JSON
│
├── _mongodb/                 # MongoDB-specific optimization layer
│   ├── aggregation_builder.py       # Build aggregation pipelines
│   ├── query_patterns.py            # Reusable query templates
│   └── ...
│
├── _healthcare/              # Tier 2: Healthcare domain logic
│   ├── demographics/
│   │   └── patient_search.py        # search_patients tool
│   ├── clinical_timeline/
│   │   └── patient_timeline.py      # get_patient_clinical_timeline tool
│   ├── clinical_data/
│   │   └── conditions.py            # analyze_conditions tool
│   ├── financial/
│   │   └── financial_analytics.py   # get_financial_summary tool
│   ├── medications/
│   │   ├── patient_medications.py   # get_medication_history tool
│   │   ├── drug_reference.py        # search_drugs tool
│   │   └── drug_analysis.py         # analyze_drug_classes tool
│   └── encounters/
│       └── encounters.py            # Future: encounter queries
│
├── base_tool.py             # Shared base class with connection management
├── models.py                # Pydantic models (will be split into domains in Phase 3)
├── utils.py                 # Shared utility functions
└── server.py               # FastMCP server registration
```

## Tier Organization

### Tier 1: Core Foundation (`_core/`)

Fundamental MongoDB operations that all other tools build upon:

- **database_introspection.py**: Discover available collections and their schemas
- **query_validation.py**: Validate queries for syntax and safety
- **query_execution.py**: Execute validated queries with observability logging
- **result_serialization.py**: Convert MongoDB BSON types to JSON

#### Key Features:
- Query Observability: All queries logged before execution with collection, filter, and limit
- Safety Validation: Read-only enforcement, depth checking, destructive operation prevention
- Performance Tracking: Execution time, result counts, memory efficiency

### Tier 2: MongoDB Optimization (`_mongodb/`)

MongoDB-specific helpers and optimization tools (currently in development):

- **aggregation_builder.py**: Construct and optimize aggregation pipelines
- **query_patterns.py**: Reusable query templates for common scenarios
- **index_manager.py**: Index creation and analysis
- **bulk_operations.py**: Batch read operations

### Tier 3: Healthcare Domain (`_healthcare/`)

Business logic organized by operational category, not by collection:

#### Demographics (Patient Identification)
- **demographics/patient_search.py**: Search patients by demographics, identifiers, location

#### Clinical Timeline (Patient History)
- **clinical_timeline/patient_timeline.py**: Comprehensive chronological clinical history

#### Clinical Data (Clinical Findings)
- **clinical_data/conditions.py**: Analyze health conditions (population or patient-specific)
- **clinical_data/observations.py**: Lab results, measurements
- **clinical_data/procedures.py**: Surgical/medical procedures
- **clinical_data/immunizations.py**: Vaccination records
- **clinical_data/allergies.py**: Allergies/intolerances
- **clinical_data/care_plans.py**: Care plans

#### Medications (Drug-Related Data)
- **medications/patient_medications.py**: Patient medication history (REQUIRES patient_id)
- **medications/drug_reference.py**: RxNorm drug database search
- **medications/drug_analysis.py**: Population-level drug classification analysis

#### Financial (Billing & Insurance)
- **financial/financial_analytics.py**: Claims and cost analysis

#### Encounters (Future)
- **encounters/encounters.py**: Visit and encounter data

## Patient ID Dependency Strategy

Tools are classified by their patient ID requirements:

### Patient-Specific Tools (REQUIRED patient_id)
- `get_patient_clinical_timeline`: Cannot function without patient_id
- `get_medication_history`: Patient-centric medication retrieval

### Optionally Patient-Specific (OPTIONAL patient_id)
- `analyze_conditions`: Population-level by default, but can filter to specific patient
- `get_financial_summary`: Population-level by default, but can filter to specific patient

### Population-Level Tools (No patient_id)
- `search_patients`: Searches across all patients
- `search_drugs`: Searches drug reference database
- `analyze_drug_classes`: Analyzes drug classifications

## Query Observability

All queries executed through the core layer include observability logging:

```
========================================================================
EXECUTING QUERY FOR VERIFICATION:
  Collection: conditions
  Query Type: find
  Query Filter: {"code.coding.display": /Diabetes/i}
  Projection: None
  Limit: 50
========================================================================
```

This ensures:
1. Queries can be verified in console/logs
2. Intermediate actions by agent tools are transparent
3. Debugging is simplified
4. Audit trail is maintained

## Usage Examples

### Using Original Tool Classes (Backward Compatible)
```python
from src.mcp_server.tools import PatientTools, AnalyticsTools

patient_tools = PatientTools()
analytics_tools = AnalyticsTools()
```

### Using New Domain-Specific Classes
```python
from src.mcp_server.tools._healthcare.demographics import PatientSearchTools
from src.mcp_server.tools._healthcare.clinical_timeline import PatientTimelineTools

search_tools = PatientSearchTools()
timeline_tools = PatientTimelineTools()
```

### Using Core Functions Directly
```python
from src.mcp_server.tools._core import (
    execute_mongodb_query,
    get_database_collections,
    validate_mongodb_query
)

# Execute a raw query with full logging
result = execute_mongodb_query(
    collection_name="patients",
    query='{"gender": "M"}',
    limit=10
)
```

## Migration Roadmap

### Phase 1 ✓ Complete
- Created `_core/` modules for MongoDB operations
- Implemented query observability logging
- Separated concerns: validation, execution, serialization

### Phase 2 ✓ Complete
- Moved patient_tools logic to `_healthcare/demographics/` and `_healthcare/clinical_timeline/`
- Moved analytics tools to `_healthcare/clinical_data/` and `_healthcare/financial/`
- Moved medication/drug tools to `_healthcare/medications/`
- Tools still use original classes for backward compatibility

### Phase 3 ⏳ In Progress
- Reorganize models.py into domain-specific files
- Move domain-specific utilities into each module
- Complete dependency refactoring

### Phase 4 (Future)
- Implement `_mongodb/` optimization layer
- Add aggregation pipeline builders
- Create query pattern library
- Implement caching strategies

## Best Practices for Adding New Tools

1. **Identify the operational category**: What business question does it answer?
   - Demographics? → `_healthcare/demographics/`
   - Clinical findings? → `_healthcare/clinical_data/`
   - Financial data? → `_healthcare/financial/`

2. **Create or extend the module**: Add your tool to the appropriate category

3. **Declare patient ID requirements**: Document if patient_id is required, optional, or not used

4. **Use query observability**: Log queries before execution

5. **Follow validation patterns**: 
   - Validate patient_id when required
   - Check for required fields
   - Log filtered operations

## Key Principles

1. **Logical Organization**: Tools grouped by business domain, not database collections
2. **Clear Separation**: Database operations ↑ Domain logic ↑ Healthcare business logic
3. **Patient-Centric Design**: Patient ID requirements are explicit and validated
4. **Observability First**: All queries logged before execution
5. **Backward Compatibility**: Original tool classes still work
6. **Simplicity**: Implementation is straightforward, easy to extend for edge cases

