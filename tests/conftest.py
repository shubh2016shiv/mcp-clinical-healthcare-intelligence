"""Pytest configuration and shared fixtures for MCP Healthcare Server tests.

TESTING STRATEGY & ARCHITECTURAL DECISIONS:
============================================

Problem Statement:
------------------
Healthcare applications require rigorous testing due to:
1. Sensitive patient data - errors can have serious consequences
2. Complex business logic - multiple collections, aggregations, security rules
3. External dependencies - MongoDB, Redis, external APIs
4. Async operations - race conditions, connection pooling issues

Testing Philosophy:
-------------------
We follow the Testing Pyramid approach:

    /\\
   /  \\  E2E Tests (Few) - Full user workflows
  /____\\
 /      \\ Integration Tests (Some) - Component interactions
/________\\
Unit Tests (Many) - Individual functions/classes

1. **Unit Tests (70%)**: Fast, isolated, no external dependencies
   - Test individual functions and classes
   - Mock all external dependencies (database, cache, APIs)
   - Run in milliseconds
   - High code coverage (target: 80%+)

2. **Integration Tests (25%)**: Test component interactions
   - Real database (test MongoDB instance)
   - Real cache (test Redis instance)
   - Test data flows between layers
   - Run in seconds

3. **E2E Tests (5%)**: Test complete user workflows
   - Full server running
   - Real database with sample data
   - Test critical user journeys
   - Run in minutes

Test Organization:
-------------------
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_exceptions.py   # Exception hierarchy
│   ├── test_models.py       # Pydantic models
│   ├── test_validators.py  # Input validation
│   └── tools/
│       ├── test_patient_tools.py
│       └── test_analytics_tools.py
├── integration/             # Component interaction tests
│   ├── test_database.py     # Database operations
│   ├── test_cache.py        # Cache integration
│   └── test_security.py     # Security middleware
├── e2e/                     # End-to-end workflows
│   ├── test_patient_search_flow.py
│   └── test_clinical_timeline_flow.py
├── fixtures/                # Test data
│   ├── sample_patients.json
│   └── sample_conditions.json
└── conftest.py             # This file - shared fixtures

Fixture Design Principles:
---------------------------
1. **Scope Optimization**: Use appropriate fixture scope
   - function: New instance per test (default)
   - class: Shared across test class
   - module: Shared across test file
   - session: Shared across entire test run

2. **Dependency Injection**: Fixtures provide dependencies
   - Promotes loose coupling
   - Easy to swap implementations
   - Clear test dependencies

3. **Cleanup Automation**: Fixtures handle cleanup
   - Use yield for teardown
   - Ensures resources are released
   - Prevents test pollution

4. **Realistic Test Data**: Fixtures provide realistic data
   - Based on actual FHIR structures
   - Covers edge cases
   - Maintains data relationships

Example Usage:
--------------
```python
# Unit test with mocked database
def test_search_patients(mock_database, sample_patients):
    tools = PatientTools()
    result = await tools.search_patients(...)
    assert len(result) == 2

# Integration test with real database
@pytest.mark.integration
async def test_patient_search_integration(test_database, seed_test_data):
    tools = PatientTools()
    result = await tools.search_patients(...)
    assert result[0].patient_id == "P12345"
```

Performance Considerations:
---------------------------
- Unit tests should run in < 100ms each
- Integration tests should run in < 5s each
- Use pytest-xdist for parallel execution
- Cache expensive fixtures (session scope)
- Use pytest-benchmark for performance regression testing
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings.

    Design Rationale:
    -----------------
    Custom markers help organize and filter tests:
    - @pytest.mark.unit: Fast, isolated unit tests
    - @pytest.mark.integration: Tests requiring real dependencies
    - @pytest.mark.e2e: Full end-to-end workflow tests
    - @pytest.mark.slow: Tests that take > 1 second

    This enables running subsets of tests:
    ```bash
    pytest -m unit              # Only unit tests (fast)
    pytest -m "not slow"        # Skip slow tests
    pytest -m integration       # Only integration tests
    ```
    """
    config.addinivalue_line("markers", "unit: Fast unit tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Integration tests with real dependencies")
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Tests that take more than 1 second")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location.

    Design Rationale:
    -----------------
    Automatically applies markers based on test file location:
    - tests/unit/* → @pytest.mark.unit
    - tests/integration/* → @pytest.mark.integration
    - tests/e2e/* → @pytest.mark.e2e

    This prevents developers from forgetting to add markers.
    """
    for item in items:
        # Get test file path
        test_path = Path(item.fspath)

        # Auto-mark based on directory
        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in test_path.parts:
            item.add_marker(pytest.mark.e2e)


# =============================================================================
# EVENT LOOP FIXTURES
# =============================================================================
# Design Rationale: Async tests require an event loop. We provide a
# session-scoped event loop for better performance.


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the entire test session.

    Design Rationale:
    -----------------
    Session-scoped event loop improves test performance by:
    1. Reusing the same loop across all tests
    2. Avoiding loop creation/teardown overhead
    3. Enabling session-scoped async fixtures

    Trade-off:
    ----------
    - Pro: Faster test execution
    - Con: Tests must be careful not to pollute the loop

    Yields:
    -------
    asyncio.AbstractEventLoop
        Event loop for async tests
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# MOCK DATABASE FIXTURES
# =============================================================================
# Design Rationale: Unit tests should NOT touch real databases. These
# fixtures provide fully mocked database objects for fast, isolated testing.


@pytest.fixture
def mock_motor_client() -> MagicMock:
    """Create a mocked Motor client for unit tests.

    Design Rationale:
    -----------------
    Provides a fully mocked AsyncIOMotorClient that:
    1. Returns mocked databases and collections
    2. Supports async operations (find, insert, update, etc.)
    3. Allows verification of database calls
    4. Runs instantly (no network I/O)

    Returns:
    --------
    MagicMock configured to behave like AsyncIOMotorClient

    Example:
    --------
    >>> def test_database_query(mock_motor_client):
    ...     db = mock_motor_client["test_db"]
    ...     collection = db["patients"]
    ...     collection.find.return_value.to_list = AsyncMock(return_value=[...])
    ...     # Test code that uses the collection
    ...     collection.find.assert_called_once()
    """
    client = MagicMock(spec=AsyncIOMotorClient)
    return client


@pytest.fixture
def mock_database() -> MagicMock:
    """Create a mocked Motor database for unit tests.

    Design Rationale:
    -----------------
    Provides a pre-configured mocked database with common collections.
    This is the most commonly used fixture for unit tests.

    Collections Provided:
    ---------------------
    - patients: Patient demographics
    - encounters: Healthcare visits
    - conditions: Medical diagnoses
    - medications: Prescriptions
    - observations: Lab results
    - claims: Insurance claims
    - drugs: RxNorm drug database

    Returns:
    --------
    MagicMock configured as AsyncIOMotorDatabase with mocked collections

    Example:
    --------
    >>> async def test_patient_search(mock_database):
    ...     # Configure mock to return test data
    ...     mock_database["patients"].find.return_value.to_list = AsyncMock(
    ...         return_value=[{"patient_id": "P123", "first_name": "John"}]
    ...     )
    ...
    ...     # Test code
    ...     tools = PatientTools()
    ...     result = await tools.search_patients(...)
    ...
    ...     # Verify database was called correctly
    ...     mock_database["patients"].find.assert_called_once()
    """
    db = MagicMock(spec=AsyncIOMotorDatabase)

    # Create mocked collections
    collections = [
        "patients",
        "encounters",
        "conditions",
        "medications",
        "observations",
        "claims",
        "drugs",
        "care_plans",
        "diagnostic_reports",
        "immunizations",
        "procedures",
    ]

    for collection_name in collections:
        collection = AsyncMock()
        # Configure common async methods
        collection.find.return_value.to_list = AsyncMock(return_value=[])
        collection.find_one = AsyncMock(return_value=None)
        collection.insert_one = AsyncMock()
        collection.insert_many = AsyncMock()
        collection.update_one = AsyncMock()
        collection.delete_one = AsyncMock()
        collection.count_documents = AsyncMock(return_value=0)
        collection.aggregate.return_value.to_list = AsyncMock(return_value=[])

        # Add collection to database
        db.__getitem__.return_value = collection
        setattr(db, collection_name, collection)

    return db


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================
# Design Rationale: Realistic test data is essential for meaningful tests.
# These fixtures provide FHIR-compliant healthcare data.


@pytest.fixture
def sample_patient_document() -> dict:
    """Provide a sample patient document for testing.

    Design Rationale:
    -----------------
    Provides a realistic FHIR-compliant patient document with:
    - All required fields
    - Realistic data values
    - Edge cases (middle name, multiple addresses)

    Returns:
    --------
    dict representing a MongoDB patient document

    Example:
    --------
    >>> def test_patient_parsing(sample_patient_document):
    ...     patient = PatientSummary(**sample_patient_document)
    ...     assert patient.first_name == "John"
    """
    return {
        "_id": "507f1f77bcf86cd799439011",
        "patient_id": "P12345",
        "first_name": "John",
        "middle_name": "Robert",
        "last_name": "Doe",
        "birth_date": "1980-01-15",
        "gender": "M",
        "race": "White",
        "ethnicity": "Not Hispanic or Latino",
        "address": {
            "line": ["123 Main St", "Apt 4B"],
            "city": "San Francisco",
            "state": "CA",
            "postal_code": "94102",
            "country": "US",
        },
        "telecom": [
            {"system": "phone", "value": "555-1234"},
            {"system": "email", "value": "john.doe@example.com"},
        ],
        "marital_status": "Married",
        "language": "en",
        "active": True,
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_patients_list() -> list[dict]:
    """Provide a list of sample patients for testing search results.

    Design Rationale:
    -----------------
    Provides multiple patients with diverse characteristics:
    - Different demographics (age, gender, race)
    - Different locations (states, cities)
    - Edge cases (missing fields, special characters)

    Returns:
    --------
    list of patient documents

    Example:
    --------
    >>> def test_patient_filtering(sample_patients_list):
    ...     ca_patients = [p for p in sample_patients_list if p["state"] == "CA"]
    ...     assert len(ca_patients) == 2
    """
    return [
        {
            "_id": "507f1f77bcf86cd799439011",
            "patient_id": "P12345",
            "first_name": "John",
            "last_name": "Doe",
            "birth_date": "1980-01-15",
            "gender": "M",
            "state": "CA",
            "city": "San Francisco",
        },
        {
            "_id": "507f1f77bcf86cd799439012",
            "patient_id": "P12346",
            "first_name": "Jane",
            "last_name": "Smith",
            "birth_date": "1975-05-20",
            "gender": "F",
            "state": "NY",
            "city": "New York",
        },
        {
            "_id": "507f1f77bcf86cd799439013",
            "patient_id": "P12347",
            "first_name": "María",  # Test unicode
            "last_name": "García",
            "birth_date": "1990-12-01",
            "gender": "F",
            "state": "CA",
            "city": "Los Angeles",
        },
    ]


@pytest.fixture
def sample_condition_document() -> dict:
    """Provide a sample medical condition document.

    Returns:
    --------
    dict representing a MongoDB condition document
    """
    return {
        "_id": "607f1f77bcf86cd799439021",
        "condition_id": "C12345",
        "patient_id": "P12345",
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "44054006",
                    "display": "Type 2 Diabetes Mellitus",
                }
            ],
            "text": "Type 2 Diabetes Mellitus",
        },
        "clinical_status": "active",
        "verification_status": "confirmed",
        "onset_date": "2015-03-10",
        "recorded_date": "2015-03-15",
        "severity": "moderate",
    }


@pytest.fixture
def sample_medication_document() -> dict:
    """Provide a sample medication document.

    Returns:
    --------
    dict representing a MongoDB medication document
    """
    return {
        "_id": "707f1f77bcf86cd799439031",
        "medication_id": "M12345",
        "patient_id": "P12345",
        "medication_name": "Metformin 500mg",
        "rxcui": "860975",
        "status": "active",
        "prescribed_date": "2015-03-15",
        "dosage": "500mg twice daily",
        "prescriber": "Dr. Smith",
    }


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================
# Design Rationale: Integration tests need real databases. These fixtures
# provide test database instances with proper cleanup.


@pytest.fixture(scope="session")
async def test_mongodb_client() -> AsyncGenerator[AsyncIOMotorClient, None]:
    """Provide a real MongoDB client for integration tests.

    Design Rationale:
    -----------------
    Session-scoped fixture that:
    1. Connects to test MongoDB instance
    2. Reused across all integration tests (performance)
    3. Automatically closes connection on teardown

    Environment Variables:
    ----------------------
    TEST_MONGODB_URI: MongoDB connection string (default: mongodb://localhost:27017)

    Yields:
    -------
    AsyncIOMotorClient connected to test database

    Example:
    --------
    >>> @pytest.mark.integration
    ... async def test_with_real_db(test_mongodb_client):
    ...     db = test_mongodb_client["test_healthcare_db"]
    ...     await db.patients.insert_one({...})
    """
    import os

    mongodb_uri = os.getenv("TEST_MONGODB_URI", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(mongodb_uri)

    # Verify connection
    try:
        await client.admin.command("ping")
    except Exception as e:
        pytest.skip(f"MongoDB not available for integration tests: {e}")

    yield client

    # Cleanup: close connection
    client.close()


@pytest.fixture
async def test_database(
    test_mongodb_client: AsyncIOMotorClient,
) -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """Provide a clean test database for each integration test.

    Design Rationale:
    -----------------
    Function-scoped fixture that:
    1. Creates a fresh database for each test
    2. Ensures test isolation (no data pollution)
    3. Automatically cleans up after test

    Database Naming:
    ----------------
    Uses unique database name per test to enable parallel testing:
    test_healthcare_db_{test_name}_{timestamp}

    Yields:
    -------
    AsyncIOMotorDatabase - Clean test database

    Example:
    --------
    >>> @pytest.mark.integration
    ... async def test_patient_crud(test_database):
    ...     await test_database.patients.insert_one({...})
    ...     result = await test_database.patients.find_one({...})
    ...     assert result is not None
    """
    import time

    # Create unique database name for this test
    db_name = f"test_healthcare_db_{int(time.time() * 1000)}"
    db = test_mongodb_client[db_name]

    yield db

    # Cleanup: drop test database
    await test_mongodb_client.drop_database(db_name)


@pytest.fixture
async def seed_test_data(test_database: AsyncIOMotorDatabase) -> dict:
    """Seed test database with realistic healthcare data.

    Design Rationale:
    -----------------
    Provides a complete dataset for integration testing:
    - Multiple patients with relationships
    - Conditions, medications, encounters
    - Realistic data distribution

    Returns:
    --------
    dict with IDs of created documents for test assertions

    Example:
    --------
    >>> @pytest.mark.integration
    ... async def test_patient_timeline(test_database, seed_test_data):
    ...     patient_id = seed_test_data["patients"][0]
    ...     # Test timeline query
    ...     result = await get_patient_timeline(patient_id)
    ...     assert len(result.events) > 0
    """
    # Load test data from JSON files
    fixtures_dir = Path(__file__).parent / "fixtures"

    # Insert patients
    patients_file = fixtures_dir / "sample_patients.json"
    if patients_file.exists():
        with open(patients_file) as f:
            patients = json.load(f)
            result = await test_database.patients.insert_many(patients)
            patient_ids = [str(id) for id in result.inserted_ids]
    else:
        # Fallback: create minimal test data
        patients = [
            {"patient_id": "P001", "first_name": "Test", "last_name": "Patient1"},
            {"patient_id": "P002", "first_name": "Test", "last_name": "Patient2"},
        ]
        result = await test_database.patients.insert_many(patients)
        patient_ids = [str(id) for id in result.inserted_ids]

    return {"patients": patient_ids, "database": test_database}


# =============================================================================
# SECURITY CONTEXT FIXTURES
# =============================================================================


@pytest.fixture
def mock_security_context():
    """Provide a mocked security context for testing.

    Design Rationale:
    -----------------
    Security context is required by most tools. This fixture provides
    a pre-configured context for testing without authentication overhead.

    Returns:
    --------
    Mock SecurityContext with standard permissions

    Example:
    --------
    >>> def test_authorized_access(mock_security_context):
    ...     result = await patient_tools.search_patients(..., mock_security_context)
    ...     assert result is not None
    """
    from unittest.mock import MagicMock

    context = MagicMock()
    context.user_id = "test_user_123"
    context.permissions = ["read:patient", "read:condition", "read:medication"]
    context.roles = ["clinician"]
    context.ip_address = "127.0.0.1"

    return context
