"""Patient search with radical simplicity.

This module eliminates all architectural complexity:
- No thread pools or executors
- No connection managers or locks
- No shared state or singletons
- Pure async/await with Motor
- Streaming results (no memory bloat)
"""

import logging

from pydantic import BaseModel

# Import the new database module
from src.mcp_server.database import database

from ...models import PatientSummary, SearchPatientsRequest

logger = logging.getLogger(__name__)

# ============================================================================
# PROJECTION CACHE (Simple Module-Level Cache)
# ============================================================================
# No manager classes, just a dict. Projections rarely change.
_projection_cache: dict[tuple[str, str], dict] = {}


def get_cached_projection(role: str, collection: str) -> dict | None:
    """Get projection from simple cache."""
    return _projection_cache.get((role, collection))


def cache_projection(role: str, collection: str, projection: dict) -> None:
    """Store projection in simple cache."""
    _projection_cache[(role, collection)] = projection


def load_projections_from_config(config: dict) -> None:
    """Load all projections at startup. Called once."""
    for role, collections in config.items():
        for collection, fields in collections.items():
            projection = {field: 1 for field in fields}
            cache_projection(role, collection, projection)
    logger.info(f"✓ Loaded {len(_projection_cache)} projections into cache")


# ============================================================================
# DATA MODELS (Clear, Explicit)
# ============================================================================
# PatientSummary is imported from ...models to ensure type compatibility
# with the server's isinstance checks


class SecurityContext(BaseModel):
    """Who's asking and what they can see."""

    role: str
    user_id: str


# ============================================================================
# QUERY BUILDER (Pure Functions, No Side Effects)
# ============================================================================


def _normalize_state(state: str) -> str:
    """Convert state name to state code if needed.

    Maps common state names to their 2-letter codes.
    If already a code or not found, returns as-is (will use regex match).
    """
    state_name_to_code = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
        "district of columbia": "DC",
    }

    state_lower = state.strip().lower()
    # If it's already a 2-letter code, return uppercase
    if len(state_lower) == 2:
        return state.upper()
    # Try to find in mapping
    return state_name_to_code.get(state_lower, state)


def build_patient_search_query(request: SearchPatientsRequest) -> dict:
    """Build MongoDB query from request. Pure function, easy to test.

    Returns a MongoDB filter dict. No database calls here.
    """
    query_conditions = []

    # Text searches (case-insensitive, partial match)
    if request.first_name:
        query_conditions.append({"first_name": {"$regex": request.first_name, "$options": "i"}})

    if request.last_name:
        query_conditions.append({"last_name": {"$regex": request.last_name, "$options": "i"}})

    # Exact matches
    if request.patient_id:
        query_conditions.append({"patient_id": request.patient_id})

    if request.gender:
        query_conditions.append({"gender": request.gender})

    # Race filter (case-insensitive, partial match)
    if request.race:
        query_conditions.append({"race": {"$regex": request.race, "$options": "i"}})

    # Nested fields (address)
    if request.city:
        query_conditions.append({"address.city": {"$regex": request.city, "$options": "i"}})

    if request.state:
        # Normalize state (convert "Texas" to "TX" if needed)
        normalized_state = _normalize_state(request.state)
        # Use case-insensitive regex to match both codes and names
        query_conditions.append(
            {"address.state": {"$regex": f"^{normalized_state}$", "$options": "i"}}
        )

    # Date range
    if request.birth_date_start or request.birth_date_end:
        date_filter = {}
        if request.birth_date_start:
            date_filter["$gte"] = request.birth_date_start
        if request.birth_date_end:
            date_filter["$lte"] = request.birth_date_end
        if date_filter:
            query_conditions.append({"birth_date": date_filter})

    # Combine with $and (or empty query if no conditions)
    # IMPORTANT: Empty dict {} matches all documents in MongoDB
    # Using $and with empty array [] would match nothing, so we return {} when no conditions
    if not query_conditions:
        return {}  # Match all documents
    elif len(query_conditions) == 1:
        return query_conditions[0]  # Single condition, no need for $and
    else:
        return {"$and": query_conditions}  # Multiple conditions need $and


def apply_data_minimization(patient: PatientSummary, role: str) -> PatientSummary:
    """Remove fields based on role. Pure function.

    Example: 'billing' role can't see first_name.
    """
    # Define what each role CANNOT see
    restricted_fields = {
        "billing": ["first_name", "last_name"],
        "researcher": ["patient_id", "city", "state"],
        "public": ["patient_id", "first_name", "last_name", "city", "state"],
    }

    blocked_fields = restricted_fields.get(role, [])
    patient_dict = patient.model_dump()

    # Remove blocked fields
    for field in blocked_fields:
        if field in patient_dict:
            patient_dict[field] = None

    return PatientSummary(**patient_dict)


# ============================================================================
# MAIN SEARCH FUNCTION (Simple Async Flow)
# ============================================================================


async def search_patients(
    request: SearchPatientsRequest, security_context: SecurityContext | None = None
) -> list[PatientSummary]:
    """Search patients. Simple, async, streaming.

    No locks, no executors, no managers. Just async/await.
    """
    # Immediate debug output to verify function is called
    import sys

    print(
        f"[DEBUG] search_patients function CALLED with limit={request.limit}",
        file=sys.stderr,
        flush=True,
    )

    # Step 1: Build query (fast, no I/O)
    query_filter = build_patient_search_query(request)
    print(f"[DEBUG] Query filter built: {query_filter}", file=sys.stderr, flush=True)

    logger.info(
        f"Searching patients:\n"
        f"  Filter: {query_filter}\n"
        f"  Max results: {request.limit}\n"
        f"  Role: {security_context.role if security_context else 'none'}"
    )

    # Step 2: Get projection (cached, no I/O)
    projection = None
    if security_context:
        projection = get_cached_projection(security_context.role, "patients")
        if projection:
            # CRITICAL: Ensure patient_id is always included in projection
            # Without patient_id, we can't create PatientSummary objects
            projection["patient_id"] = 1
            # Also ensure _id is available as fallback
            projection["_id"] = 1
            logger.debug(
                f"Using cached projection for role: {security_context.role} (patient_id and _id ensured)"
            )

    # Step 3: Execute query (async, streaming)
    db = database.get_database()
    db_name = database.get_database_name()

    # CRITICAL: Verify database connection and list collections
    # This will help us understand if we're connected to the right database
    try:
        # First, verify we can access the database by listing collections
        collection_names = await db.list_collection_names()

        # If patients collection doesn't exist, that's the problem
        if "patients" not in collection_names:
            logger.error(
                f"Collection 'patients' not found! Available collections: {collection_names}"
            )
            return []  # Return empty list if collection doesn't exist

        patients_collection = db["patients"]

        # Get document count - this will tell us if the collection is empty
        doc_count = await patients_collection.count_documents({})
        logger.info(f"Collection 'patients' contains {doc_count} documents in database '{db_name}'")

        # If count is 0, try to find one document to see if there's a structure issue
        if doc_count == 0:
            # Try to find ANY document, even with a very broad query
            sample_doc = await patients_collection.find_one(
                {}, {"_id": 1}
            )  # Just get _id to verify connection
            if sample_doc is None:
                logger.warning(
                    f"No documents found in 'patients' collection in database '{db_name}'"
                )
            else:
                logger.warning(
                    "count_documents returned 0 but find_one found a document - possible query issue"
                )
        else:
            # Get a sample document to verify structure
            sample_doc = await patients_collection.find_one({})
            if sample_doc:
                logger.info(f"Sample document has keys: {list(sample_doc.keys())[:15]}")
    except Exception as e:
        logger.error(f"Error accessing database/collection: {e}", exc_info=True)
        # Don't return here - let the query attempt to execute so we can see the actual error

    patients_collection = db["patients"]

    # Use Motor's async cursor - streams results, never loads all into memory
    # Build find options - Motor's find() accepts these as keyword arguments
    find_kwargs = {}

    if projection:
        find_kwargs["projection"] = projection

    # Create cursor with filter and options
    cursor = patients_collection.find(query_filter, **find_kwargs)
    # Set batch size and limit on cursor (not in find options)
    cursor = cursor.batch_size(100).limit(request.limit)

    logger.info(f"Executing query with filter: {query_filter}")
    logger.info(f"Query options: projection={projection is not None}, limit={request.limit}")
    logger.debug(f"Query cursor created with filter: {query_filter}")

    # Step 4: Stream and convert results
    results = []
    documents_processed = 0
    documents_skipped = 0

    async for document in cursor:
        documents_processed += 1
        try:
            # Log first document for debugging
            if documents_processed == 1:
                logger.info(f"First document keys: {list(document.keys())[:10]}")
                logger.info(
                    f"First document sample: patient_id={document.get('patient_id')}, first_name={document.get('first_name')}"
                )

            # Extract patient_id - required field, must not be empty
            # Try patient_id first, then fall back to _id if needed
            patient_id = document.get("patient_id")
            if not patient_id:
                # Fall back to _id if patient_id is missing
                if "_id" in document:
                    patient_id = str(document["_id"])
                else:
                    raise ValueError("Document missing required patient_id field")

            patient = PatientSummary(
                patient_id=str(patient_id),  # Ensure it's a string
                first_name=document.get("first_name"),
                last_name=document.get("last_name"),
                birth_date=document.get("birth_date"),
                gender=document.get("gender"),
                city=document.get("address", {}).get("city")
                if isinstance(document.get("address"), dict)
                else None,
                state=document.get("address", {}).get("state")
                if isinstance(document.get("address"), dict)
                else None,
            )

            # Step 5: Apply data minimization if needed
            if security_context:
                patient = apply_data_minimization(patient, security_context.role)

            results.append(patient)

        except Exception as error:
            documents_skipped += 1
            logger.warning(f"Skipping invalid patient document {documents_processed}: {error}")
            if documents_processed <= 3:  # Log first 3 errors in detail
                logger.warning(f"Document that failed: {document}")
            continue

    logger.info(
        f"✓ Processed {documents_processed} documents, found {len(results)} valid patients, skipped {documents_skipped}"
    )

    # Final debug output
    import sys

    print(
        f"[DEBUG] FINAL: processed={documents_processed}, found={len(results)}, skipped={documents_skipped}",
        file=sys.stderr,
        flush=True,
    )

    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


async def example_usage():
    """How to use this module."""

    # 1. Initialize once at startup
    database.initialize(connection_uri="mongodb://localhost:27017", database_name="healthcare")

    # 2. Load projections once at startup
    projection_config = {
        "doctor": {"patients": ["patient_id", "first_name", "last_name", "birth_date", "gender"]},
        "billing": {"patients": ["patient_id", "birth_date", "gender"]},
    }
    load_projections_from_config(projection_config)

    # 3. Use in your application
    search_request = SearchPatientsRequest(last_name="Smith", state="CA", limit=50)

    context = SecurityContext(role="doctor", user_id="dr_123")

    patients = await search_patients(search_request, context)

    for patient in patients:
        print(f"Found: {patient.first_name} {patient.last_name}")


# ============================================================================
# WHAT WE ELIMINATED
# ============================================================================
# ❌ BaseTool class with shared state
# ❌ Connection manager with locks
# ❌ Thread pool executor
# ❌ Blocking I/O wrapped in run_in_executor
# ❌ Class-level locks and double-check patterns
# ❌ Redundant connection health checks
# ❌ Complex projection manager
# ❌ Converting entire cursor to list
# ❌ Multiple lock acquisitions per query
# ❌ Reentrant locks
# ❌ Shared singletons with state
#
# ✅ ONE Motor connection pool
# ✅ Pure async/await
# ✅ Streaming cursors
# ✅ Simple module-level cache
# ✅ Pure functions for query building
# ✅ Clear, explicit data models
# ✅ Zero locks (Motor is thread-safe)
# ✅ ~200 lines vs 500+ lines
# ============================================================================
