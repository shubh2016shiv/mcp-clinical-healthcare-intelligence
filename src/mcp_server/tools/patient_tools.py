"""Patient-focused MCP tools for healthcare data queries.

This module implements tools for searching patients and retrieving comprehensive
clinical timelines. These tools serve as the primary entry points for most
healthcare data queries.

Key Tools:
    - search_patients: Flexible patient search by demographics and identifiers
    - get_patient_clinical_timeline: Complete chronological clinical history

Design Philosophy:
    - Capability-based rather than collection-based tools
    - Single API calls replace multiple collection queries
    - Natural language mapping (e.g., "patient history" → clinical timeline)
    - Async operations with ThreadPoolExecutor for non-blocking database access
"""

import asyncio
import logging
from typing import Any

from ..database.async_executor import get_executor_pool
from ..security import get_security_manager
from ..security.projection_manager import get_projection_manager
from .base_tool import BaseTool
from .models import (
    ClinicalEvent,
    ClinicalTimelineRequest,
    ClinicalTimelineResponse,
    CollectionNames,
    PatientSummary,
    SearchPatientsRequest,
)
from .utils import (
    build_compound_filter,
    build_date_filter,
    build_text_filter,
    handle_mongo_errors,
    safe_get_nested_value,
)

logger = logging.getLogger(__name__)


class PatientTools(BaseTool):
    """Tools for patient-centric healthcare data queries.

    This class provides methods for searching patients and retrieving
    comprehensive clinical timelines. All methods use the shared database
    connection manager and include proper error handling.

    Inherits optimized connection management from BaseTool, ensuring
    efficient connection pooling and minimal connection overhead.
    """

    def __init__(self):
        """Initialize patient tools with optimized database connection."""
        super().__init__()

    @handle_mongo_errors
    async def search_patients(
        self, request: SearchPatientsRequest, security_context=None
    ) -> list[PatientSummary]:
        """Search for patients using flexible criteria.

        This tool supports searching patients by various demographic fields,
        identifiers, and location information. All text searches are case-insensitive
        partial matches.

        Query Strategy:
        - Builds dynamic filter based on provided fields
        - Uses regex for flexible text matching
        - Limits results for performance
        - Returns structured patient summaries
        - Non-blocking async operations using ThreadPoolExecutor

        Args:
            request: Search criteria including names, demographics, location

        Returns:
            List of matching patient summaries
        """
        db = self.get_database()
        collection = db[CollectionNames.PATIENTS.value]

        # Build search filters dynamically
        filters = []

        # Text field searches (case-insensitive partial matches)
        if request.first_name:
            filters.append(build_text_filter("first_name", request.first_name))

        if request.last_name:
            filters.append(build_text_filter("last_name", request.last_name))

        # Exact match fields
        if request.patient_id:
            filters.append({"patient_id": request.patient_id})

        if request.gender:
            filters.append({"gender": request.gender})

        # Nested address fields
        if request.city:
            filters.append(build_text_filter("address.city", request.city))

        if request.state:
            filters.append({"address.state": request.state})

        # Date range for birth_date
        if request.birth_date_start or request.birth_date_end:
            birth_filter = build_date_filter(
                "birth_date", request.birth_date_start, request.birth_date_end
            )
            if birth_filter:
                filters.append(birth_filter)

        # Combine all filters
        query_filter = build_compound_filter(*filters)

        logger.debug(f"Patient search query: {query_filter}")

        # Execute blocking query in thread pool
        loop = asyncio.get_event_loop()
        executor = get_executor_pool().get_executor()

        # Get query-level projection based on security context
        projection = None
        if security_context and security_context.role:
            projection_manager = get_projection_manager()
            projection = projection_manager.get_query_projection(security_context.role, "patient")
            logger.debug(f"Applied query projection for role: {security_context.role}")

        # Run find and limit operations in executor with projection
        def execute_find():
            if projection:
                return collection.find(query_filter, projection).limit(request.limit)
            else:
                return collection.find(query_filter).limit(request.limit)

        cursor = await loop.run_in_executor(executor, execute_find)

        # Convert cursor to list in executor (blocking I/O)
        docs = await loop.run_in_executor(executor, list, cursor)

        # Convert to PatientSummary objects
        results = []
        for doc in docs:
            patient = PatientSummary(
                patient_id=doc.get("patient_id", ""),
                first_name=safe_get_nested_value(doc, "first_name"),
                last_name=safe_get_nested_value(doc, "last_name"),
                birth_date=doc.get("birth_date"),
                gender=doc.get("gender"),
                city=safe_get_nested_value(doc, "address.city"),
                state=safe_get_nested_value(doc, "address.state"),
            )
            results.append(patient)

        # Apply data minimization based on security context
        if security_context:
            security_manager = get_security_manager()
            # Safely convert PatientSummary objects to dicts with validation
            patient_dicts = []
            for patient in results:
                if not isinstance(patient, PatientSummary):
                    logger.warning(
                        f"Skipping non-PatientSummary object in results: {type(patient)}"
                    )
                    continue
                try:
                    patient_dicts.append(patient.model_dump())
                except Exception as e:
                    logger.error(
                        f"Failed to convert PatientSummary to dict: {e}, type: {type(patient)}"
                    )
                    continue

            minimized_data = security_manager.data_minimizer.filter_record_list(
                patient_dicts, security_context.role
            )
            # Validate and convert back to PatientSummary objects
            # Ensure filter_record_list returned a list of dicts
            if not isinstance(minimized_data, list):
                logger.warning(f"filter_record_list returned non-list type: {type(minimized_data)}")
                minimized_data = []

            results = []
            for patient_dict in minimized_data:
                # Validate each item is a dict before conversion
                if not isinstance(patient_dict, dict):
                    logger.warning(f"Skipping invalid patient data type: {type(patient_dict)}")
                    continue
                try:
                    results.append(PatientSummary(**patient_dict))
                except Exception as e:
                    logger.error(
                        f"Failed to convert patient dict to PatientSummary: {e}, data: {patient_dict}"
                    )
                    # Continue processing other records instead of failing completely
                    continue

        logger.info(f"Found {len(results)} patients matching search criteria")
        return results

    @handle_mongo_errors
    async def get_patient_clinical_timeline(
        self, request: ClinicalTimelineRequest, security_context=None
    ) -> ClinicalTimelineResponse:
        """Retrieve comprehensive clinical timeline for a patient.

        This tool fetches a complete chronological history of clinical events
        for a specific patient across multiple collections. It aggregates data
        from encounters, conditions, medications, procedures, immunizations,
        and observations into a unified timeline.

        Aggregation Strategy:
        - Queries multiple collections in parallel using asyncio.gather()
        - Merges results chronologically
        - Filters by date range and event types if specified
        - Returns structured timeline with event categorization
        - Non-blocking async operations with ThreadPoolExecutor

        Args:
            request: Timeline request with patient ID and optional filters

        Returns:
            Complete clinical timeline response
        """
        db = self.get_database()

        # Define event sources and their mapping
        event_sources = {
            "encounters": {
                "collection": CollectionNames.ENCOUNTERS.value,
                "date_field": "start_date",
                "name_field": "visit_reason",
                "fields": ["location", "provider", "encounter_type", "status"],
            },
            "conditions": {
                "collection": CollectionNames.CONDITIONS.value,
                "date_field": "onset_date",
                "name_field": "condition_name",
                "fields": ["status", "verification_status", "severity", "description"],
            },
            "medications": {
                "collection": CollectionNames.MEDICATIONS.value,
                "date_field": "prescribed_date",
                "name_field": "medication_name",
                "fields": ["prescriber", "status", "dosage_instruction"],
            },
            "procedures": {
                "collection": CollectionNames.PROCEDURES.value,
                "date_field": "performed_date",
                "name_field": "procedure_name",
                "fields": ["location", "status", "outcome"],
            },
            "immunizations": {
                "collection": CollectionNames.IMMUNIZATIONS.value,
                "date_field": "administration_date",
                "name_field": "vaccine_name",
                "fields": ["location", "status", "lot_number"],
            },
            "observations": {
                "collection": CollectionNames.OBSERVATIONS.value,
                "date_field": "effective_date_time",
                "name_field": "test_name",
                "fields": ["value", "unit", "observation_type", "status"],
            },
        }

        # Filter event types if specified
        if request.event_types:
            event_sources = {k: v for k, v in event_sources.items() if k in request.event_types}

        # Get query-level projection based on security context
        projection = None
        if security_context and security_context.role:
            projection_manager = get_projection_manager()
            projection = projection_manager.get_query_projection(
                security_context.role, "clinical_event"
            )
            logger.debug(
                f"Applied query projection for clinical timeline, role: {security_context.role}"
            )

        # Define async task for fetching events from a single collection
        async def fetch_collection_events(
            event_type: str, source_config: dict[str, Any]
        ) -> list[ClinicalEvent]:
            """Fetch events from a single collection asynchronously.

            Args:
                event_type: Type of event (encounters, conditions, etc.)
                source_config: Configuration for this event source

            Returns:
                List of ClinicalEvent objects from this collection
            """
            loop = asyncio.get_event_loop()
            executor = get_executor_pool().get_executor()

            collection = db[source_config["collection"]]

            # Build query for this collection
            query = {"patient_id": request.patient_id}

            # Add date filter if provided
            if request.start_date or request.end_date:
                date_filter = build_date_filter(
                    source_config["date_field"], request.start_date, request.end_date
                )
                if date_filter:
                    query.update(date_filter)

            logger.debug(f"Querying {event_type}: {query}")

            # Execute query in thread pool with projection
            def execute_collection_find():
                if projection:
                    return collection.find(query, projection).limit(request.limit)
                else:
                    return collection.find(query).limit(request.limit)

            cursor = await loop.run_in_executor(executor, execute_collection_find)

            # Convert cursor to list in executor
            docs = await loop.run_in_executor(executor, list, cursor)

            # Convert documents to ClinicalEvent objects
            events = []
            for doc in docs:
                event = ClinicalEvent(
                    event_type=event_type,
                    event_date=doc.get(source_config["date_field"]),
                    event_name=doc.get(source_config["name_field"]),
                    source_id=str(doc.get("_id")),
                    details={
                        field: doc.get(field)
                        for field in source_config["fields"]
                        if doc.get(field) is not None
                    },
                )
                events.append(event)

            return events

        # Create tasks for all collections (parallel execution)
        tasks = [
            fetch_collection_events(event_type, source_config)
            for event_type, source_config in event_sources.items()
        ]

        # Execute all tasks concurrently and await results
        results = await asyncio.gather(*tasks)

        # Flatten results from all collections
        timeline_events = [event for events in results for event in events]

        # Sort by date descending (most recent first)
        # Handle None dates by placing them at the end
        def sort_key(event):
            if event.event_date:
                try:
                    return event.event_date
                except (TypeError, ValueError):
                    pass
            return ""  # Empty string sorts last

        timeline_events.sort(key=sort_key, reverse=True)

        # Apply final limit after sorting
        timeline_events = timeline_events[: request.limit]

        # Apply data minimization based on security context
        if security_context:
            security_manager = get_security_manager()
            minimized_events = security_manager.data_minimizer.filter_record_list(
                [event.model_dump() for event in timeline_events], security_context.role
            )
            # Convert back to ClinicalEvent objects
            timeline_events = [ClinicalEvent(**event_dict) for event_dict in minimized_events]

        # Build response
        response = ClinicalTimelineResponse(
            patient_id=request.patient_id,
            total_events=len(timeline_events),
            events=timeline_events,
            date_range={"start": request.start_date, "end": request.end_date},
        )

        logger.info(
            f"Retrieved {len(timeline_events)} clinical events for patient {request.patient_id}"
        )
        return response
