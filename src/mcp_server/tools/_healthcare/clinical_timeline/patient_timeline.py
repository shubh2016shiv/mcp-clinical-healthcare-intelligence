"""Patient clinical timeline tools for comprehensive patient history.

This module provides tools for retrieving complete chronological
clinical timelines for patients across all clinical events.
"""

import asyncio
import logging
from typing import Any

# Async executor removed - now using pure Motor async
from src.mcp_server.security import get_security_manager
from src.mcp_server.tools.base_tool import BaseTool
from src.mcp_server.tools.models import (
    ClinicalEvent,
    ClinicalTimelineRequest,
    ClinicalTimelineResponse,
    CollectionNames,
)
from src.mcp_server.tools.utils import build_date_filter, handle_mongo_errors

logger = logging.getLogger(__name__)


class PatientTimelineTools(BaseTool):
    """Tools for retrieving patient clinical timelines."""

    def __init__(self):
        """Initialize patient timeline tools."""
        super().__init__()

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

        PATIENT ID DEPENDENCY (REQUIRED): This tool operates on a single patient.
        The patient_id field is mandatory and is used to fetch all related clinical events.

        Args:
            request: Timeline request with patient ID (REQUIRED) and optional filters
            security_context: Security context for access control and data minimization (field projection)

        Returns:
            Complete clinical timeline response containing all clinical events for patient

        Raises:
            ValueError: If patient_id is missing in request
        """
        # PATIENT ID VALIDATION
        if not request.patient_id:
            error_msg = "Patient ID is required for clinical timeline retrieval"
            logger.error(error_msg)
            raise ValueError(error_msg)

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
        # Note: Projection is optional - Motor handles this natively
        projection = None
        if security_context and security_context.role:
            # Simple projection cache could be added here if needed
            # For now, we rely on data minimization after query
            logger.debug(
                f"Security context applied for clinical timeline, role: {security_context.role}"
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
            collection = db[source_config["collection"]]

            # Build query for this collection (PATIENT ID REQUIRED)
            query = {"patient_id": request.patient_id}

            # Add date filter if provided
            if request.start_date or request.end_date:
                date_filter = build_date_filter(
                    source_config["date_field"], request.start_date, request.end_date
                )
                if date_filter:
                    query.update(date_filter)

            logger.debug(f"Querying {event_type}: {query}")

            # Execute query directly with Motor (async-native)
            if projection:
                cursor = collection.find(query, projection).limit(request.limit)
            else:
                cursor = collection.find(query).limit(request.limit)

            # Convert cursor to list directly
            docs = await cursor.to_list(length=request.limit)

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

        logger.info(
            f"\n{'=' * 70}\n"
            f"CLINICAL TIMELINE QUERY:\n"
            f"  Patient ID: {request.patient_id}\n"
            f"  Event Types: {list(event_sources.keys())}\n"
            f"  Date Range: {request.start_date} to {request.end_date}\n"
            f"  Limit: {request.limit}\n"
            f"{'=' * 70}"
        )

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
            f"✓ Retrieved {len(timeline_events)} clinical events for patient {request.patient_id}"
        )
        return response
