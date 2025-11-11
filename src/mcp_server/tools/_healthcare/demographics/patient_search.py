"""Patient search tools for finding patients by demographics.

This module provides tools for searching and identifying patients
based on demographic information and identifiers.
"""

import asyncio
import logging

from src.mcp_server.database.async_executor import get_executor_pool
from src.mcp_server.security import get_security_manager
from src.mcp_server.security.projection_manager import get_projection_manager
from src.mcp_server.tools.base_tool import BaseTool
from src.mcp_server.tools.models import (
    CollectionNames,
    PatientSummary,
    SearchPatientsRequest,
)
from src.mcp_server.tools.utils import (
    build_compound_filter,
    build_date_filter,
    build_text_filter,
    handle_mongo_errors,
    safe_get_nested_value,
)

logger = logging.getLogger(__name__)


class PatientSearchTools(BaseTool):
    """Tools for patient search and demographic queries."""

    def __init__(self):
        """Initialize patient search tools."""
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

        PATIENT ID DEPENDENCY: This tool is patient-agnostic (searches across all patients).
        Use patient_id filter when available to focus on specific patients.

        Args:
            request: Search criteria including names, demographics, location
            security_context: Security context for access control and data minimization (field projection)

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

        logger.info(
            f"\n{'=' * 70}\n"
            f"PATIENT SEARCH QUERY:\n"
            f"  Filter: {query_filter}\n"
            f"  Limit: {request.limit}\n"
            f"{'=' * 70}"
        )

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

        logger.info(f"✓ Found {len(results)} patients matching search criteria")
        return results
