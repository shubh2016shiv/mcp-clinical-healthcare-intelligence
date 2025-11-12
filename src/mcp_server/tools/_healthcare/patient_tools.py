"""Patient-focused MCP tools for healthcare data queries.

This module implements backward-compatible wrapper for patient tools, delegating
to the refactored _healthcare domain-specific tool classes.

ARCHITECTURE: This is now a wrapper class that uses composition to delegate to:
- PatientSearchTools from _healthcare/demographics/patient_search.py
- PatientTimelineTools from _healthcare/clinical_timeline/patient_timeline.py

This maintains backward compatibility for existing imports while eliminating code duplication.

Key Tools (delegated):
    - search_patients: Flexible patient search by demographics and identifiers
    - get_patient_clinical_timeline: Complete chronological clinical history

Design Philosophy:
    - Capability-based rather than collection-based tools
    - Single API calls replace multiple collection queries
    - Natural language mapping (e.g., "patient history" → clinical timeline)
    - Async operations with ThreadPoolExecutor for non-blocking database access
"""

import logging
from typing import Any

from ..base_tool import BaseTool
from ..models import (
    ClinicalTimelineRequest,
    ClinicalTimelineResponse,
    PatientSummary,
    SearchPatientsRequest,
)
from .clinical_timeline import PatientTimelineTools
from .demographics import search_patients as search_patients_func

logger = logging.getLogger(__name__)


class PatientTools(BaseTool):
    """Backward-compatible wrapper for patient-centric healthcare data queries.

    This class delegates to specialized tool classes organized by operational category:
    - search_patients_func: Handles patient demographics searches
    - PatientTimelineTools: Handles comprehensive clinical timelines

    All methods use the shared database connection manager and include proper error handling.
    Inherits optimized connection management from BaseTool, ensuring efficient
    connection pooling and minimal connection overhead.
    """

    def __init__(self):
        """Initialize patient tools with delegated tool instances."""
        super().__init__()
        # Initialize specialized tool instances
        self._timeline_tools = PatientTimelineTools()

    async def search_patients(
        self, request: SearchPatientsRequest, security_context: Any = None
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
            security_context: Optional security context for field projection and data minimization

        Returns:
            List of matching patient summaries
        """
        return await search_patients_func(request, security_context)

    async def get_patient_clinical_timeline(
        self, request: ClinicalTimelineRequest, security_context: Any = None
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
            security_context: Optional security context for field projection and data minimization

        Returns:
            Complete clinical timeline response
        """
        return await self._timeline_tools.get_patient_clinical_timeline(request, security_context)
