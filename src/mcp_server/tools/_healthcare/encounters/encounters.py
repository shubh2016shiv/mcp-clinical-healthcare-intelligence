"""Patient encounter and visit tools.

This module provides tools for querying and analyzing patient
encounters, visits, and healthcare service interactions from FHIR Encounter resources.

PATIENT ID DEPENDENCY: Supports optional patient_id filtering - when provided,
results are constrained to that patient's encounters. Recommended for patient-specific visit history.
"""

import logging

logger = logging.getLogger(__name__)


class EncountersTools:
    """Tools for querying and analyzing patient encounters and healthcare visits."""

    def __init__(self):
        """Initialize encounters tools."""
        pass

    async def get_patient_encounters(
        self,
        patient_id=None,
        encounter_type=None,
        status=None,
        start_date=None,
        end_date=None,
        location=None,
        provider=None,
        visit_reason=None,
        limit=50,
        security_context=None,
    ):
        """Query patient encounters and healthcare visits.

        This tool retrieves encounter information for patients, including
        inpatient stays, outpatient visits, emergency room visits, and other
        healthcare service interactions.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        results are filtered to that patient's encounters only. When omitted,
        returns encounters across all patients (population-level analysis).

        Args:
            patient_id: Optional patient ID to filter encounters for specific patient
            encounter_type: Optional encounter type (ambulatory, inpatient, emergency, etc.)
            status: Optional encounter status (finished, in-progress, cancelled, etc.)
            start_date: Optional start date for encounter filtering
            end_date: Optional end date for encounter filtering
            location: Optional encounter location/facility
            provider: Optional healthcare provider involved
            visit_reason: Optional reason for the visit (partial match)
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing encounter query results with proper observability logging

        Raises:
            ValueError: If patient_id format is invalid when provided
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if encounters collection exists
        collection_name = "encounters"

        logger.warning(f"Encounters collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the query attempt for verification
        logger.info(
            f"\n{'=' * 70}\n"
            f"ENCOUNTERS QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Encounter Type: {encounter_type}\n"
            f"  Status: {status}\n"
            f"  Date Range: {start_date} to {end_date}\n"
            f"  Location: {location}\n"
            f"  Provider: {provider}\n"
            f"  Visit Reason: {visit_reason}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Encounters collection '{collection_name}' does not exist in the database yet",
            "message": "Encounter FHIR resources have not been ingested yet",
            "query_parameters": {
                "patient_id": patient_id,
                "encounter_type": encounter_type,
                "status": status,
                "start_date": start_date,
                "end_date": end_date,
                "location": location,
                "provider": provider,
                "visit_reason": visit_reason,
                "limit": limit,
            },
        }

    async def analyze_encounter_patterns(
        self,
        patient_id=None,
        group_by=None,  # "encounter_type", "status", "location", "provider", "time_period"
        encounter_type=None,
        status="finished",
        limit=20,
        security_context=None,
    ):
        """Analyze encounter patterns across patients or for a specific patient.

        This tool provides analytical insights into healthcare encounter patterns,
        either for population-level utilization analysis or patient-specific visit patterns.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis is limited to that patient's encounters. When omitted, provides
        population-level encounter pattern analysis.

        Args:
            patient_id: Optional patient ID for patient-specific analysis
            group_by: How to group results ("encounter_type", "status", "location", "provider", "time_period")
            encounter_type: Optional filter by encounter type
            status: Encounter status to analyze (default: "finished")
            limit: Maximum results to return
            security_context: Security context for access control

        Returns:
            Dict containing encounter analysis results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if encounters collection exists
        collection_name = "encounters"
        logger.warning(f"Encounters collection '{collection_name}' does not exist yet")

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Encounters collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot perform encounter pattern analysis until Encounter data is ingested",
        }

    async def get_encounter_history(
        self,
        patient_id=None,
        encounter_types=None,  # List of encounter types to include
        start_date=None,
        end_date=None,
        chronological_order=True,
        include_visit_details=True,
        limit=100,
        security_context=None,
    ):
        """Get comprehensive encounter history for a patient.

        This tool provides a complete chronological history of healthcare encounters
        for a patient, useful for understanding care continuity and utilization patterns.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        returns encounter history for that specific patient. When omitted,
        provides population-level encounter statistics.

        Args:
            patient_id: Optional patient ID for patient-specific encounter history
            encounter_types: Optional list of encounter types to filter by
            start_date: Optional start date for encounter history
            end_date: Optional end date for encounter history
            chronological_order: Whether to return results in chronological order (default: True)
            include_visit_details: Whether to include detailed visit information
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing comprehensive encounter history results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if encounters collection exists
        collection_name = "encounters"
        logger.warning(f"Encounters collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the encounter history query attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"ENCOUNTER HISTORY QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Encounter Types: {encounter_types}\n"
            f"  Date Range: {start_date} to {end_date}\n"
            f"  Chronological Order: {chronological_order}\n"
            f"  Include Visit Details: {include_visit_details}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Encounters collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot query encounter history until Encounter data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "encounter_types": encounter_types,
                "start_date": start_date,
                "end_date": end_date,
                "chronological_order": chronological_order,
                "include_visit_details": include_visit_details,
                "limit": limit,
            },
        }

    async def analyze_visit_reasons(
        self,
        patient_id=None,
        encounter_type=None,
        time_period="month",  # "day", "week", "month", "quarter", "year"
        start_date=None,
        end_date=None,
        min_frequency=1,
        limit=25,
        security_context=None,
    ):
        """Analyze common visit reasons and encounter patterns.

        This tool analyzes the most common reasons for healthcare visits,
        providing insights into healthcare utilization and common presenting complaints.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis focuses on that patient's visit reasons. When omitted,
        provides population-level visit reason analysis.

        Args:
            patient_id: Optional patient ID for patient-specific visit reason analysis
            encounter_type: Optional filter by encounter type
            time_period: Time grouping period (default: "month")
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            min_frequency: Minimum frequency threshold for inclusion
            limit: Maximum results to return
            security_context: Security context for access control

        Returns:
            Dict containing visit reason analysis results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if encounters collection exists
        collection_name = "encounters"
        logger.warning(f"Encounters collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the visit reasons analysis attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"VISIT REASONS ANALYSIS ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Encounter Type: {encounter_type}\n"
            f"  Time Period: {time_period}\n"
            f"  Date Range: {start_date} to {end_date}\n"
            f"  Min Frequency: {min_frequency}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Encounters collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot analyze visit reasons until Encounter data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "encounter_type": encounter_type,
                "time_period": time_period,
                "start_date": start_date,
                "end_date": end_date,
                "min_frequency": min_frequency,
                "limit": limit,
            },
        }

    async def get_provider_encounters(
        self,
        provider=None,
        patient_id=None,
        encounter_type=None,
        status="finished",
        start_date=None,
        end_date=None,
        limit=50,
        security_context=None,
    ):
        """Get encounters for a specific healthcare provider.

        This tool provides insights into a provider's patient encounters,
        useful for provider productivity analysis, patient load assessment, and quality metrics.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        filters to encounters between the specified provider and patient.

        Args:
            provider: Optional healthcare provider name to filter encounters
            patient_id: Optional patient ID to filter encounters with specific patient
            encounter_type: Optional encounter type filter
            status: Encounter status to filter (default: "finished")
            start_date: Optional start date for encounter filtering
            end_date: Optional end date for encounter filtering
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing provider encounter results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if encounters collection exists
        collection_name = "encounters"
        logger.warning(f"Encounters collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the provider encounters query attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"PROVIDER ENCOUNTERS QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Provider: {provider}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Encounter Type: {encounter_type}\n"
            f"  Status: {status}\n"
            f"  Date Range: {start_date} to {end_date}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Encounters collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot query provider encounters until Encounter data is ingested",
            "query_parameters": {
                "provider": provider,
                "patient_id": patient_id,
                "encounter_type": encounter_type,
                "status": status,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
            },
        }
