"""Immunization tools for vaccination records.

This module provides tools for querying and analyzing patient
vaccination and immunization records from FHIR Immunization resources.

PATIENT ID DEPENDENCY: Supports optional patient_id filtering - when provided,
results are constrained to that patient's immunization records. Recommended for patient-specific vaccination history.
"""

import logging

logger = logging.getLogger(__name__)


class ImmunizationsTools:
    """Tools for querying and analyzing patient immunization and vaccination records."""

    def __init__(self):
        """Initialize immunizations tools."""
        pass

    async def get_patient_immunizations(
        self,
        patient_id=None,
        vaccine_name=None,
        status=None,
        administration_date_start=None,
        administration_date_end=None,
        location=None,
        limit=50,
        security_context=None,
    ):
        """Query patient immunization and vaccination records.

        This tool retrieves vaccination information for patients, including
        vaccine types, administration dates, locations, and completion status.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        results are filtered to that patient's immunization records only. When omitted,
        returns immunization records across all patients (population-level analysis).

        Args:
            patient_id: Optional patient ID to filter immunizations for specific patient
            vaccine_name: Optional vaccine name (partial match)
            status: Optional immunization status (completed, not-done, etc.)
            administration_date_start: Optional start date for vaccine administration
            administration_date_end: Optional end date for vaccine administration
            location: Optional vaccination location/facility
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing immunization query results with proper observability logging

        Raises:
            ValueError: If patient_id format is invalid when provided
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if immunizations collection exists
        collection_name = "immunizations"

        logger.warning(f"Immunizations collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the query attempt for verification
        logger.info(
            f"\n{'='*70}\n"
            f"IMMUNIZATIONS QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Vaccine Name: {vaccine_name}\n"
            f"  Status: {status}\n"
            f"  Administration Date Range: {administration_date_start} to {administration_date_end}\n"
            f"  Location: {location}\n"
            f"  Limit: {limit}\n"
            f"{'='*70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Immunizations collection '{collection_name}' does not exist in the database yet",
            "message": "Immunization FHIR resources have not been ingested yet",
            "query_parameters": {
                "patient_id": patient_id,
                "vaccine_name": vaccine_name,
                "status": status,
                "administration_date_start": administration_date_start,
                "administration_date_end": administration_date_end,
                "location": location,
                "limit": limit,
            },
        }

    async def analyze_immunization_patterns(
        self,
        patient_id=None,
        group_by=None,  # "vaccine", "status", "location", "time_period"
        vaccine_name=None,
        status="completed",
        limit=20,
        security_context=None,
    ):
        """Analyze immunization patterns across patients or for a specific patient.

        This tool provides analytical insights into vaccination patterns, either
        for population-level immunization coverage or patient-specific vaccination analysis.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis is limited to that patient's immunizations. When omitted, provides
        population-level immunization pattern analysis.

        Args:
            patient_id: Optional patient ID for patient-specific analysis
            group_by: How to group results ("vaccine", "status", "location", "time_period")
            vaccine_name: Optional filter by specific vaccine type
            status: Immunization status to analyze (default: "completed")
            limit: Maximum results to return
            security_context: Security context for access control

        Returns:
            Dict containing immunization analysis results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if immunizations collection exists
        collection_name = "immunizations"
        logger.warning(f"Immunizations collection '{collection_name}' does not exist yet")

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Immunizations collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot perform immunization pattern analysis until Immunization data is ingested",
        }

    async def get_vaccine_history(
        self,
        patient_id=None,
        vaccine_types=None,  # List of vaccine names to check
        include_due_dates=False,
        limit=100,
        security_context=None,
    ):
        """Get comprehensive vaccine history for a patient or population.

        This tool provides a complete vaccination history, optionally checking
        for specific vaccine types and identifying potential vaccination gaps.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        returns vaccination history for that specific patient. When omitted,
        provides population-level vaccination statistics.

        Args:
            patient_id: Optional patient ID for patient-specific vaccine history
            vaccine_types: Optional list of vaccine names to filter by
            include_due_dates: Whether to include recommended vaccination schedules
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing comprehensive vaccine history results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if immunizations collection exists
        collection_name = "immunizations"
        logger.warning(f"Immunizations collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the vaccine history query attempt
        logger.info(
            f"\n{'='*70}\n"
            f"VACCINE HISTORY QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Vaccine Types: {vaccine_types}\n"
            f"  Include Due Dates: {include_due_dates}\n"
            f"  Limit: {limit}\n"
            f"{'='*70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Immunizations collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot query vaccine history until Immunization data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "vaccine_types": vaccine_types,
                "include_due_dates": include_due_dates,
                "limit": limit,
            },
        }

    async def check_vaccination_status(
        self,
        patient_id=None,
        vaccine_schedule=None,  # List of required vaccines with age ranges
        age_in_years=None,
        check_completeness=True,
        limit=50,
        security_context=None,
    ):
        """Check vaccination status and compliance for patients.

        This tool analyzes vaccination records against standard vaccination schedules
        to determine vaccination completeness and identify missing vaccinations.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        checks vaccination status for that specific patient. When omitted,
        provides population-level vaccination compliance statistics.

        Args:
            patient_id: Optional patient ID for patient-specific vaccination status
            vaccine_schedule: Optional list of required vaccines with age ranges
            age_in_years: Optional patient age for age-appropriate vaccine checking
            check_completeness: Whether to check for complete vaccination series
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing vaccination status and compliance results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if immunizations collection exists
        collection_name = "immunizations"
        logger.warning(f"Immunizations collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the vaccination status check attempt
        logger.info(
            f"\n{'='*70}\n"
            f"VACCINATION STATUS CHECK ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Vaccine Schedule: {vaccine_schedule}\n"
            f"  Age in Years: {age_in_years}\n"
            f"  Check Completeness: {check_completeness}\n"
            f"  Limit: {limit}\n"
            f"{'='*70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Immunizations collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot check vaccination status until Immunization data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "vaccine_schedule": vaccine_schedule,
                "age_in_years": age_in_years,
                "check_completeness": check_completeness,
                "limit": limit,
            },
        }