"""Allergy and intolerance tools.

This module provides tools for querying and analyzing patient
allergies and drug intolerances from AllergyIntolerance FHIR resources.

PATIENT ID DEPENDENCY: Supports optional patient_id filtering - when provided,
results are constrained to that patient's allergies. Recommended for patient-specific allergy analysis.

Note: The allergies collection is not yet present in the database. This tool
is prepared for when AllergyIntolerance resources become available.
"""

import logging

logger = logging.getLogger(__name__)


class AllergyTools:
    """Tools for querying and analyzing patient allergies and intolerances."""

    def __init__(self):
        """Initialize allergy tools."""
        pass

    async def get_patient_allergies(
        self,
        patient_id=None,
        allergy_name=None,
        category=None,
        severity=None,
        status=None,
        recorded_date_start=None,
        recorded_date_end=None,
        limit=50,
        security_context=None,
    ):
        """Query patient allergies and intolerances.

        This tool retrieves allergy and intolerance information for patients.
        Allergies are critical clinical data that can affect treatment decisions.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        results are filtered to that patient's allergies only. When omitted,
        returns allergies across all patients (population-level analysis).

        Args:
            patient_id: Optional patient ID to filter allergies for specific patient
            allergy_name: Optional allergy/intolerance name (partial match)
            category: Optional allergy category (food, medication, environment, etc.)
            severity: Optional severity level (mild, moderate, severe)
            status: Optional status (active, inactive, resolved)
            recorded_date_start: Optional start date for allergy recording
            recorded_date_end: Optional end date for allergy recording
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing allergy query results with proper observability logging

        Raises:
            ValueError: If patient_id format is invalid when provided
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if allergies collection exists
        collection_name = "allergies"

        logger.warning(f"Allergies collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the query attempt for verification
        logger.info(
            f"\n{'=' * 70}\n"
            f"ALLERGIES QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Allergy Name: {allergy_name}\n"
            f"  Category: {category}\n"
            f"  Severity: {severity}\n"
            f"  Status: {status}\n"
            f"  Date Range: {recorded_date_start} to {recorded_date_end}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Allergies collection '{collection_name}' does not exist in the database yet",
            "message": "AllergyIntolerance FHIR resources have not been ingested yet",
            "query_parameters": {
                "patient_id": patient_id,
                "allergy_name": allergy_name,
                "category": category,
                "severity": severity,
                "status": status,
                "recorded_date_start": recorded_date_start,
                "recorded_date_end": recorded_date_end,
                "limit": limit,
            },
        }

    async def analyze_allergy_patterns(
        self,
        patient_id=None,
        group_by=None,  # "allergy", "category", "severity", "status"
        limit=20,
        security_context=None,
    ):
        """Analyze allergy patterns across patients or for a specific patient.

        This tool provides analytical insights into allergy patterns, either
        for population-level analysis or patient-specific allergy patterns.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis is limited to that patient's allergies. When omitted, provides
        population-level allergy pattern analysis.

        Args:
            patient_id: Optional patient ID for patient-specific analysis
            group_by: How to group results ("allergy", "category", "severity", "status")
            limit: Maximum results to return
            security_context: Security context for access control

        Returns:
            Dict containing allergy analysis results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if allergies collection exists
        collection_name = "allergies"
        logger.warning(f"Allergies collection '{collection_name}' does not exist yet")

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Allergies collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot perform allergy pattern analysis until AllergyIntolerance data is ingested",
        }
