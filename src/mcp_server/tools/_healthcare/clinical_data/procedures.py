"""Procedure tools for surgical and medical procedures.

This module provides tools for querying and analyzing medical and
surgical procedures performed on patients from FHIR Procedure resources.

PATIENT ID DEPENDENCY: Supports optional patient_id filtering - when provided,
results are constrained to that patient's procedures. Recommended for patient-specific procedure history.
"""

import logging

logger = logging.getLogger(__name__)


class ProceduresTools:
    """Tools for querying and analyzing medical and surgical procedures."""

    def __init__(self):
        """Initialize procedures tools."""
        pass

    async def get_patient_procedures(
        self,
        patient_id=None,
        procedure_name=None,
        status=None,
        performed_date_start=None,
        performed_date_end=None,
        location=None,
        procedure_category=None,  # surgical, diagnostic, therapeutic, etc.
        limit=50,
        security_context=None,
    ):
        """Query patient medical and surgical procedures.

        This tool retrieves procedure information for patients, including
        surgical operations, diagnostic procedures, and therapeutic interventions.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        results are filtered to that patient's procedures only. When omitted,
        returns procedures across all patients (population-level analysis).

        Args:
            patient_id: Optional patient ID to filter procedures for specific patient
            procedure_name: Optional procedure name (partial match)
            status: Optional procedure status (completed, in-progress, stopped, etc.)
            performed_date_start: Optional start date for procedure performance
            performed_date_end: Optional end date for procedure performance
            location: Optional procedure location/facility
            procedure_category: Optional procedure category (surgical, diagnostic, etc.)
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing procedure query results with proper observability logging

        Raises:
            ValueError: If patient_id format is invalid when provided
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if procedures collection exists
        collection_name = "procedures"

        logger.warning(f"Procedures collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the query attempt for verification
        logger.info(
            f"\n{'=' * 70}\n"
            f"PROCEDURES QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Procedure Name: {procedure_name}\n"
            f"  Status: {status}\n"
            f"  Performed Date Range: {performed_date_start} to {performed_date_end}\n"
            f"  Location: {location}\n"
            f"  Category: {procedure_category}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Procedures collection '{collection_name}' does not exist in the database yet",
            "message": "Procedure FHIR resources have not been ingested yet",
            "query_parameters": {
                "patient_id": patient_id,
                "procedure_name": procedure_name,
                "status": status,
                "performed_date_start": performed_date_start,
                "performed_date_end": performed_date_end,
                "location": location,
                "procedure_category": procedure_category,
                "limit": limit,
            },
        }

    async def analyze_procedure_patterns(
        self,
        patient_id=None,
        group_by=None,  # "procedure_name", "status", "location", "time_period"
        procedure_category=None,
        status="completed",
        limit=20,
        security_context=None,
    ):
        """Analyze procedure patterns across patients or for a specific patient.

        This tool provides analytical insights into medical procedure patterns,
        either for population-level procedure analysis or patient-specific procedure history.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis is limited to that patient's procedures. When omitted, provides
        population-level procedure pattern analysis.

        Args:
            patient_id: Optional patient ID for patient-specific analysis
            group_by: How to group results ("procedure_name", "status", "location", "time_period")
            procedure_category: Optional filter by procedure category
            status: Procedure status to analyze (default: "completed")
            limit: Maximum results to return
            security_context: Security context for access control

        Returns:
            Dict containing procedure analysis results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if procedures collection exists
        collection_name = "procedures"
        logger.warning(f"Procedures collection '{collection_name}' does not exist yet")

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Procedures collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot perform procedure pattern analysis until Procedure data is ingested",
        }

    async def get_procedure_history(
        self,
        patient_id=None,
        procedure_types=None,  # List of specific procedure names to track
        performed_date_start=None,
        performed_date_end=None,
        include_outcomes=False,
        chronological_order=True,
        limit=100,
        security_context=None,
    ):
        """Get comprehensive procedure history for a patient.

        This tool provides a complete chronological history of medical procedures
        performed on a patient, with optional outcome tracking and filtering.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        returns procedure history for that specific patient. When omitted,
        provides population-level procedure statistics.

        Args:
            patient_id: Optional patient ID for patient-specific procedure history
            procedure_types: Optional list of procedure names to filter by
            performed_date_start: Optional start date for procedures
            performed_date_end: Optional end date for procedures
            include_outcomes: Whether to include procedure outcomes/results
            chronological_order: Whether to return results in chronological order (default: True)
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing comprehensive procedure history results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if procedures collection exists
        collection_name = "procedures"
        logger.warning(f"Procedures collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the procedure history query attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"PROCEDURE HISTORY QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Procedure Types: {procedure_types}\n"
            f"  Date Range: {performed_date_start} to {performed_date_end}\n"
            f"  Include Outcomes: {include_outcomes}\n"
            f"  Chronological Order: {chronological_order}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Procedures collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot query procedure history until Procedure data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "procedure_types": procedure_types,
                "performed_date_start": performed_date_start,
                "performed_date_end": performed_date_end,
                "include_outcomes": include_outcomes,
                "chronological_order": chronological_order,
                "limit": limit,
            },
        }

    async def analyze_procedure_outcomes(
        self,
        patient_id=None,
        procedure_name=None,
        procedure_category=None,
        time_period=None,  # "month", "quarter", "year"
        outcome_metrics=None,  # success_rate, complication_rate, etc.
        limit=20,
        security_context=None,
    ):
        """Analyze procedure outcomes and success rates.

        This tool analyzes the outcomes and success rates of medical procedures,
        providing insights into procedure effectiveness and complication rates.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis focuses on that patient's procedure outcomes. When omitted,
        provides population-level outcome analysis.

        Args:
            patient_id: Optional patient ID for patient-specific outcome analysis
            procedure_name: Optional specific procedure to analyze outcomes for
            procedure_category: Optional procedure category to analyze
            time_period: Optional time period for outcome analysis
            outcome_metrics: Optional specific outcome metrics to calculate
            limit: Maximum results to return
            security_context: Security context for access control

        Returns:
            Dict containing procedure outcome analysis results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if procedures collection exists
        collection_name = "procedures"
        logger.warning(f"Procedures collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the procedure outcomes analysis attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"PROCEDURE OUTCOMES ANALYSIS ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Procedure Name: {procedure_name}\n"
            f"  Category: {procedure_category}\n"
            f"  Time Period: {time_period}\n"
            f"  Outcome Metrics: {outcome_metrics}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Procedures collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot analyze procedure outcomes until Procedure data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "procedure_name": procedure_name,
                "procedure_category": procedure_category,
                "time_period": time_period,
                "outcome_metrics": outcome_metrics,
                "limit": limit,
            },
        }

    async def get_procedure_statistics(
        self,
        procedure_category=None,
        time_period="month",  # "day", "week", "month", "quarter", "year"
        performed_date_start=None,
        performed_date_end=None,
        location=None,
        include_demographics=False,
        limit=20,
        security_context=None,
    ):
        """Get statistical analysis of procedure frequencies and patterns.

        This tool provides statistical insights into procedure utilization,
        frequency patterns, and demographic correlations across the patient population.

        Args:
            procedure_category: Optional procedure category to analyze
            time_period: Time grouping period (default: "month")
            performed_date_start: Optional start date for analysis
            performed_date_end: Optional end date for analysis
            location: Optional location/facility filter
            include_demographics: Whether to include demographic correlations
            limit: Maximum results to return
            security_context: Security context for access control

        Returns:
            Dict containing procedure statistics and analysis results
        """
        # Check if procedures collection exists
        collection_name = "procedures"
        logger.warning(f"Procedures collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the procedure statistics query attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"PROCEDURE STATISTICS QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Category: {procedure_category}\n"
            f"  Time Period: {time_period}\n"
            f"  Date Range: {performed_date_start} to {performed_date_end}\n"
            f"  Location: {location}\n"
            f"  Include Demographics: {include_demographics}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Procedures collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot query procedure statistics until Procedure data is ingested",
            "query_parameters": {
                "procedure_category": procedure_category,
                "time_period": time_period,
                "performed_date_start": performed_date_start,
                "performed_date_end": performed_date_end,
                "location": location,
                "include_demographics": include_demographics,
                "limit": limit,
            },
        }
