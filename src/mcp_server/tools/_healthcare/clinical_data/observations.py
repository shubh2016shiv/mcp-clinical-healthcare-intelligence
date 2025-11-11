"""Observation tools for lab results and clinical measurements.

This module provides tools for querying lab results, vital signs,
and other clinical observations from FHIR Observation resources.

PATIENT ID DEPENDENCY: Supports optional patient_id filtering - when provided,
results are constrained to that patient's observations. Recommended for patient-specific clinical data analysis.
"""

import logging

logger = logging.getLogger(__name__)


class ObservationsTools:
    """Tools for querying and analyzing patient clinical observations and measurements."""

    def __init__(self):
        """Initialize observations tools."""
        pass

    async def get_patient_observations(
        self,
        patient_id=None,
        observation_type=None,
        test_name=None,
        status=None,
        test_date_start=None,
        test_date_end=None,
        value_min=None,
        value_max=None,
        unit=None,
        limit=50,
        security_context=None,
    ):
        """Query patient clinical observations and measurements.

        This tool retrieves clinical observations including lab results, vital signs,
        and other measurements with comprehensive filtering capabilities.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        results are filtered to that patient's observations only. When omitted,
        returns observations across all patients (population-level analysis).

        Args:
            patient_id: Optional patient ID to filter observations for specific patient
            observation_type: Optional type (vital_sign, lab, etc.)
            test_name: Optional observation/test name (partial match)
            status: Optional observation status (final, preliminary, etc.)
            test_date_start: Optional start date for observation
            test_date_end: Optional end date for observation
            value_min: Optional minimum measurement value
            value_max: Optional maximum measurement value
            unit: Optional measurement unit (cm, kg, mmHg, etc.)
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing observation query results with proper observability logging

        Raises:
            ValueError: If patient_id format is invalid when provided
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if observations collection exists
        collection_name = "observations"

        logger.warning(f"Observations collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the query attempt for verification
        logger.info(
            f"\n{'=' * 70}\n"
            f"OBSERVATIONS QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Observation Type: {observation_type}\n"
            f"  Test Name: {test_name}\n"
            f"  Status: {status}\n"
            f"  Test Date Range: {test_date_start} to {test_date_end}\n"
            f"  Value Range: {value_min} to {value_max}\n"
            f"  Unit: {unit}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Observations collection '{collection_name}' does not exist in the database yet",
            "message": "Observation FHIR resources have not been ingested yet",
            "query_parameters": {
                "patient_id": patient_id,
                "observation_type": observation_type,
                "test_name": test_name,
                "status": status,
                "test_date_start": test_date_start,
                "test_date_end": test_date_end,
                "value_min": value_min,
                "value_max": value_max,
                "unit": unit,
                "limit": limit,
            },
        }

    async def analyze_observation_patterns(
        self,
        patient_id=None,
        group_by=None,  # "test_name", "observation_type", "unit", "time_period"
        observation_type=None,
        test_name=None,
        status="final",
        limit=20,
        security_context=None,
    ):
        """Analyze observation patterns across patients or for a specific patient.

        This tool provides analytical insights into clinical observation patterns,
        either for population-level analysis or patient-specific measurement trends.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis is limited to that patient's observations. When omitted, provides
        population-level observation pattern analysis.

        Args:
            patient_id: Optional patient ID for patient-specific analysis
            group_by: How to group results ("test_name", "observation_type", "unit", "time_period")
            observation_type: Optional filter by observation type
            test_name: Optional filter by specific test/measurement
            status: Observation status to analyze (default: "final")
            limit: Maximum results to return
            security_context: Security context for access control

        Returns:
            Dict containing observation analysis results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if observations collection exists
        collection_name = "observations"
        logger.warning(f"Observations collection '{collection_name}' does not exist yet")

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Observations collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot perform observation pattern analysis until Observation data is ingested",
        }

    async def get_vital_signs_history(
        self,
        patient_id=None,
        vital_sign_types=None,  # List of vital signs to retrieve
        test_date_start=None,
        test_date_end=None,
        include_trends=False,
        limit=100,
        security_context=None,
    ):
        """Get comprehensive vital signs history for a patient.

        This tool specializes in retrieving vital signs measurements with
        optional trend analysis for monitoring patient health over time.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        returns vital signs history for that specific patient. When omitted,
        provides population-level vital signs statistics.

        Args:
            patient_id: Optional patient ID for patient-specific vital signs history
            vital_sign_types: Optional list of vital sign types (height, weight, blood_pressure, etc.)
            test_date_start: Optional start date for vital signs
            test_date_end: Optional end date for vital signs
            include_trends: Whether to calculate trend analysis
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing vital signs history results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if observations collection exists
        collection_name = "observations"
        logger.warning(f"Observations collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the vital signs history query attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"VITAL SIGNS HISTORY QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Vital Sign Types: {vital_sign_types}\n"
            f"  Date Range: {test_date_start} to {test_date_end}\n"
            f"  Include Trends: {include_trends}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Observations collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot query vital signs history until Observation data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "vital_sign_types": vital_sign_types,
                "test_date_start": test_date_start,
                "test_date_end": test_date_end,
                "include_trends": include_trends,
                "limit": limit,
            },
        }

    async def get_lab_results(
        self,
        patient_id=None,
        lab_test_names=None,  # List of lab tests to retrieve
        status="final",
        test_date_start=None,
        test_date_end=None,
        abnormal_only=False,
        limit=50,
        security_context=None,
    ):
        """Get laboratory test results for a patient or population.

        This tool specializes in retrieving laboratory test results with
        filtering capabilities for normal/abnormal results and specific test types.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        returns lab results for that specific patient. When omitted,
        provides population-level lab results analysis.

        Args:
            patient_id: Optional patient ID for patient-specific lab results
            lab_test_names: Optional list of laboratory test names
            status: Lab result status (default: "final")
            test_date_start: Optional start date for lab tests
            test_date_end: Optional end date for lab tests
            abnormal_only: Whether to return only abnormal results
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing laboratory results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if observations collection exists
        collection_name = "observations"
        logger.warning(f"Observations collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the lab results query attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"LAB RESULTS QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Lab Test Names: {lab_test_names}\n"
            f"  Status: {status}\n"
            f"  Date Range: {test_date_start} to {test_date_end}\n"
            f"  Abnormal Only: {abnormal_only}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Observations collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot query lab results until Observation data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "lab_test_names": lab_test_names,
                "status": status,
                "test_date_start": test_date_start,
                "test_date_end": test_date_end,
                "abnormal_only": abnormal_only,
                "limit": limit,
            },
        }

    async def analyze_measurement_trends(
        self,
        patient_id=None,
        test_name=None,
        observation_type=None,
        time_period="month",  # "day", "week", "month", "quarter", "year"
        test_date_start=None,
        test_date_end=None,
        calculate_statistics=True,
        limit=20,
        security_context=None,
    ):
        """Analyze measurement trends over time for clinical observations.

        This tool provides trend analysis for clinical measurements, calculating
        statistical trends, averages, and changes over specified time periods.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis focuses on that patient's measurement trends. When omitted,
        provides population-level trend analysis.

        Args:
            patient_id: Optional patient ID for patient-specific trend analysis
            test_name: Optional specific test/measurement to analyze
            observation_type: Optional observation type to analyze
            time_period: Time grouping period (default: "month")
            test_date_start: Optional start date for trend analysis
            test_date_end: Optional end date for trend analysis
            calculate_statistics: Whether to calculate statistical measures
            limit: Maximum number of trend periods to return
            security_context: Security context for access control

        Returns:
            Dict containing measurement trend analysis results
        """
        # PATIENT ID VALIDATION
        if patient_id is not None:
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Patient ID must be a non-empty string when provided")
            patient_id = patient_id.strip()

        # Check if observations collection exists
        collection_name = "observations"
        logger.warning(f"Observations collection '{collection_name}' does not exist yet")

        # OBSERVABILITY: Log the measurement trends analysis attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"MEASUREMENT TRENDS ANALYSIS ATTEMPTED (COLLECTION NOT FOUND):\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Test Name: {test_name}\n"
            f"  Observation Type: {observation_type}\n"
            f"  Time Period: {time_period}\n"
            f"  Date Range: {test_date_start} to {test_date_end}\n"
            f"  Calculate Statistics: {calculate_statistics}\n"
            f"  Limit: {limit}\n"
            f"{'=' * 70}"
        )

        return {
            "success": False,
            "collection": collection_name,
            "error": f"Observations collection '{collection_name}' does not exist in the database yet",
            "message": "Cannot analyze measurement trends until Observation data is ingested",
            "query_parameters": {
                "patient_id": patient_id,
                "test_name": test_name,
                "observation_type": observation_type,
                "time_period": time_period,
                "test_date_start": test_date_start,
                "test_date_end": test_date_end,
                "calculate_statistics": calculate_statistics,
                "limit": limit,
            },
        }
