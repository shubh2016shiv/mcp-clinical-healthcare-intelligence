"""Diagnostic reports tools for querying diagnostic test results.

This module provides tools for querying and analyzing diagnostic test reports
and diagnostic results from FHIR DiagnosticReport resources.

PATIENT ID DEPENDENCY: Supports optional patient_id filtering - when provided,
results are constrained to that patient's diagnostic reports. Recommended for
patient-specific diagnostic data access.
"""

import logging
from typing import Any

from ...base_tool import BaseTool
from ...models import (
    CollectionNames,
    DiagnosticReportRecord,
    DiagnosticReportRequest,
    DiagnosticReportResponse,
    SearchDiagnosticReportsRequest,
    SearchDiagnosticReportsResponse,
)
from ...utils import (
    build_compound_filter,
    build_date_filter,
    build_text_filter,
    handle_mongo_errors,
)

logger = logging.getLogger(__name__)


class DiagnosticReportsTools(BaseTool):
    """Tools for querying and analyzing diagnostic test reports.

    This class provides methods for retrieving diagnostic reports from the
    diagnosticreports collection. Supports both individual record queries
    and text-based search capabilities.

    Inherits optimized connection management from BaseTool.
    """

    def __init__(self):
        """Initialize diagnostic reports tools with optimized database connection."""
        super().__init__()

    @handle_mongo_errors
    async def get_diagnostic_reports(
        self,
        request: DiagnosticReportRequest,
        security_context: Any = None,
    ) -> DiagnosticReportResponse:
        """Retrieve diagnostic test reports for patients.

        This tool retrieves diagnostic report information, including test results,
        clinical summaries, and report types. Supports filtering by patient,
        report type, and date range.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        results are filtered to that patient's reports only. When omitted,
        returns diagnostic reports across all patients (population-level analysis).

        Args:
            request: DiagnosticReportRequest with filter parameters
            security_context: Security context for access control

        Returns:
            DiagnosticReportResponse with diagnostic report records and total count

        Raises:
            ValueError: If patient_id format is invalid when provided
        """
        db = self.get_database()
        collection = db[CollectionNames.DIAGNOSTIC_REPORTS.value]

        # Build base query filter
        filters = []

        # PATIENT ID VALIDATION: If patient_id is provided, filter to that patient's reports
        if request.patient_id:
            filters.append({"patient_id": request.patient_id})
            logger.info(f"Filtering diagnostic reports to patient: {request.patient_id}")

        if request.report_type:
            filters.append(build_text_filter("report_type", request.report_type))

        # Date range filter
        if request.report_date_start or request.report_date_end:
            date_filter = build_date_filter(
                "report_date", request.report_date_start, request.report_date_end
            )
            if date_filter:
                filters.append(date_filter)

        query_filter = build_compound_filter(*filters)

        logger.debug(f"Diagnostic reports query: {query_filter}")

        # Execute query directly with Motor (async-native)
        docs = (
            await collection.find(query_filter).limit(request.limit).to_list(length=request.limit)
        )

        # Convert documents to DiagnosticReportRecord objects
        reports = []
        for doc in docs:
            try:
                record = DiagnosticReportRecord(
                    patient_id=doc.get("patient_id"),
                    report_type=doc.get("report_type"),
                    report_date=doc.get("report_date"),
                    clinical_summary=doc.get("clinical_summary"),
                    source_fhir_id=doc.get("source_fhir_id"),
                )
                reports.append(record)
            except Exception as e:
                logger.warning(f"Failed to convert diagnostic report document: {e}")
                continue

        # OBSERVABILITY: Log the query execution
        logger.info(
            f"\n{'=' * 70}\n"
            f"DIAGNOSTIC REPORTS QUERY:\n"
            f"  Collection: {CollectionNames.DIAGNOSTIC_REPORTS.value}\n"
            f"  Patient ID: {request.patient_id}\n"
            f"  Report Type: {request.report_type}\n"
            f"  Date Range: {request.report_date_start} to {request.report_date_end}\n"
            f"  Limit: {request.limit}\n"
            f"  Results: {len(reports)}\n"
            f"{'=' * 70}"
        )

        return DiagnosticReportResponse(total_reports=len(reports), reports=reports)

    @handle_mongo_errors
    async def search_diagnostic_reports(
        self,
        request: SearchDiagnosticReportsRequest,
        security_context: Any = None,
    ) -> SearchDiagnosticReportsResponse:
        """Search diagnostic reports by clinical content and metadata.

        This tool provides text-based search capabilities for diagnostic reports,
        searching the clinical_summary field for relevant keywords and allowing
        filters by report type and date range.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        search results are limited to that patient's reports.

        Args:
            request: SearchDiagnosticReportsRequest with search and filter parameters
            security_context: Security context for access control

        Returns:
            SearchDiagnosticReportsResponse with matching diagnostic reports

        Raises:
            ValueError: If patient_id format is invalid when provided
        """
        db = self.get_database()
        collection = db[CollectionNames.DIAGNOSTIC_REPORTS.value]

        # Build base query filter
        filters = []

        # PATIENT ID VALIDATION: If patient_id is provided, filter to that patient's reports
        if request.patient_id:
            filters.append({"patient_id": request.patient_id})
            logger.info(f"Filtering diagnostic report search to patient: {request.patient_id}")

        # Text search on clinical summary
        if request.search_text:
            filters.append(build_text_filter("clinical_summary", request.search_text))

        if request.report_type:
            filters.append(build_text_filter("report_type", request.report_type))

        # Date range filter
        if request.report_date_start or request.report_date_end:
            date_filter = build_date_filter(
                "report_date", request.report_date_start, request.report_date_end
            )
            if date_filter:
                filters.append(date_filter)

        query_filter = build_compound_filter(*filters)

        logger.debug(f"Diagnostic reports search query: {query_filter}")

        # Execute query directly with Motor (async-native)
        docs = (
            await collection.find(query_filter).limit(request.limit).to_list(length=request.limit)
        )

        # Convert documents to DiagnosticReportRecord objects
        reports = []
        for doc in docs:
            try:
                record = DiagnosticReportRecord(
                    patient_id=doc.get("patient_id"),
                    report_type=doc.get("report_type"),
                    report_date=doc.get("report_date"),
                    clinical_summary=doc.get("clinical_summary"),
                    source_fhir_id=doc.get("source_fhir_id"),
                )
                reports.append(record)
            except Exception as e:
                logger.warning(f"Failed to convert diagnostic report document: {e}")
                continue

        # OBSERVABILITY: Log the search execution
        logger.info(
            f"\n{'=' * 70}\n"
            f"DIAGNOSTIC REPORTS SEARCH:\n"
            f"  Collection: {CollectionNames.DIAGNOSTIC_REPORTS.value}\n"
            f"  Search Text: {request.search_text}\n"
            f"  Patient ID: {request.patient_id}\n"
            f"  Report Type: {request.report_type}\n"
            f"  Date Range: {request.report_date_start} to {request.report_date_end}\n"
            f"  Limit: {request.limit}\n"
            f"  Results: {len(reports)}\n"
            f"{'=' * 70}"
        )

        return SearchDiagnosticReportsResponse(total_reports=len(reports), reports=reports)
