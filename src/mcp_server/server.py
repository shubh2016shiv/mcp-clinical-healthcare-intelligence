"""Healthcare MCP Server using FastMCP.

This module implements a Model Context Protocol (MCP) server for healthcare data queries
using the FastMCP framework. The server provides tools for searching patients, analyzing
conditions, retrieving medication history, and querying drug information from MongoDB.

Key Features:
    - FastMCP-based server implementation
    - Five capability-based tools for healthcare data
    - Comprehensive error handling and logging
    - Pydantic-based request/response validation
    - Connection pooling and health checks

Architecture:
    - Patient Tools: search_patients, get_patient_clinical_timeline
    - Analytics Tools: analyze_conditions, get_financial_summary
    - Medication Tools: get_medication_history
    - Drug Tools: search_drugs, analyze_drug_classes

Usage:
    Run this server and connect MCP clients (Claude Desktop, etc.) to query healthcare data.
"""

import logging
from typing import Any

from fastmcp import FastMCP

from src.config.settings import settings

from .database.connection import get_connection_manager
from .security import initialize_security, require_auth
from .tool_prompts import get_system_instructions, get_tool_prompt

# Import from new modular structure (for compatibility and observability)
# These provide the enhanced query logging and validation features
from .tools.analytics_tools import AnalyticsTools
from .tools._healthcare.medications import DrugAnalysisTools
from .tools.medication_tools import MedicationTools
from .tools.models import (
    ClinicalTimelineRequest,
    ConditionAnalysisRequest,
    DrugClassAnalysisRequest,
    ErrorResponse,
    FinancialSummaryRequest,
    MedicationHistoryRequest,
    PatientSummary,
    SearchDrugsRequest,
    SearchPatientsRequest,
)
from .tools.patient_tools import PatientTools


def with_centralized_prompt(tool_name: str):
    """Decorator to set function docstring from centralized prompts."""

    def decorator(func):
        prompt = get_tool_prompt(tool_name)
        if prompt:
            func.__doc__ = prompt
        return func

    return decorator


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_server() -> FastMCP:
    """Create and configure the FastMCP server with all healthcare tools.

    Returns:
        Configured FastMCP server instance
    """
    # Initialize FastMCP server
    system_instructions = get_system_instructions()
    if not system_instructions:
        logger.warning("System instructions not found, using default instructions")
        system_instructions = (
            "You are a healthcare data analysis assistant. Use the available tools to query "
            "patient information, analyze medical conditions, review medication histories, "
            "and search drug information. Always provide context and explanations for your "
            "findings. Be mindful of patient privacy and data sensitivity."
        )

    server = FastMCP(name="healthcare-mcp-server", instructions=system_instructions)

    # Initialize database connection
    logger.info("Initializing database connection...")
    db_manager = get_connection_manager()
    db_manager.connect()

    # Initialize security layer
    if settings.security_enabled:
        logger.info("Initializing security layer...")
        initialize_security()
        logger.info("Security layer initialized successfully")
    else:
        logger.warning("Security layer is DISABLED - not recommended for production")

    # Initialize tool classes
    patient_tools = PatientTools()
    analytics_tools = AnalyticsTools()
    medication_tools = MedicationTools()
    drug_tools = DrugTools()

    # =============================================================================
    # PATIENT TOOLS REGISTRATION
    # =============================================================================

    @server.tool()
    @require_auth()
    @with_centralized_prompt("search_patients")
    async def search_patients(
        first_name: str = None,
        last_name: str = None,
        patient_id: str = None,
        city: str = None,
        state: str = None,
        birth_date_start: str = None,
        birth_date_end: str = None,
        gender: str = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        try:
            request = SearchPatientsRequest(
                first_name=first_name,
                last_name=last_name,
                patient_id=patient_id,
                city=city,
                state=state,
                birth_date_start=birth_date_start,
                birth_date_end=birth_date_end,
                gender=gender,
                limit=limit,
            )
            # Get security context (validated by @require_auth decorator)
            from .security.authentication import get_security_context

            security_context = get_security_context()

            result = await patient_tools.search_patients(request, security_context)

            # Handle ErrorResponse from decorator
            if isinstance(result, ErrorResponse):
                return result.model_dump()

            # Validate result is a list
            if not isinstance(result, list):
                logger.error(f"search_patients returned non-list type: {type(result)}")
                return ErrorResponse(
                    error="Invalid return type from search_patients",
                    details=f"Expected list, got {type(result).__name__}",
                    operation="search_patients",
                ).model_dump()

            # Convert list of PatientSummary to list of dicts with validation
            patients = []
            for patient in result:
                if not isinstance(patient, PatientSummary):
                    logger.warning(f"Skipping invalid patient type: {type(patient)}")
                    continue
                try:
                    patients.append(patient.model_dump())
                except Exception as e:
                    logger.error(f"Failed to serialize patient: {e}, type: {type(patient)}")
                    continue

            return {"patients": patients, "count": len(patients)}
        except Exception as e:
            logger.error(f"Error in search_patients: {e}")
            return ErrorResponse(
                error="Failed to search patients", details=str(e), operation="search_patients"
            ).model_dump()

    @server.tool()
    @require_auth()
    @with_centralized_prompt("get_patient_clinical_timeline")
    async def get_patient_clinical_timeline(
        patient_id: str,
        start_date: str = None,
        end_date: str = None,
        event_types: list = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        try:
            request = ClinicalTimelineRequest(
                patient_id=patient_id,
                start_date=start_date,
                end_date=end_date,
                event_types=event_types,
                limit=limit,
            )
            # Get security context (validated by @require_auth decorator)
            from .security.authentication import get_security_context

            security_context = get_security_context()

            result = await patient_tools.get_patient_clinical_timeline(request, security_context)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error in get_patient_clinical_timeline: {e}")
            return ErrorResponse(
                error="Failed to get patient clinical timeline",
                details=str(e),
                operation="get_patient_clinical_timeline",
            ).model_dump()

    # =============================================================================
    # ANALYTICS TOOLS REGISTRATION
    # =============================================================================

    @server.tool()
    @require_auth()  # Access determined by required collections
    @with_centralized_prompt("analyze_conditions")
    async def analyze_conditions(
        condition_name: str = None,
        status: str = None,
        onset_date_start: str = None,
        onset_date_end: str = None,
        group_by: str = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        try:
            request = ConditionAnalysisRequest(
                condition_name=condition_name,
                status=status,
                onset_date_start=onset_date_start,
                onset_date_end=onset_date_end,
                group_by=group_by,
                limit=limit,
            )
            # Get security context (validated by @require_auth decorator)
            from .security.authentication import get_security_context

            security_context = get_security_context()

            result = await analytics_tools.analyze_conditions(request, security_context)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error in analyze_conditions: {e}")
            return ErrorResponse(
                error="Failed to analyze conditions", details=str(e), operation="analyze_conditions"
            ).model_dump()

    @server.tool()
    @require_auth()  # Access determined by required collections
    @with_centralized_prompt("get_financial_summary")
    async def get_financial_summary(
        patient_id: str = None,
        start_date: str = None,
        end_date: str = None,
        insurance_provider: str = None,
        min_amount: float = None,
        max_amount: float = None,
        group_by: str = None,
    ) -> dict[str, Any]:
        try:
            request = FinancialSummaryRequest(
                patient_id=patient_id,
                start_date=start_date,
                end_date=end_date,
                insurance_provider=insurance_provider,
                min_amount=min_amount,
                max_amount=max_amount,
                group_by=group_by,
            )
            # Get security context (validated by @require_auth decorator)
            from .security.authentication import get_security_context

            security_context = get_security_context()

            result = await analytics_tools.get_financial_summary(request, security_context)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error in get_financial_summary: {e}")
            return ErrorResponse(
                error="Failed to get financial summary",
                details=str(e),
                operation="get_financial_summary",
            ).model_dump()

    # =============================================================================
    # MEDICATION TOOLS REGISTRATION
    # =============================================================================

    @server.tool()
    @require_auth()  # Access determined by required collections
    @with_centralized_prompt("get_medication_history")
    async def get_medication_history(
        patient_id: str = None,
        medication_name: str = None,
        drug_class: str = None,
        status: str = None,
        prescribed_date_start: str = None,
        include_drug_details: bool = True,
        limit: int = 50,
    ) -> dict[str, Any]:
        try:
            request = MedicationHistoryRequest(
                patient_id=patient_id,
                medication_name=medication_name,
                drug_class=drug_class,
                status=status,
                prescribed_date_start=prescribed_date_start,
                include_drug_details=include_drug_details,
                limit=limit,
            )
            # Get security context (validated by @require_auth decorator)
            from .security.authentication import get_security_context

            security_context = get_security_context()

            result = await medication_tools.get_medication_history(request, security_context)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error in get_medication_history: {e}")
            return ErrorResponse(
                error="Failed to get medication history",
                details=str(e),
                operation="get_medication_history",
            ).model_dump()

    # =============================================================================
    # DRUG TOOLS REGISTRATION
    # =============================================================================

    @server.tool()
    @require_auth()  # Access determined by required collections
    @with_centralized_prompt("search_drugs")
    async def search_drugs(
        drug_name: str = None,
        therapeutic_class: str = None,
        drug_class: str = None,
        drug_subclass: str = None,
        rxcui: str = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        try:
            request = SearchDrugsRequest(
                drug_name=drug_name,
                therapeutic_class=therapeutic_class,
                drug_class=drug_class,
                drug_subclass=drug_subclass,
                rxcui=rxcui,
                limit=limit,
            )
            # Drug information is public - no security context needed
            result = await drug_tools.search_drugs(request)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error in search_drugs: {e}")
            return ErrorResponse(
                error="Failed to search drugs", details=str(e), operation="search_drugs"
            ).model_dump()

    @server.tool()
    @require_auth()  # Access determined by required collections
    @with_centralized_prompt("analyze_drug_classes")
    async def analyze_drug_classes(
        group_by: str, min_count: int = 1, limit: int = 50
    ) -> dict[str, Any]:
        try:
            request = DrugClassAnalysisRequest(group_by=group_by, min_count=min_count, limit=limit)
            # Drug information is public - no security context needed
            result = await drug_tools.analyze_drug_classes(request)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error in analyze_drug_classes: {e}")
            return ErrorResponse(
                error="Failed to analyze drug classes",
                details=str(e),
                operation="analyze_drug_classes",
            ).model_dump()

    return server


def main():
    """Main entry point for the MCP server."""
    try:
        logger.info("Starting Healthcare MCP Server...")

        # Create and run the server
        server = create_server()

        logger.info("Healthcare MCP Server initialized successfully")
        logger.info(
            "Available tools: search_patients, get_patient_clinical_timeline, analyze_conditions, get_financial_summary, get_medication_history, search_drugs, analyze_drug_classes"
        )

        # Run the server
        server.run()

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    main()
