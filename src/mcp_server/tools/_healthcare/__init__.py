"""Healthcare domain-specific MCP tools.

This package contains all healthcare business logic tools organized by
operational categories: demographics, clinical data, medications, financial, etc.

Usage:
    from src.mcp_server.tools._healthcare.clinical_data import AllergyTools, ConditionAnalyticsTools
    from src.mcp_server.tools._healthcare.demographics import search_patients
    from src.mcp_server.tools._healthcare.clinical_timeline import PatientTimelineTools
    from src.mcp_server.tools._healthcare.financial import FinancialAnalyticsTools
    from src.mcp_server.tools._healthcare.medications import (
        MedicationTools, DrugTools
    )
    from src.mcp_server.tools._healthcare.analytics_tools import AnalyticsTools  # Unified wrapper
"""

# Import all tool classes for convenience
from .analytics_tools import AnalyticsTools
from .clinical_data import (
    AllergyTools,
    CarePlansTools,
    ConditionsTools,
    ImmunizationsTools,
    ObservationsTools,
    ProceduresTools,
)
from .clinical_timeline import PatientTimelineTools
from .demographics import search_patients
from .encounters import EncountersTools
from .financial import AnalyticsTools as FinancialAnalyticsTools
from .financial import ClaimsTools, EOBTools
from .medications import DrugAnalysisTools, DrugTools, MedicationTools
from .patient_tools import PatientTools

__all__ = [
    "PatientTools",
    "AnalyticsTools",
    "search_patients",
    "PatientTimelineTools",
    "ConditionsTools",
    "AllergyTools",
    "CarePlansTools",
    "ImmunizationsTools",
    "ObservationsTools",
    "ProceduresTools",
    "EncountersTools",
    "ClaimsTools",
    "EOBTools",
    "FinancialAnalyticsTools",
    "MedicationTools",
    "DrugTools",
    "DrugAnalysisTools",
]
