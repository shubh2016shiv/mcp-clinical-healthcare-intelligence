"""Healthcare domain-specific MCP tools.

This package contains all healthcare business logic tools organized by
operational categories: demographics, clinical data, medications, financial, etc.

Usage:
    from src.mcp_server.tools._healthcare.clinical_data import AllergyTools
    from src.mcp_server.tools._healthcare.demographics import PatientSearchTools
    from src.mcp_server.tools._healthcare.clinical_timeline import PatientTimelineTools
    from src.mcp_server.tools._healthcare.clinical_data import AnalyticsTools
    from src.mcp_server.tools._healthcare.financial import AnalyticsTools
    from src.mcp_server.tools._healthcare.medications import (
        MedicationTools, DrugTools
    )
"""

# Import all tool classes for convenience
from .clinical_data import AllergyTools, CarePlansTools, ConditionsTools, ImmunizationsTools
from .clinical_timeline import PatientTimelineTools
from .demographics import PatientSearchTools
from .financial import AnalyticsTools as FinancialAnalyticsTools
from .medications import DrugTools, MedicationTools

__all__ = [
    "PatientSearchTools",
    "PatientTimelineTools",
    "ConditionsTools",
    "AllergyTools",
    "CarePlansTools",
    "ImmunizationsTools",
    "FinancialAnalyticsTools",
    "MedicationTools",
    "DrugTools",
]
