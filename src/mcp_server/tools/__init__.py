"""Healthcare MCP Tools Package.

This package contains all the MCP tools for querying healthcare data from MongoDB.
The tools are organized by functionality and theme for better maintainability.

Available Tool Classes:
    - PatientTools: Patient search and clinical timeline tools
    - AnalyticsTools: Population-level condition and financial analysis
    - MedicationTools: Medication history and drug classification tools
    - DrugTools: Drug database search and classification analysis

All tools use shared utilities and follow consistent error handling patterns.
"""

from ._healthcare.analytics_tools import AnalyticsTools
from ._healthcare.medications import DrugAnalysisTools as DrugTools
from ._healthcare.medications import MedicationTools
from ._healthcare.patient_tools import PatientTools

__all__ = ["PatientTools", "AnalyticsTools", "MedicationTools", "DrugTools"]
