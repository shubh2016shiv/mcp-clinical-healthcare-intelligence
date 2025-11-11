"""Condition analysis tools for health conditions and diagnoses.

This module provides tools for querying and analyzing health conditions,
including individual condition records and population-level analysis.

PATIENT ID VALIDATION: Supports optional patient_id filtering - when provided,
results are constrained to that patient. Recommend using when analyzing patient-specific data.
"""

# For Phase 1, we're importing from the original location
# In Phase 2, the actual implementation will be moved here
from ...analytics_tools import AnalyticsTools

# Re-export for discovery
__all__ = ["AnalyticsTools"]
