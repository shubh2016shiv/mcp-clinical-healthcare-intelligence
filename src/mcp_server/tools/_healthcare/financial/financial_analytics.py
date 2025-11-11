"""Financial analytics and summary tools.

This module provides tools for analyzing financial data across
claims and billing, including totals, averages, and distributions.

PATIENT ID VALIDATION: Supports optional patient_id filtering - when provided,
results are limited to that patient's claims. Recommended for patient-specific financial analysis.
"""

# For Phase 1, we're importing from the original location
# In Phase 2, the actual implementation will be moved here
from ...analytics_tools import AnalyticsTools

# Re-export for discovery
__all__ = ["AnalyticsTools"]
