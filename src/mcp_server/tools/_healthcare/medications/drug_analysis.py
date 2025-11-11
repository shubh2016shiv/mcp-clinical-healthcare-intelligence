"""Drug classification analysis tools.

This module provides tools for analyzing drug classifications and
their distributions across therapeutic categories.

This tool is population-level (not patient-specific) - it performs
statistical analysis on drug reference data.
"""

# For Phase 1, we're importing from the original location
# In Phase 2, the actual implementation will be moved here
from ...drug_tools import DrugTools

# Re-export for discovery
__all__ = ["DrugTools"]
