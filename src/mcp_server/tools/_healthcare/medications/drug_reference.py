"""Drug reference database tools.

This module provides tools for searching drug information from
RxNorm database including drug names, ATC classifications, and RxCUI codes.

This tool is population-level (not patient-specific) - it searches the
drug reference database, not patient medication records.
"""

# For Phase 1, we're importing from the original location
# In Phase 2, the actual implementation will be moved here
from ...drug_tools import DrugTools

# Re-export for discovery
__all__ = ["DrugTools"]
