"""Demographics tools for patient identification and lookup.

This package provides tools for searching and identifying patients
based on demographic information.
"""

from .patient_search import (
    PatientSummary,
    SecurityContext,
    load_projections_from_config,
    search_patients,
)

__all__ = ["search_patients", "PatientSummary", "SecurityContext", "load_projections_from_config"]
