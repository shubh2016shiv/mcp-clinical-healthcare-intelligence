"""Patient medication history tools.

This module provides tools for retrieving and analyzing patient
medication history with optional drug classification enrichment.

PATIENT ID DEPENDENCY (REQUIRED): This tool operates on a single patient.
The patient_id field is mandatory and used to fetch all medications for that patient.
"""

# For Phase 1, we're importing from the original location
# In Phase 2, the actual implementation will be moved here
from ...medication_tools import MedicationTools

# Re-export for discovery
__all__ = ["MedicationTools"]
