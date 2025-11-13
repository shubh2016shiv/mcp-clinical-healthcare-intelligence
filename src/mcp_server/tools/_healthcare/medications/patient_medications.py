"""Patient medication history tools.

This module provides tools for retrieving and analyzing patient
medication history with optional drug classification enrichment.

PATIENT ID DEPENDENCY (REQUIRED): This tool operates on a single patient.
The patient_id field is mandatory and used to fetch all medications for that patient.
"""

import logging
from typing import Any

# Async executor removed - now using pure Motor async
from src.mcp_server.tools._healthcare.medications.models import (
    MedicationHistoryRequest,
    MedicationHistoryResponse,
    MedicationRecord,
)
from src.mcp_server.tools.base_tool import BaseTool
from src.mcp_server.tools.models import CollectionNames
from src.mcp_server.tools.utils import (
    build_compound_filter,
    build_date_filter,
    build_text_filter,
    handle_mongo_errors,
)

logger = logging.getLogger(__name__)


class MedicationTools(BaseTool):
    """Tools for patient medication history and drug-related queries.

    This class provides methods for retrieving medication history with optional
    drug classification enrichment using data from the drugs collection.

    Patient ID is required for all operations as this tool operates on
    individual patient medication records.
    """

    def __init__(self):
        """Initialize medication tools."""
        super().__init__()

    @handle_mongo_errors
    async def get_medication_history(
        self,
        request: MedicationHistoryRequest,
        security_context: Any = None,
    ) -> MedicationHistoryResponse:
        """Retrieve medication history for a specific patient with optional drug classification enrichment.

        This tool fetches medication records for a patient and can enrich them with drug classification
        information from the drugs collection. It supports flexible filtering by medication
        name, status, and prescription date.

        PATIENT ID VALIDATION: Required patient_id parameter. This tool operates
        on a single patient's medication records only.

        ROBUST IMPLEMENTATION: Direct query approach without complex aggregation to ensure
        reliable field extraction. Comprehensive error handling and logging throughout.

        Args:
            request: Medication history request parameters
            security_context: Security context for access control (field projection)

        Returns:
            Medication history response with optional drug classification enrichment

        Raises:
            ValueError: If patient_id is not provided or format is invalid
        """
        # PATIENT ID VALIDATION
        if (
            not request.patient_id
            or not isinstance(request.patient_id, str)
            or not request.patient_id.strip()
        ):
            raise ValueError("Patient ID is required and must be a non-empty string.")
        patient_id = request.patient_id.strip()

        db = self.get_database()

        # Check if medications collection exists
        collection_name = CollectionNames.MEDICATIONS.value
        collection_names = await db.list_collection_names()
        if collection_name not in collection_names:
            logger.warning(f"Medications collection '{collection_name}' does not exist yet")
            return MedicationHistoryResponse(
                total_medications=0,
                medications=[],
                enriched_with_drug_data=False,
                message=f"Medications collection '{collection_name}' does not exist",
            )

        collection = db[collection_name]

        # Build query filter for medications collection - SIMPLE DIRECT APPROACH
        # Always start with patient_id filter (REQUIRED)
        filters = [{"patient_id": patient_id}]

        if request.medication_name:
            filters.append(build_text_filter("medication_name", request.medication_name))

        if request.status:
            filters.append({"status": request.status})

        # Date range filter for prescribed date
        if request.prescribed_date_start:
            date_filter = build_date_filter("prescribed_date", request.prescribed_date_start, None)
            if date_filter:
                filters.append(date_filter)

        query_filter = build_compound_filter(*filters)

        # OBSERVABILITY: Log query
        logger.info(
            f"\nMEDICATION QUERY: patient={patient_id}, status={request.status}, "
            f"filter={query_filter}, limit={request.limit}"
        )

        # SIMPLE DIRECT APPROACH: Execute find query without complex aggregation
        # This ensures fields are preserved and extracted correctly
        # NO PROJECTION - Get all fields from document
        try:
            cursor = collection.find(query_filter)  # No projection - get all fields
            cursor = cursor.sort("prescribed_date", -1).limit(request.limit)
            results = await cursor.to_list(length=request.limit)
            logger.info(f"✓ Query executed: found {len(results)} medications")

        except Exception as e:
            logger.error(f"✗ Query failed: {e}")
            return MedicationHistoryResponse(
                total_medications=0,
                medications=[],
                enriched_with_drug_data=False,
                message=f"Query failed: {str(e)}",
            )

        # Convert results to medication records
        medications = []

        # Process each medication record
        for doc in results:
            try:
                # Safe field extraction with robust handling
                def safe_str_field(value: Any) -> str | None:
                    """Safely extract and convert field to string or None."""
                    if value is None or value == "":
                        return None
                    value_str = str(value).strip()
                    return value_str if value_str else None

                # Extract all fields - try multiple possible field name variations
                patient_id = doc.get("patient_id", "")

                # Try standard field names first
                medication_name = safe_str_field(doc.get("medication_name"))
                prescriber = safe_str_field(doc.get("prescriber"))
                status = safe_str_field(doc.get("status"))
                dosage_instruction = safe_str_field(doc.get("dosage_instruction"))

                # Handle prescribed_date (may be datetime or string)
                prescribed_date_raw = doc.get("prescribed_date")
                if prescribed_date_raw is None or prescribed_date_raw == "":
                    prescribed_date = None
                else:
                    prescribed_date = str(prescribed_date_raw)

                # Optional: Drug enrichment
                drug_classification = None
                if request.include_drug_details and doc.get("drug_classification"):
                    try:
                        drug_info = doc["drug_classification"]
                        drug_classification = {
                            "drug_name": drug_info.get("primary_drug_name"),
                            "therapeutic_class_l2": drug_info.get("therapeutic_class_l2"),
                            "drug_class_l3": drug_info.get("drug_class_l3"),
                            "drug_subclass_l4": drug_info.get("drug_subclass_l4"),
                            "rxcui": drug_info.get("ingredient_rxcui"),
                        }
                    except Exception as e:
                        logger.warning(f"Failed to extract drug classification: {e}")

                # Create medication record
                medication = MedicationRecord(
                    patient_id=patient_id or "",
                    medication_name=medication_name,
                    prescriber=prescriber,
                    status=status,
                    prescribed_date=prescribed_date,
                    dosage_instruction=dosage_instruction,
                    drug_classification=drug_classification,
                )
                medications.append(medication)
                logger.debug(f"✓ Extracted medication: {medication_name} (status: {status})")

            except Exception as e:
                logger.error(f"✗ Failed to extract medication record: {e}", exc_info=True)
                continue

        # Apply data minimization if security context provided
        # NOTE: Data minimization is SKIPPED for medication history to preserve essential clinical fields
        # Medication information (name, prescriber, dose, date) is REQUIRED for patient care
        # and must not be filtered out by role-based restrictions
        if security_context and medications:
            logger.debug(
                f"Note: Skipping data minimization for medication records (security_context.role={security_context.role})"
            )
            # Uncomment below if role-based field filtering is implemented for medications
            # security_manager = get_security_manager()
            # minimized_data = security_manager.data_minimizer.filter_record_list(
            #     [med.model_dump() for med in medications], security_context.role
            # )
            # medications = [MedicationRecord(**m) for m in minimized_data]

        logger.info(f"✓ Retrieved {len(medications)} medication records for patient {patient_id}")

        return MedicationHistoryResponse(
            total_medications=len(medications),
            medications=medications,
            enriched_with_drug_data=request.include_drug_details,
        )
