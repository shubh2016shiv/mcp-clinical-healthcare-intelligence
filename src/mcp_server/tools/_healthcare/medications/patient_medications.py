"""Patient medication history tools.

This module provides tools for retrieving and analyzing patient
medication history with optional drug classification enrichment.

PATIENT ID DEPENDENCY (REQUIRED): This tool operates on a single patient.
The patient_id field is mandatory and used to fetch all medications for that patient.
"""

import asyncio
import logging
from typing import Any

from ....base_tool import BaseTool
from ....database.async_executor import get_executor_pool
from ....models import CollectionNames
from ....security import get_security_manager
from ....utils import (
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
        patient_id: str,
        medication_name: str | None = None,
        status: str | None = None,
        prescribed_date_start: str | None = None,
        prescribed_date_end: str | None = None,
        include_drug_details: bool = True,
        limit: int = 50,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Retrieve medication history for a specific patient with optional drug classification enrichment.

        This tool fetches medication records for a patient and can enrich them with drug classification
        information from the drugs collection. It supports flexible filtering by medication
        name, status, and prescription date.

        PATIENT ID VALIDATION: Required patient_id parameter. This tool operates
        on a single patient's medication records only.

        Args:
            patient_id: Patient ID to retrieve medication history for (required)
            medication_name: Optional medication name filter (partial match)
            status: Optional medication status filter (active, completed, etc.)
            prescribed_date_start: Optional start date for prescription date range
            prescribed_date_end: Optional end date for prescription date range
            include_drug_details: Whether to enrich with drug classification data
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing medication history with optional drug classification enrichment

        Raises:
            ValueError: If patient_id is not provided or format is invalid
        """
        # PATIENT ID VALIDATION
        if not patient_id or not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("Patient ID is required and must be a non-empty string.")
        patient_id = patient_id.strip()

        db = self.get_database()

        # Check if medications collection exists
        collection_name = CollectionNames.MEDICATIONS.value
        if collection_name not in db.list_collection_names():
            logger.warning(f"Medications collection '{collection_name}' does not exist yet")

            # OBSERVABILITY: Log the query attempt for verification
            logger.info(
                f"\n{'='*70}\n"
                f"MEDICATION HISTORY QUERY ATTEMPTED (COLLECTION NOT FOUND):\n"
                f"  Collection: {collection_name}\n"
                f"  Patient ID: {patient_id}\n"
                f"  Medication Name: {medication_name}\n"
                f"  Status: {status}\n"
                f"  Date Range: {prescribed_date_start} to {prescribed_date_end}\n"
                f"  Include Drug Details: {include_drug_details}\n"
                f"  Limit: {limit}\n"
                f"{'='*70}"
            )

            return {
                "success": False,
                "collection": collection_name,
                "error": f"Medications collection '{collection_name}' does not exist in the database yet",
                "message": "Medication FHIR resources have not been ingested yet",
                "query_parameters": {
                    "patient_id": patient_id,
                    "medication_name": medication_name,
                    "status": status,
                    "prescribed_date_start": prescribed_date_start,
                    "prescribed_date_end": prescribed_date_end,
                    "include_drug_details": include_drug_details,
                    "limit": limit,
                },
            }

        collection = db[collection_name]

        # Build query filter for medications collection
        filters = [{"patient_id": patient_id}]  # Always filter by patient_id

        if medication_name:
            filters.append(build_text_filter("medication_name", medication_name))

        if status:
            filters.append({"status": status})

        # Date range filter for prescribed date
        if prescribed_date_start or prescribed_date_end:
            date_filter = build_date_filter(
                "prescribed_date", prescribed_date_start, prescribed_date_end
            )
            if date_filter:
                filters.append(date_filter)

        query_filter = build_compound_filter(*filters)

        # OBSERVABILITY: Log the medication history query before execution
        logger.info(
            f"\n{'='*70}\n"
            f"EXECUTING MEDICATION HISTORY QUERY:\n"
            f"  Collection: {collection_name}\n"
            f"  Patient ID: {patient_id}\n"
            f"  Medication Name: {medication_name}\n"
            f"  Status: {status}\n"
            f"  Date Range: {prescribed_date_start} to {prescribed_date_end}\n"
            f"  Include Drug Details: {include_drug_details}\n"
            f"  Filter: {query_filter}\n"
            f"  Limit: {limit}\n"
            f"{'='*70}"
        )

        # Execute blocking query in thread pool
        loop = asyncio.get_event_loop()
        executor = get_executor_pool().get_executor()

        if include_drug_details:
            # Use aggregation pipeline to join with drugs collection
            pipeline = [
                {"$match": query_filter},
                {
                    "$lookup": {
                        "from": CollectionNames.DRUGS.value,
                        "let": {"med_name": "$medication_name"},
                        "pipeline": [
                            {
                                "$match": {
                                    "$expr": {
                                        "$or": [
                                            # Exact match on primary drug name
                                            {"$eq": ["$primary_drug_name", "$$med_name"]},
                                            # Partial match (case-insensitive)
                                            {
                                                "$regexMatch": {
                                                    "input": "$$med_name",
                                                    "regex": "$primary_drug_name",
                                                    "options": "i"
                                                }
                                            }
                                        ]
                                    }
                                }
                            },
                            # Limit to top match for performance
                            {"$limit": 1}
                        ],
                        "as": "drug_info"
                    }
                },
                {
                    "$addFields": {
                        "drug_classification": {
                            "$cond": {
                                "if": {"$gt": [{"$size": "$drug_info"}, 0]},
                                "then": {"$arrayElemAt": ["$drug_info", 0]},
                                "else": None
                            }
                        }
                    }
                },
                {
                    "$project": {
                        "drug_info": 0  # Remove temporary field
                    }
                },
                {"$sort": {"prescribed_date": -1}},  # Most recent first
                {"$limit": limit}
            ]

            # Execute aggregation in thread pool (blocking I/O)
            results = await loop.run_in_executor(
                executor,
                lambda: list(collection.aggregate(pipeline))
            )

        else:
            # Simple query without enrichment
            def fetch_medications():
                cursor = collection.find(query_filter)
                cursor = cursor.sort("prescribed_date", -1).limit(limit)
                return list(cursor)

            results = await loop.run_in_executor(executor, fetch_medications)

        # Convert results to medication records
        medications = []
        enriched_count = 0

        for doc in results:
            # Extract drug classification if available
            drug_classification = None
            if include_drug_details and doc.get("drug_classification"):
                drug_info = doc["drug_classification"]
                drug_classification = {
                    "drug_name": drug_info.get("primary_drug_name"),
                    "therapeutic_class_l2": drug_info.get("therapeutic_class_l2"),
                    "drug_class_l3": drug_info.get("drug_class_l3"),
                    "drug_subclass_l4": drug_info.get("drug_subclass_l4"),
                    "rxcui": drug_info.get("ingredient_rxcui"),
                }
                enriched_count += 1

            # Create medication record
            medication = {
                "medication_id": str(doc.get("_id", "")),
                "patient_id": doc.get("patient_id", ""),
                "medication_name": doc.get("medication_name", ""),
                "prescriber": doc.get("prescriber", ""),
                "status": doc.get("status", ""),
                "prescribed_date": doc.get("prescribed_date", ""),
                "dosage_instruction": doc.get("dosage_instruction", ""),
                "drug_classification": drug_classification,
            }
            medications.append(medication)

        # Apply data minimization if security context provided
        if security_context and medications:
            security_manager = get_security_manager()
            minimized_data = security_manager.data_minimizer.filter_record_list(
                medications, security_context.role
            )
            medications = minimized_data

        logger.info(f"✓ Retrieved {len(medications)} medication records for patient {patient_id}")
        if include_drug_details:
            logger.info(f"✓ Enriched {enriched_count} medications with drug classification data")

        return {
            "success": True,
            "collection": collection_name,
            "query_type": "medication_history",
            "patient_id": patient_id,
            "results": medications,
            "count": len(medications),
            "limit_applied": limit,
            "enriched_with_drug_data": include_drug_details,
            "enrichment_count": enriched_count if include_drug_details else 0,
            "query_filter": query_filter,
        }
