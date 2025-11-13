"""Drug classification analysis tools.

This module provides tools for analyzing drug classifications and
their distributions across therapeutic categories from RxNorm drug data.

This tool is population-level (not patient-specific) - it performs
statistical analysis on drug reference data from FHIR Medication resources.

PATIENT ID DEPENDENCY: Not applicable - this tool operates on drug reference data,
not patient-specific medication records.
"""

import logging

# Async executor removed - now using pure Motor async
from src.mcp_server.tools._healthcare.medications.models import (
    DrugClassAnalysisRequest,
    DrugClassAnalysisResponse,
    DrugClassGroup,
    DrugRecord,
    DrugSearchResponse,
    SearchDrugsRequest,
)
from src.mcp_server.tools.base_tool import BaseTool
from src.mcp_server.tools.models import CollectionNames
from src.mcp_server.tools.utils import build_compound_filter, build_text_filter, handle_mongo_errors

logger = logging.getLogger(__name__)


class DrugAnalysisTools(BaseTool):
    """Tools for drug and pharmacological data queries and analysis.

    This class provides methods for searching drugs and analyzing
    drug classifications using the RxNorm drugs collection.

    This tool operates on drug reference data, not patient-specific records.
    """

    def __init__(self):
        """Initialize drug analysis tools."""
        super().__init__()

    @handle_mongo_errors
    async def search_drugs(self, request: SearchDrugsRequest) -> DrugSearchResponse:
        """Search for drugs in the drugs collection with flexible criteria.

        This tool supports searching drugs by various criteria including name,
        ATC classification levels, and RxCUI. All text searches are case-insensitive
        partial matches.

        POPULATION-LEVEL ANALYSIS: This tool searches drug reference data,
        not patient-specific medication records.

        Args:
            request: Drug search request parameters

        Returns:
            Drug search response containing matching drug records
        """
        db = self.get_database()

        # Check if drugs collection exists
        collection_name = CollectionNames.DRUGS.value
        collection_names = await db.list_collection_names()
        if collection_name not in collection_names:
            logger.warning(f"Drugs collection '{collection_name}' does not exist yet")

            # OBSERVABILITY: Log the search attempt for verification
            logger.info(
                f"\n{'=' * 70}\n"
                f"DRUG SEARCH ATTEMPTED (COLLECTION NOT FOUND):\n"
                f"  Collection: {collection_name}\n"
                f"  Drug Name: {request.drug_name}\n"
                f"  Therapeutic Class: {request.therapeutic_class}\n"
                f"  Drug Class: {request.drug_class}\n"
                f"  Drug Subclass: {request.drug_subclass}\n"
                f"  RxCUI: {request.rxcui}\n"
                f"  Limit: {request.limit}\n"
                f"{'=' * 70}"
            )

            return DrugSearchResponse(total_drugs=0, drugs=[])

        collection = db[collection_name]

        # Build search filters dynamically
        filters = []

        # Drug name search (case-insensitive partial match)
        if request.drug_name:
            filters.append(build_text_filter("primary_drug_name", request.drug_name))

        # Classification filters (exact matches)
        if request.therapeutic_class:
            filters.append(build_text_filter("therapeutic_class_l2", request.therapeutic_class))

        if request.drug_class:
            filters.append(build_text_filter("drug_class_l3", request.drug_class))

        if request.drug_subclass:
            filters.append(build_text_filter("drug_subclass_l4", request.drug_subclass))

        # Exact RxCUI match
        if request.rxcui:
            filters.append({"ingredient_rxcui": request.rxcui})

        # Combine all filters
        query_filter = build_compound_filter(*filters)

        # OBSERVABILITY: Log the search query before execution
        logger.info(
            f"\n{'=' * 70}\n"
            f"EXECUTING DRUG SEARCH:\n"
            f"  Collection: {collection_name}\n"
            f"  Drug Name: {request.drug_name}\n"
            f"  Therapeutic Class: {request.therapeutic_class}\n"
            f"  Drug Class: {request.drug_class}\n"
            f"  Drug Subclass: {request.drug_subclass}\n"
            f"  RxCUI: {request.rxcui}\n"
            f"  Filter: {query_filter}\n"
            f"  Limit: {request.limit}\n"
            f"{'=' * 70}"
        )

        # Execute directly with Motor (async-native)
        docs = (
            await collection.find(query_filter).limit(request.limit).to_list(length=request.limit)
        )

        # Convert to drug records
        drugs = []
        for doc in docs:
            drug = DrugRecord(
                ingredient_rxcui=doc.get("ingredient_rxcui", ""),
                primary_drug_name=doc.get("primary_drug_name", ""),
                therapeutic_class_l2=doc.get("therapeutic_class_l2", ""),
                drug_class_l3=doc.get("drug_class_l3", ""),
                drug_subclass_l4=doc.get("drug_subclass_l4", ""),
                ingestion_metadata=doc.get("ingestion_metadata", {}),
            )
            drugs.append(drug)

        logger.info(f"✓ Found {len(drugs)} drugs matching search criteria")

        return DrugSearchResponse(total_drugs=len(drugs), drugs=drugs)

    @handle_mongo_errors
    async def analyze_drug_classes(
        self, request: DrugClassAnalysisRequest
    ) -> DrugClassAnalysisResponse:
        """Analyze drug classifications and their distributions.

        This tool performs aggregation analysis on drug classifications,
        providing insights into the distribution of drugs across therapeutic,
        pharmacological, and chemical categories.

        POPULATION-LEVEL ANALYSIS: This tool analyzes drug reference data
        to understand classification distributions across the entire drug database.

        Args:
            request: Drug class analysis request parameters

        Returns:
            Drug classification analysis response
        """
        db = self.get_database()

        # Check if drugs collection exists
        collection_name = CollectionNames.DRUGS.value
        collection_names = await db.list_collection_names()
        if collection_name not in collection_names:
            logger.warning(f"Drugs collection '{collection_name}' does not exist yet")

            return DrugClassAnalysisResponse(
                analysis_type=f"drug_classes_grouped_by_{request.group_by}",
                total_classes=0,
                classes=[],
            )

        collection = db[collection_name]

        # Determine which field to group by
        group_field_map = {
            "therapeutic_class": "therapeutic_class_l2",
            "drug_class": "drug_class_l3",
            "drug_subclass": "drug_subclass_l4",
        }

        if request.group_by not in group_field_map:
            raise ValueError(
                f"Invalid group_by value: {request.group_by}. Must be one of: {list(group_field_map.keys())}"
            )

        group_field = group_field_map[request.group_by]

        # Build aggregation pipeline
        pipeline = [
            # Match only documents that have the classification field
            {"$match": {group_field: {"$ne": None}}},
            # Group by the classification field
            {
                "$group": {
                    "_id": f"${group_field}",
                    "drug_count": {"$sum": 1},
                    "drug_names": {"$push": "$primary_drug_name"},
                }
            },
            # Filter by minimum count
            {"$match": {"drug_count": {"$gte": request.min_count}}},
            # Sort by drug count descending
            {"$sort": {"drug_count": -1}},
            # Limit results
            {"$limit": request.limit},
            # Project final results with sample drugs
            {
                "$project": {
                    "class_name": "$_id",
                    "drug_count": 1,
                    "example_drugs": {"$slice": ["$drug_names", 5]},  # First 5 examples
                    "_id": 0,
                }
            },
        ]

        # OBSERVABILITY: Log the analysis query
        logger.info(
            f"\n{'=' * 70}\n"
            f"DRUG CLASS ANALYSIS:\n"
            f"  Collection: {collection_name}\n"
            f"  Group By: {request.group_by}\n"
            f"  Field: {group_field}\n"
            f"  Min Count: {request.min_count}\n"
            f"  Limit: {request.limit}\n"
            f"  Pipeline Stages: {len(pipeline)}\n"
            f"{'=' * 70}"
        )

        # Execute aggregation directly with Motor (async-native)
        results = await collection.aggregate(pipeline).to_list(length=request.limit)

        # Convert to Pydantic models
        classes = [DrugClassGroup(**res) for res in results]

        logger.info(f"✓ Completed drug class analysis: {len(classes)} classification groups")

        return DrugClassAnalysisResponse(
            analysis_type=f"drug_classes_grouped_by_{request.group_by}",
            total_classes=len(classes),
            classes=classes,
        )
