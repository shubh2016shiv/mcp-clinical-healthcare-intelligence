"""Drug classification analysis tools.

This module provides tools for analyzing drug classifications and
their distributions across therapeutic categories from RxNorm drug data.

This tool is population-level (not patient-specific) - it performs
statistical analysis on drug reference data from FHIR Medication resources.

PATIENT ID DEPENDENCY: Not applicable - this tool operates on drug reference data,
not patient-specific medication records.
"""

import asyncio
import logging
from typing import Any, Dict, List

from ....base_tool import BaseTool
from ....database.async_executor import get_executor_pool
from ....models import CollectionNames
from ....utils import build_compound_filter, build_text_filter, handle_mongo_errors

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
    async def search_drugs(
        self,
        drug_name: str | None = None,
        therapeutic_class: str | None = None,
        drug_class: str | None = None,
        drug_subclass: str | None = None,
        rxcui: str | None = None,
        limit: int = 50,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Search for drugs in the drugs collection with flexible criteria.

        This tool supports searching drugs by various criteria including name,
        ATC classification levels, and RxCUI. All text searches are case-insensitive
        partial matches.

        POPULATION-LEVEL ANALYSIS: This tool searches drug reference data,
        not patient-specific medication records.

        Args:
            drug_name: Optional drug name (partial match)
            therapeutic_class: Optional therapeutic class (L2 level)
            drug_class: Optional drug class (L3 level)
            drug_subclass: Optional drug subclass (L4 level)
            rxcui: Optional RxCUI for exact drug identification
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing drug search results with proper observability logging
        """
        db = self.get_database()

        # Check if drugs collection exists
        collection_name = CollectionNames.DRUGS.value
        if collection_name not in db.list_collection_names():
            logger.warning(f"Drugs collection '{collection_name}' does not exist yet")

            # OBSERVABILITY: Log the search attempt for verification
            logger.info(
                f"\n{'='*70}\n"
                f"DRUG SEARCH ATTEMPTED (COLLECTION NOT FOUND):\n"
                f"  Collection: {collection_name}\n"
                f"  Drug Name: {drug_name}\n"
                f"  Therapeutic Class: {therapeutic_class}\n"
                f"  Drug Class: {drug_class}\n"
                f"  Drug Subclass: {drug_subclass}\n"
                f"  RxCUI: {rxcui}\n"
                f"  Limit: {limit}\n"
                f"{'='*70}"
            )

            return {
                "success": False,
                "collection": collection_name,
                "error": f"Drugs collection '{collection_name}' does not exist in the database yet",
                "message": "Drug reference FHIR resources have not been ingested yet",
                "query_parameters": {
                    "drug_name": drug_name,
                    "therapeutic_class": therapeutic_class,
                    "drug_class": drug_class,
                    "drug_subclass": drug_subclass,
                    "rxcui": rxcui,
                    "limit": limit,
                },
            }

        collection = db[collection_name]

        # Build search filters dynamically
        filters = []

        # Drug name search (case-insensitive partial match)
        if drug_name:
            filters.append(build_text_filter("primary_drug_name", drug_name))

        # Classification filters (exact matches)
        if therapeutic_class:
            filters.append(build_text_filter("therapeutic_class_l2", therapeutic_class))

        if drug_class:
            filters.append(build_text_filter("drug_class_l3", drug_class))

        if drug_subclass:
            filters.append(build_text_filter("drug_subclass_l4", drug_subclass))

        # Exact RxCUI match
        if rxcui:
            filters.append({"ingredient_rxcui": rxcui})

        # Combine all filters
        query_filter = build_compound_filter(*filters)

        # OBSERVABILITY: Log the search query before execution
        logger.info(
            f"\n{'='*70}\n"
            f"EXECUTING DRUG SEARCH:\n"
            f"  Collection: {collection_name}\n"
            f"  Drug Name: {drug_name}\n"
            f"  Therapeutic Class: {therapeutic_class}\n"
            f"  Drug Class: {drug_class}\n"
            f"  Drug Subclass: {drug_subclass}\n"
            f"  RxCUI: {rxcui}\n"
            f"  Filter: {query_filter}\n"
            f"  Limit: {limit}\n"
            f"{'='*70}"
        )

        # Execute blocking query in thread pool
        loop = asyncio.get_event_loop()
        executor = get_executor_pool().get_executor()

        # Run find and limit operations in executor
        cursor = await loop.run_in_executor(
            executor,
            lambda: collection.find(query_filter).limit(limit)
        )

        # Convert cursor to list in executor (blocking I/O)
        docs = await loop.run_in_executor(executor, list, cursor)

        # Convert to drug records
        drugs = []
        for doc in docs:
            drug = {
                "ingredient_rxcui": doc.get("ingredient_rxcui", ""),
                "primary_drug_name": doc.get("primary_drug_name", ""),
                "therapeutic_class_l2": doc.get("therapeutic_class_l2", ""),
                "drug_class_l3": doc.get("drug_class_l3", ""),
                "drug_subclass_l4": doc.get("drug_subclass_l4", ""),
                "ingestion_metadata": doc.get("ingestion_metadata", {}),
            }
            drugs.append(drug)

        logger.info(f"✓ Found {len(drugs)} drugs matching search criteria")

        return {
            "success": True,
            "collection": collection_name,
            "query_type": "drug_search",
            "results": drugs,
            "count": len(drugs),
            "limit_applied": limit,
            "query_filter": query_filter,
        }

    @handle_mongo_errors
    async def analyze_drug_classes(
        self,
        group_by: str = "therapeutic_class",
        min_count: int = 1,
        limit: int = 50,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Analyze drug classifications and their distributions.

        This tool performs aggregation analysis on drug classifications,
        providing insights into the distribution of drugs across therapeutic,
        pharmacological, and chemical categories.

        POPULATION-LEVEL ANALYSIS: This tool analyzes drug reference data
        to understand classification distributions across the entire drug database.

        Args:
            group_by: How to group results ("therapeutic_class", "drug_class", "drug_subclass")
            min_count: Minimum number of drugs per class to include
            limit: Maximum number of results to return
            security_context: Security context for access control

        Returns:
            Dict containing drug classification analysis results
        """
        db = self.get_database()

        # Check if drugs collection exists
        collection_name = CollectionNames.DRUGS.value
        if collection_name not in db.list_collection_names():
            logger.warning(f"Drugs collection '{collection_name}' does not exist yet")

            return {
                "success": False,
                "collection": collection_name,
                "error": f"Drugs collection '{collection_name}' does not exist in the database yet",
                "message": "Cannot perform drug class analysis until Drug reference data is ingested",
            }

        collection = db[collection_name]

        # Determine which field to group by
        group_field_map = {
            "therapeutic_class": "therapeutic_class_l2",
            "drug_class": "drug_class_l3",
            "drug_subclass": "drug_subclass_l4"
        }

        if group_by not in group_field_map:
            raise ValueError(f"Invalid group_by value: {group_by}. Must be one of: {list(group_field_map.keys())}")

        group_field = group_field_map[group_by]

        # Build aggregation pipeline
        pipeline = [
            # Match only documents that have the classification field
            {"$match": {group_field: {"$ne": None}}},
            # Group by the classification field
            {
                "$group": {
                    "_id": f"${group_field}",
                    "drug_count": {"$sum": 1},
                    "drug_names": {"$push": "$primary_drug_name"}
                }
            },
            # Filter by minimum count
            {"$match": {"drug_count": {"$gte": min_count}}},
            # Sort by drug count descending
            {"$sort": {"drug_count": -1}},
            # Limit results
            {"$limit": limit},
            # Project final results with sample drugs
            {
                "$project": {
                    "class_name": "$_id",
                    "drug_count": 1,
                    "example_drugs": {"$slice": ["$drug_names", 5]},  # First 5 examples
                    "_id": 0
                }
            }
        ]

        # OBSERVABILITY: Log the analysis query
        logger.info(
            f"\n{'='*70}\n"
            f"DRUG CLASS ANALYSIS:\n"
            f"  Collection: {collection_name}\n"
            f"  Group By: {group_by}\n"
            f"  Field: {group_field}\n"
            f"  Min Count: {min_count}\n"
            f"  Limit: {limit}\n"
            f"  Pipeline Stages: {len(pipeline)}\n"
            f"{'='*70}"
        )

        # Execute aggregation in thread pool (blocking I/O)
        loop = asyncio.get_event_loop()
        executor = get_executor_pool().get_executor()

        results = await loop.run_in_executor(
            executor,
            lambda: list(collection.aggregate(pipeline))
        )

        logger.info(f"✓ Completed drug class analysis: {len(results)} classification groups")

        return {
            "success": True,
            "collection": collection_name,
            "analysis_type": f"drug_classes_grouped_by_{group_by}",
            "group_by": group_by,
            "min_count": min_count,
            "results": results,
            "count": len(results),
        }
