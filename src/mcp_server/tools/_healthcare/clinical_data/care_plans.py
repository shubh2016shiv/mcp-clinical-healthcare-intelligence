"""Care plan tools for patient care management.

This module provides tools for querying and analyzing patient
care plans and treatment plans from FHIR CarePlan resources.

PATIENT ID DEPENDENCY: Supports optional patient_id filtering - when provided,
results are constrained to that patient's care plans. Recommended for patient-specific care management.
"""

import logging
from typing import Any

from ...base_tool import BaseTool
from ...models import (
    CarePlanAnalysisGroup,
    CarePlanAnalysisRequest,
    CarePlanAnalysisResponse,
    CarePlanRecord,
    CarePlanRequest,
    CarePlanResponse,
    CollectionNames,
)
from ...utils import (
    build_compound_filter,
    build_date_filter,
    build_text_filter,
    handle_mongo_errors,
)

logger = logging.getLogger(__name__)


class CarePlansTools(BaseTool):
    """Tools for querying and analyzing patient care plans and treatment plans.

    This class provides methods for retrieving and analyzing care plans from the
    care_plans collection. Supports both individual record queries and aggregated
    analysis with flexible grouping options.

    Inherits optimized connection management from BaseTool.
    """

    def __init__(self):
        """Initialize care plans tools with optimized database connection."""
        super().__init__()

    @handle_mongo_errors
    async def get_patient_care_plans(
        self,
        request: CarePlanRequest,
        security_context: Any = None,
    ) -> CarePlanResponse:
        """Query patient care plans and treatment plans.

        This tool retrieves care plan information for patients, which are
        comprehensive plans for patient care including goals, activities, and interventions.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        results are filtered to that patient's care plans only. When omitted,
        returns care plans across all patients (population-level analysis).

        Args:
            request: CarePlanRequest with filter parameters
            security_context: Security context for access control

        Returns:
            CarePlanResponse with care plan records and total count

        Raises:
            ValueError: If patient_id format is invalid when provided
        """
        db = self.get_database()
        collection = db[CollectionNames.CARE_PLANS.value]

        # Build base query filter
        filters = []

        # PATIENT ID VALIDATION: If patient_id is provided, filter to that patient's care plans
        if request.patient_id:
            filters.append({"patient_id": request.patient_id})
            logger.info(f"Filtering care plans to patient: {request.patient_id}")

        if request.plan_name:
            filters.append(build_text_filter("plan_name", request.plan_name))

        if request.status:
            filters.append({"status": request.status})

        # Date range filter
        if request.period_start or request.period_end:
            date_filter = build_date_filter("start_date", request.period_start, request.period_end)
            if date_filter:
                filters.append(date_filter)

        query_filter = build_compound_filter(*filters)

        logger.debug(f"Care plans query: {query_filter}")

        # Execute query directly with Motor (async-native)
        docs = (
            await collection.find(query_filter).limit(request.limit).to_list(length=request.limit)
        )

        # Convert documents to CarePlanRecord objects
        care_plans = []
        for doc in docs:
            try:
                record = CarePlanRecord(
                    patient_id=doc.get("patient_id"),
                    plan_name=doc.get("plan_name"),
                    status=doc.get("status"),
                    start_date=doc.get("start_date"),
                    end_date=doc.get("end_date"),
                    activities=doc.get("activities", []),
                )
                care_plans.append(record)
            except Exception as e:
                logger.warning(f"Failed to convert care plan document: {e}")
                continue

        # OBSERVABILITY: Log the query execution
        logger.info(
            f"\n{'=' * 70}\n"
            f"CARE PLANS QUERY:\n"
            f"  Collection: {CollectionNames.CARE_PLANS.value}\n"
            f"  Patient ID: {request.patient_id}\n"
            f"  Plan Name: {request.plan_name}\n"
            f"  Status: {request.status}\n"
            f"  Period: {request.period_start} to {request.period_end}\n"
            f"  Limit: {request.limit}\n"
            f"  Results: {len(care_plans)}\n"
            f"{'=' * 70}"
        )

        return CarePlanResponse(total_count=len(care_plans), care_plans=care_plans)

    @handle_mongo_errors
    async def analyze_care_plan_patterns(
        self,
        request: CarePlanAnalysisRequest,
        security_context: Any = None,
    ) -> CarePlanAnalysisResponse:
        """Analyze care plan patterns across patients or for a specific patient.

        This tool provides analytical insights into care plan patterns, either
        for population-level analysis or patient-specific care management insights.

        PATIENT ID VALIDATION: Optional patient_id parameter. When provided,
        analysis is limited to that patient's care plans. When omitted, provides
        population-level care plan pattern analysis.

        Args:
            request: CarePlanAnalysisRequest with grouping and filter parameters
            security_context: Security context for access control

        Returns:
            CarePlanAnalysisResponse with analysis results

        Raises:
            ValueError: If request parameters are invalid
        """
        db = self.get_database()
        collection = db[CollectionNames.CARE_PLANS.value]

        if not request.group_by:
            raise ValueError("group_by parameter is required for care plan analysis")

        # Map grouping option to MongoDB field
        group_field_map = {
            "status": "$status",
            "plan_name": "$plan_name",
            "time_period": "$start_date",
        }

        if request.group_by not in group_field_map:
            raise ValueError(
                f"Invalid group_by value. Must be one of: {list(group_field_map.keys())}"
            )

        group_field = group_field_map[request.group_by]

        # Build initial match filter
        match_filter = {}
        if request.patient_id:
            match_filter["patient_id"] = request.patient_id
            logger.info(f"Filtering care plan analysis to patient: {request.patient_id}")

        # Build aggregation pipeline
        pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": group_field,
                    "plan_count": {"$sum": 1},
                    "plan_names": {"$push": "$plan_name"},
                }
            },
            {"$sort": {"plan_count": -1}},
            {"$limit": request.limit},
            {
                "$project": {
                    "group_key": "$_id",
                    "plan_count": 1,
                    "example_plans": {"$slice": ["$plan_names", 5]},
                    "_id": 0,
                }
            },
        ]

        # OBSERVABILITY: Log the analysis query
        logger.info(
            f"\n{'=' * 70}\n"
            f"CARE PLAN ANALYSIS:\n"
            f"  Collection: {CollectionNames.CARE_PLANS.value}\n"
            f"  Group By: {request.group_by}\n"
            f"  Field: {group_field}\n"
            f"  Patient ID: {request.patient_id}\n"
            f"  Limit: {request.limit}\n"
            f"{'=' * 70}"
        )

        # Execute aggregation directly with Motor (async-native)
        results = await collection.aggregate(pipeline).to_list(length=request.limit)

        # Convert to Pydantic models
        groups = [CarePlanAnalysisGroup(**res) for res in results]

        logger.info(f"✓ Completed care plan analysis: {len(groups)} groups")

        return CarePlanAnalysisResponse(
            analysis_type=request.group_by, total_count=len(groups), groups=groups
        )
