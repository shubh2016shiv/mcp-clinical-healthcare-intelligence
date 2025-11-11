"""Analytics and population-level MCP tools for healthcare data.

This module implements tools for analyzing healthcare data across patient
populations, including condition analysis and financial summaries.

Key Tools:
    - analyze_conditions: Population-level condition analysis and trends
    - get_financial_summary: Financial analysis across claims and billing

Design Philosophy:
    - Aggregation-focused tools for insights across patients
    - Flexible grouping options (by condition, demographics, time)
    - Complex MongoDB aggregation pipelines for efficiency
    - Support both individual record queries and grouped analytics
    - Async operations with ThreadPoolExecutor for non-blocking database access
"""

import asyncio
import logging

from ..database.async_executor import get_executor_pool
from ..security import get_security_manager
from .base_tool import BaseTool
from .models import (
    CollectionNames,
    ConditionAnalysisRequest,
    ConditionAnalysisResponse,
    ConditionGroup,
    ConditionRecord,
    FinancialGroup,
    FinancialRecord,
    FinancialSummaryRequest,
    FinancialSummaryResponse,
)
from .utils import build_compound_filter, build_date_filter, build_text_filter, handle_mongo_errors

logger = logging.getLogger(__name__)


class AnalyticsTools(BaseTool):
    """Tools for population-level healthcare data analysis.

    This class provides methods for analyzing conditions and financial data
    across patient populations using MongoDB aggregation pipelines.

    Inherits optimized connection management from BaseTool, ensuring
    efficient connection pooling and minimal connection overhead.
    """

    def __init__(self):
        """Initialize analytics tools with optimized database connection."""
        super().__init__()

        # Initialize Redis aggregation cache if available
        self._agg_cache = None
        try:
            from src.mcp_server.cache import get_cache_manager

            cache_manager = get_cache_manager()
            if cache_manager.is_available():
                self._agg_cache = cache_manager.aggregation_cache
        except Exception as e:
            logger.debug(f"Aggregation cache not available: {e}")

    @handle_mongo_errors
    async def analyze_conditions(
        self, request: ConditionAnalysisRequest, security_context=None
    ) -> ConditionAnalysisResponse:
        """Analyze conditions across patient populations.

        This tool performs complex analysis of health conditions, either returning
        individual condition records or aggregated statistics grouped by various
        dimensions (condition name, patient demographics, time periods).

        Aggregation Strategies:
        - Individual records: Simple filtered queries
        - Group by condition: Statistics per condition type
        - Group by demographics: Analysis by patient characteristics
        - Group by time: Temporal trends and patterns

        Uses Redis caching to avoid expensive re-computation of aggregations.

        Args:
            request: Analysis parameters including filters and grouping options

        Returns:
            Condition analysis response with records or grouped statistics
        """
        # Try to get cached result for aggregation queries (grouped only)
        if request.group_by and self._agg_cache:
            try:
                request_dict = request.model_dump()
                cached_result = self._agg_cache.get_result_for_query(
                    request_dict, "analyze_conditions"
                )
                if cached_result:
                    logger.debug("Using cached condition analysis result")

                    # Reconstruct response from cache
                    if "conditions" in cached_result:
                        conditions = [
                            ConditionRecord(**cond) for cond in cached_result.get("conditions", [])
                        ]
                    else:
                        conditions = None

                    if "groups" in cached_result:
                        groups = [ConditionGroup(**grp) for grp in cached_result.get("groups", [])]
                    else:
                        groups = None

                    return ConditionAnalysisResponse(
                        analysis_type=cached_result.get("analysis_type", ""),
                        total_count=cached_result.get("total_count", 0),
                        conditions=conditions,
                        groups=groups,
                    )
            except Exception as e:
                logger.debug(f"Failed to get cached condition analysis: {e}")

        db = self.get_database()
        collection = db[CollectionNames.CONDITIONS.value]

        # Build base query filter
        filters = []

        # PATIENT ID VALIDATION: If patient_id is provided, filter to that patient's conditions
        if request.patient_id:
            filters.append({"patient_id": request.patient_id})
            logger.info(f"Filtering condition analysis to patient: {request.patient_id}")

        if request.condition_name:
            filters.append(build_text_filter("condition_name", request.condition_name))

        if request.status:
            filters.append({"status": request.status})

        # Date range filter
        if request.onset_date_start or request.onset_date_end:
            date_filter = build_date_filter(
                "onset_date", request.onset_date_start, request.onset_date_end
            )
            if date_filter:
                filters.append(date_filter)

        query_filter = build_compound_filter(*filters)

        logger.debug(f"Condition analysis query: {query_filter}")

        # Get event loop and executor for async operations
        loop = asyncio.get_event_loop()
        executor = get_executor_pool().get_executor()

        # If no grouping requested, return individual records
        if not request.group_by:
            # Execute query in thread pool
            def fetch_conditions():
                cursor = collection.find(query_filter).limit(request.limit)
                return list(cursor)

            docs = await loop.run_in_executor(executor, fetch_conditions)
            records = []

            for doc in docs:
                record = ConditionRecord(
                    patient_id=doc.get("patient_id", ""),
                    condition_name=doc.get("condition_name"),
                    status=doc.get("status"),
                    onset_date=doc.get("onset_date"),
                    verification_status=doc.get("verification_status"),
                )
                records.append(record)

            # Apply data minimization for individual records
            if security_context:
                security_manager = get_security_manager()
                minimized_records = security_manager.data_minimizer.filter_record_list(
                    [record.model_dump() for record in records], security_context.role
                )
                # Convert back to ConditionRecord objects
                records = [ConditionRecord(**record_dict) for record_dict in minimized_records]

            return ConditionAnalysisResponse(
                analysis_type="individual_records", total_count=len(records), conditions=records
            )

        # Build aggregation pipeline for grouped analysis
        pipeline = [{"$match": query_filter}]

        if request.group_by == "condition":
            # Group by condition name with statistics
            pipeline.extend(
                [
                    {
                        "$group": {
                            "_id": "$condition_name",
                            "total_cases": {"$sum": 1},
                            "active_cases": {
                                "$sum": {"$cond": [{"$eq": ["$status", "active"]}, 1, 0]}
                            },
                            "resolved_cases": {
                                "$sum": {"$cond": [{"$eq": ["$status", "resolved"]}, 1, 0]}
                            },
                            "unique_patients": {"$addToSet": "$patient_id"},
                        }
                    },
                    {
                        "$project": {
                            "condition_name": "$_id",
                            "total_cases": 1,
                            "active_cases": 1,
                            "resolved_cases": 1,
                            "patient_count": {"$size": "$unique_patients"},
                            "_id": 0,
                        }
                    },
                    {"$sort": {"total_cases": -1}},
                    {"$limit": request.limit},
                ]
            )

        elif request.group_by == "patient_demographics":
            # Join with patients and group by demographics
            pipeline.extend(
                [
                    {
                        "$lookup": {
                            "from": CollectionNames.PATIENTS.value,
                            "localField": "patient_id",
                            "foreignField": "patient_id",
                            "as": "patient_info",
                        }
                    },
                    {"$unwind": "$patient_info"},
                    {
                        "$group": {
                            "_id": {
                                "gender": "$patient_info.gender",
                                "race": "$patient_info.race",
                                "ethnicity": "$patient_info.ethnicity",
                                "age_group": {
                                    "$switch": {
                                        "branches": [
                                            {
                                                "case": {"$lt": ["$patient_info.age", 18]},
                                                "then": "0-17",
                                            },
                                            {
                                                "case": {"$lt": ["$patient_info.age", 35]},
                                                "then": "18-34",
                                            },
                                            {
                                                "case": {"$lt": ["$patient_info.age", 55]},
                                                "then": "35-54",
                                            },
                                            {
                                                "case": {"$lt": ["$patient_info.age", 75]},
                                                "then": "55-74",
                                            },
                                        ],
                                        "default": "75+",
                                    }
                                },
                            },
                            "condition_count": {"$sum": 1},
                            "unique_conditions": {"$addToSet": "$condition_name"},
                            "active_conditions": {
                                "$sum": {"$cond": [{"$eq": ["$status", "active"]}, 1, 0]}
                            },
                        }
                    },
                    {
                        "$project": {
                            "demographics": "$_id",
                            "condition_count": 1,
                            "active_condition_count": "$active_conditions",
                            "unique_condition_count": {"$size": "$unique_conditions"},
                            "_id": 0,
                        }
                    },
                    {"$sort": {"condition_count": -1}},
                    {"$limit": request.limit},
                ]
            )

        elif request.group_by == "time_period":
            # Group by year and month
            pipeline.extend(
                [
                    {
                        "$addFields": {
                            "year": {"$year": "$onset_date"},
                            "month": {"$month": "$onset_date"},
                        }
                    },
                    {
                        "$group": {
                            "_id": {"year": "$year", "month": "$month"},
                            "condition_count": {"$sum": 1},
                            "unique_patients": {"$addToSet": "$patient_id"},
                            "unique_conditions": {"$addToSet": "$condition_name"},
                            "active_conditions": {
                                "$sum": {"$cond": [{"$eq": ["$status", "active"]}, 1, 0]}
                            },
                        }
                    },
                    {
                        "$project": {
                            "time_period": "$_id",
                            "condition_count": 1,
                            "patient_count": {"$size": "$unique_patients"},
                            "unique_condition_count": {"$size": "$unique_conditions"},
                            "active_condition_count": "$active_conditions",
                            "_id": 0,
                        }
                    },
                    {"$sort": {"time_period.year": -1, "time_period.month": -1}},
                    {"$limit": request.limit},
                ]
            )

        # Execute aggregation in thread pool (blocking I/O)
        results = await loop.run_in_executor(executor, lambda: list(collection.aggregate(pipeline)))

        # Convert to response groups
        groups = []
        for result in results:
            if request.group_by == "condition":
                group = ConditionGroup(
                    condition_name=result.get("condition_name"),
                    condition_count=result.get("total_cases", 0),
                    patient_count=result.get("patient_count", 0),
                )
            elif request.group_by == "patient_demographics":
                group = ConditionGroup(
                    demographics=result.get("demographics"),
                    condition_count=result.get("condition_count", 0),
                    patient_count=result.get("unique_condition_count", 0),
                )
            elif request.group_by == "time_period":
                group = ConditionGroup(
                    time_period=result.get("time_period"),
                    condition_count=result.get("condition_count", 0),
                    patient_count=result.get("patient_count", 0),
                )
            else:
                group = ConditionGroup(condition_count=result.get("condition_count", 0))

            groups.append(group)

        # Apply data minimization for grouped results
        if security_context:
            security_manager = get_security_manager()
            minimized_groups = security_manager.data_minimizer.filter_record_list(
                [group.model_dump() for group in groups], security_context.role
            )
            # Convert back to ConditionGroup objects
            groups = [ConditionGroup(**group_dict) for group_dict in minimized_groups]

        response = ConditionAnalysisResponse(
            analysis_type=f"grouped_by_{request.group_by}", total_count=len(groups), groups=groups
        )

        # Cache the aggregation result before data minimization for consistency
        if request.group_by and self._agg_cache:
            try:
                request_dict = request.model_dump()
                result_dict = response.model_dump()
                self._agg_cache.cache_for_query(request_dict, result_dict, "analyze_conditions")
            except Exception as e:
                logger.debug(f"Failed to cache condition analysis result: {e}")

        return response

    @handle_mongo_errors
    async def get_financial_summary(
        self, request: FinancialSummaryRequest, security_context=None
    ) -> FinancialSummaryResponse:
        """Analyze financial data from claims and explanation of benefits.

        This tool aggregates financial information from claims data, providing
        insights into healthcare costs, billing patterns, and insurance usage.
        Can group results by patient, insurance provider, facility, or time period.

        Aggregation Strategies:
        - Individual claims: Detailed claim records
        - Group by patient: Per-patient financial summaries
        - Group by insurance: Provider-specific analysis
        - Group by facility: Location-based cost analysis
        - Group by time: Temporal financial trends

        Uses Redis caching to avoid expensive re-computation of aggregations.

        Args:
            request: Financial analysis parameters including filters and grouping

        Returns:
            Financial summary with records or grouped statistics
        """
        # Try to get cached result for aggregation queries (grouped only)
        if request.group_by and self._agg_cache:
            try:
                request_dict = request.model_dump()
                cached_result = self._agg_cache.get_result_for_query(
                    request_dict, "get_financial_summary"
                )
                if cached_result:
                    logger.debug("Using cached financial summary result")

                    # Reconstruct response from cache
                    if "records" in cached_result:
                        records = [
                            FinancialRecord(**rec) for rec in cached_result.get("records", [])
                        ]
                    else:
                        records = None

                    if "groups" in cached_result:
                        groups = [FinancialGroup(**grp) for grp in cached_result.get("groups", [])]
                    else:
                        groups = None

                    return FinancialSummaryResponse(
                        analysis_type=cached_result.get("analysis_type", ""),
                        total_amount=cached_result.get("total_amount", 0),
                        records=records,
                        groups=groups,
                    )
            except Exception as e:
                logger.debug(f"Failed to get cached financial summary: {e}")

        db = self.get_database()
        collection = db[CollectionNames.CLAIMS.value]

        # Build query filter
        filters = []

        # PATIENT ID VALIDATION: If patient_id is provided, filter to that patient's claims
        if request.patient_id:
            filters.append({"patient_id": request.patient_id})
            logger.info(f"Filtering financial analysis to patient: {request.patient_id}")

        if request.insurance_provider:
            filters.append(build_text_filter("insurance_display", request.insurance_provider))

        # Date range
        if request.start_date or request.end_date:
            date_filter = build_date_filter(
                "billable_period_start", request.start_date, request.end_date
            )
            if date_filter:
                filters.append(date_filter)

        # Amount range
        if request.min_amount is not None:
            filters.append({"total_value": {"$gte": request.min_amount}})

        if request.max_amount is not None:
            filters.append({"total_value": {"$lte": request.max_amount}})

        query_filter = build_compound_filter(*filters)

        logger.debug(f"Financial summary query: {query_filter}")

        # Get event loop and executor for async operations
        loop = asyncio.get_event_loop()
        executor = get_executor_pool().get_executor()

        # Build aggregation pipeline
        pipeline = [{"$match": query_filter}]

        if not request.group_by:
            # Return individual claims with summary statistics
            pipeline.extend(
                [
                    {
                        "$group": {
                            "_id": None,
                            "total_claims": {"$sum": 1},
                            "total_amount": {"$sum": "$total_value"},
                            "average_claim": {"$avg": "$total_value"},
                            "min_claim": {"$min": "$total_value"},
                            "max_claim": {"$max": "$total_value"},
                            "claims": {"$push": "$$ROOT"},
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "summary": {
                                "total_claims": "$total_claims",
                                "total_amount": "$total_amount",
                                "average_claim": "$average_claim",
                                "min_claim": "$min_claim",
                                "max_claim": "$max_claim",
                            },
                            "claims": {"$slice": ["$claims", 50]},  # Limit details
                        }
                    },
                ]
            )

        elif request.group_by == "patient":
            pipeline.extend(
                [
                    {
                        "$group": {
                            "_id": "$patient_id",
                            "patient_name": {"$first": "$patient_display_name"},
                            "total_claims": {"$sum": 1},
                            "total_amount": {"$sum": "$total_value"},
                            "average_claim": {"$avg": "$total_value"},
                            "min_claim": {"$min": "$total_value"},
                            "max_claim": {"$max": "$total_value"},
                        }
                    },
                    {
                        "$project": {
                            "patient_id": "$_id",
                            "patient_name": 1,
                            "total_claims": 1,
                            "total_amount": 1,
                            "average_claim": 1,
                            "min_claim": 1,
                            "max_claim": 1,
                            "_id": 0,
                        }
                    },
                    {"$sort": {"total_amount": -1}},
                    {"$limit": 100},
                ]
            )

        elif request.group_by == "insurance":
            pipeline.extend(
                [
                    {
                        "$group": {
                            "_id": "$insurance_display",
                            "total_claims": {"$sum": 1},
                            "total_amount": {"$sum": "$total_value"},
                            "average_claim": {"$avg": "$total_value"},
                            "unique_patients": {"$addToSet": "$patient_id"},
                        }
                    },
                    {
                        "$project": {
                            "insurance_provider": "$_id",
                            "total_claims": 1,
                            "total_amount": 1,
                            "average_claim": 1,
                            "patient_count": {"$size": "$unique_patients"},
                            "_id": 0,
                        }
                    },
                    {"$sort": {"total_amount": -1}},
                    {"$limit": 50},
                ]
            )

        elif request.group_by == "facility":
            pipeline.extend(
                [
                    {
                        "$group": {
                            "_id": "$facility_display",
                            "total_claims": {"$sum": 1},
                            "total_amount": {"$sum": "$total_value"},
                            "average_claim": {"$avg": "$total_value"},
                            "unique_patients": {"$addToSet": "$patient_id"},
                        }
                    },
                    {
                        "$project": {
                            "facility": "$_id",
                            "total_claims": 1,
                            "total_amount": 1,
                            "average_claim": 1,
                            "patient_count": {"$size": "$unique_patients"},
                            "_id": 0,
                        }
                    },
                    {"$sort": {"total_amount": -1}},
                    {"$limit": 50},
                ]
            )

        elif request.group_by == "time_period":
            pipeline.extend(
                [
                    {
                        "$addFields": {
                            "year": {"$year": "$billable_period_start"},
                            "month": {"$month": "$billable_period_start"},
                        }
                    },
                    {
                        "$group": {
                            "_id": {"year": "$year", "month": "$month"},
                            "total_claims": {"$sum": 1},
                            "total_amount": {"$sum": "$total_value"},
                            "average_claim": {"$avg": "$total_value"},
                        }
                    },
                    {
                        "$project": {
                            "time_period": "$_id",
                            "total_claims": 1,
                            "total_amount": 1,
                            "average_claim": 1,
                            "_id": 0,
                        }
                    },
                    {"$sort": {"time_period.year": -1, "time_period.month": -1}},
                    {"$limit": 50},
                ]
            )

        # Execute aggregation in thread pool (blocking I/O)
        results = await loop.run_in_executor(executor, lambda: list(collection.aggregate(pipeline)))

        # Convert results to response format
        if not request.group_by:
            # Individual claims response
            result = results[0] if results else {}
            summary = result.get("summary", {})

            # Convert claims to FinancialRecord objects
            records = []
            for claim in result.get("claims", []):
                record = FinancialRecord(
                    patient_id=claim.get("patient_id"),
                    patient_name=claim.get("patient_display_name"),
                    claim_id=str(claim.get("_id")),
                    facility_name=claim.get("facility_display"),
                    insurance_provider=claim.get("insurance_display"),
                    billable_period=claim.get("billable_period_start"),
                    total_amount=claim.get("total_value", 0),
                )
                records.append(record)

            # Apply data minimization for individual claims
            if security_context:
                security_manager = get_security_manager()
                minimized_records = security_manager.data_minimizer.filter_record_list(
                    [record.model_dump() for record in records], security_context.role
                )
                # Convert back to FinancialRecord objects
                records = [FinancialRecord(**record_dict) for record_dict in minimized_records]

            return FinancialSummaryResponse(
                analysis_type="individual_claims", summary=summary, records=records
            )

        else:
            # Grouped response
            groups = []
            for result in results:
                if request.group_by == "patient":
                    group = FinancialGroup(
                        group_key=result.get("patient_id", ""),
                        total_claims=result.get("total_claims", 0),
                        total_amount=result.get("total_amount", 0),
                        average_claim=result.get("average_claim", 0),
                    )
                elif request.group_by == "insurance":
                    group = FinancialGroup(
                        group_key=result.get("insurance_provider", ""),
                        total_claims=result.get("total_claims", 0),
                        total_amount=result.get("total_amount", 0),
                        average_claim=result.get("average_claim", 0),
                        patient_count=result.get("patient_count", 0),
                    )
                elif request.group_by == "facility":
                    group = FinancialGroup(
                        group_key=result.get("facility", ""),
                        total_claims=result.get("total_claims", 0),
                        total_amount=result.get("total_amount", 0),
                        average_claim=result.get("average_claim", 0),
                        patient_count=result.get("patient_count", 0),
                    )
                elif request.group_by == "time_period":
                    group = FinancialGroup(
                        group_key=f"{result.get('time_period', {}).get('year', '')}-{result.get('time_period', {}).get('month', '')}",
                        total_claims=result.get("total_claims", 0),
                        total_amount=result.get("total_amount", 0),
                        average_claim=result.get("average_claim", 0),
                    )
                groups.append(group)

            # Apply data minimization for grouped financial results
            if security_context:
                security_manager = get_security_manager()
                minimized_groups = security_manager.data_minimizer.filter_record_list(
                    [group.model_dump() for group in groups], security_context.role
                )
                # Convert back to FinancialGroup objects
                groups = [FinancialGroup(**group_dict) for group_dict in minimized_groups]

            response = FinancialSummaryResponse(
                analysis_type=f"financial_summary_grouped_by_{request.group_by}", groups=groups
            )

            # Cache the aggregation result before data minimization for consistency
            if request.group_by and self._agg_cache:
                try:
                    request_dict = request.model_dump()
                    result_dict = response.model_dump()
                    self._agg_cache.cache_for_query(
                        request_dict, result_dict, "get_financial_summary"
                    )
                except Exception as e:
                    logger.debug(f"Failed to cache financial summary result: {e}")

            return response
