"""Aggregation pipeline construction and optimization tools.

This module provides utilities for building MongoDB aggregation pipelines,
optimizing queries, and managing complex data transformations.

OBSERVABILITY: All pipeline operations are logged before execution with full
details for verification and audit purposes.
"""

import logging
from typing import Any

from ....base_tool import BaseTool
from ....utils import handle_mongo_errors

logger = logging.getLogger(__name__)

# Read-only stage operations allowed in healthcare data access
ALLOWED_PIPELINE_STAGES = {
    "$match",
    "$project",
    "$group",
    "$sort",
    "$limit",
    "$skip",
    "$count",
    "$lookup",
    "$unwind",
    "$facet",
    "$bucket",
    "$sample",
    "$addFields",  # Allowed for data transformation, not mutation
}

# Destructive operations that violate read-only mode
DESTRUCTIVE_OPERATIONS = {"$out", "$merge", "$delete", "$insert", "$replace"}

# Maximum pipeline stages to prevent complexity attacks
MAX_PIPELINE_STAGES = 20


class AggregationBuilderTools(BaseTool):
    """Tools for building, validating, and optimizing MongoDB aggregation pipelines.

    This class provides methods for constructing safe aggregation pipelines for
    healthcare data analysis with comprehensive validation and optimization.
    """

    def __init__(self):
        """Initialize aggregation builder tools."""
        super().__init__()

    @handle_mongo_errors
    async def build_aggregation_pipeline(
        self,
        stages: list[dict],
        validate: bool = True,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Build and validate a MongoDB aggregation pipeline.

        This method constructs aggregation pipelines from stages, validates
        them for safety and correctness, and checks for read-only violations.

        Args:
            stages: List of aggregation pipeline stages (e.g., [{"$match": {...}}, {"$group": {...}}])
            validate: Whether to validate the pipeline (default: True)
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether pipeline was successfully built
                - pipeline: The validated pipeline stages
                - stage_count: Number of stages in pipeline
                - estimated_complexity: Complexity assessment
                - errors: List of validation errors (if any)

        Raises:
            ValueError: If pipeline structure is invalid
        """
        # Validation: Check input type
        if not isinstance(stages, list):
            raise ValueError("Pipeline stages must be a list of dictionaries")

        if not stages:
            raise ValueError("Pipeline must contain at least one stage")

        if len(stages) > MAX_PIPELINE_STAGES:
            raise ValueError(
                f"Pipeline exceeds maximum complexity: {len(stages)} stages (max: {MAX_PIPELINE_STAGES})"
            )

        # Observability: Log pipeline construction attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"BUILDING AGGREGATION PIPELINE:\n"
            f"  Stage Count: {len(stages)}\n"
            f"  Validate: {validate}\n"
            f"  Stages: {[list(stage.keys())[0] if isinstance(stage, dict) else 'INVALID' for stage in stages[:5]]}"
            f"{'...' if len(stages) > 5 else ''}\n"
            f"{'=' * 70}"
        )

        # Validation: Check stage structure and read-only violations
        validation_errors = []

        for idx, stage in enumerate(stages):
            if not isinstance(stage, dict):
                validation_errors.append(
                    f"Stage {idx}: Must be a dictionary, got {type(stage).__name__}"
                )
                continue

            if len(stage) != 1:
                validation_errors.append(
                    f"Stage {idx}: Must contain exactly one key (e.g., {{'$match': ...}}), got {len(stage)} keys"
                )
                continue

            stage_name = list(stage.keys())[0]

            # Check for destructive operations
            if stage_name in DESTRUCTIVE_OPERATIONS:
                validation_errors.append(
                    f"Stage {idx}: Destructive operation '{stage_name}' not allowed in read-only mode"
                )
                continue

            # Check if stage is allowed
            if stage_name not in ALLOWED_PIPELINE_STAGES:
                validation_errors.append(
                    f"Stage {idx}: Operation '{stage_name}' is not supported for healthcare data"
                )

        if validation_errors and validate:
            logger.warning(f"Pipeline validation failed: {validation_errors}")
            return {
                "success": False,
                "pipeline": None,
                "stage_count": len(stages),
                "errors": validation_errors,
                "estimated_complexity": self._estimate_complexity(stages),
            }

        # Estimate pipeline complexity
        complexity = self._estimate_complexity(stages)

        logger.info(
            f"✓ Pipeline validated successfully: {len(stages)} stages, complexity={complexity}"
        )

        return {
            "success": True,
            "pipeline": stages,
            "stage_count": len(stages),
            "estimated_complexity": complexity,
            "errors": validation_errors if not validate else [],
        }

    @handle_mongo_errors
    async def validate_pipeline_stages(
        self,
        stages: list[dict],
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Validate each stage in an aggregation pipeline.

        This method performs detailed validation of pipeline stages to ensure
        correctness, safety, and compliance with read-only mode.

        Args:
            stages: List of pipeline stages to validate
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether all stages are valid
                - total_stages: Total number of stages
                - valid_stages: List of valid stage indices
                - invalid_stages: List of invalid stage indices with errors
                - allowed_operations: List of allowed operations
        """
        if not isinstance(stages, list):
            raise ValueError("Stages must be a list")

        # Observability: Log validation attempt
        logger.info(
            f"\n{'=' * 70}\nVALIDATING PIPELINE STAGES:\n  Total Stages: {len(stages)}\n{'=' * 70}"
        )

        valid_indices = []
        invalid_stages = []

        for idx, stage in enumerate(stages):
            if not isinstance(stage, dict) or len(stage) != 1:
                invalid_stages.append(
                    {"index": idx, "error": "Stage must be a dictionary with exactly one key"}
                )
                continue

            stage_name = list(stage.keys())[0]

            if stage_name in DESTRUCTIVE_OPERATIONS:
                invalid_stages.append(
                    {
                        "index": idx,
                        "stage": stage_name,
                        "error": "Destructive operation not allowed in read-only mode",
                    }
                )
            elif stage_name not in ALLOWED_PIPELINE_STAGES:
                invalid_stages.append(
                    {
                        "index": idx,
                        "stage": stage_name,
                        "error": "Operation not supported for healthcare data",
                    }
                )
            else:
                valid_indices.append(idx)

        logger.info(
            f"✓ Validation complete: {len(valid_indices)} valid, {len(invalid_stages)} invalid"
        )

        return {
            "success": len(invalid_stages) == 0,
            "total_stages": len(stages),
            "valid_stages": valid_indices,
            "invalid_stages": invalid_stages,
            "allowed_operations": sorted(ALLOWED_PIPELINE_STAGES),
        }

    @handle_mongo_errors
    async def analyze_pipeline_performance(
        self,
        collection_name: str,
        pipeline: list[dict],
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Analyze aggregation pipeline performance using MongoDB explain().

        This method analyzes how MongoDB executes the pipeline, identifying
        performance characteristics and suggesting optimizations.

        Args:
            collection_name: Name of the collection to analyze
            pipeline: Aggregation pipeline stages
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether analysis was successful
                - collection: Collection analyzed
                - stage_count: Number of stages
                - execution_stats: MongoDB explain output
                - performance_notes: Observations about performance
                - recommendations: Suggested optimizations
        """
        db = self.get_database()

        # Validation: Check collection exists
        if collection_name not in db.list_collection_names():
            logger.warning(f"Collection '{collection_name}' does not exist")
            return {
                "success": False,
                "collection": collection_name,
                "error": f"Collection '{collection_name}' does not exist",
            }

        # Validation: Validate pipeline
        validation = await self.build_aggregation_pipeline(pipeline, validate=True)
        if not validation["success"]:
            return {
                "success": False,
                "collection": collection_name,
                "error": "Pipeline validation failed",
                "errors": validation.get("errors", []),
            }

        collection = db[collection_name]

        # Observability: Log analysis attempt
        logger.info(
            f"\n{'=' * 70}\n"
            f"ANALYZING PIPELINE PERFORMANCE:\n"
            f"  Collection: {collection_name}\n"
            f"  Stages: {len(pipeline)}\n"
            f"{'=' * 70}"
        )

        # Execute analysis directly with Motor (async-native)
        try:
            # Use Motor's async explain
            explain_result = await collection.aggregate(pipeline).explain()
        except Exception as e:
            logger.error(f"Explain failed: {e}")
            explain_result = None

        if explain_result is None:
            return {
                "success": False,
                "collection": collection_name,
                "error": "Failed to execute pipeline explain",
            }

        # Analyze results
        notes = []
        recommendations = []

        # Check for COLLSCAN (full collection scan - bad)
        if self._contains_collscan(explain_result):
            notes.append("Pipeline performs full collection scan (COLLSCAN)")
            recommendations.append("Consider adding indexes on $match fields")

        # Check stage count
        if len(pipeline) > 10:
            notes.append(f"Pipeline has {len(pipeline)} stages (potentially complex)")
            recommendations.append("Consider simplifying pipeline or using $facet")

        logger.info(f"✓ Performance analysis complete: {len(notes)} notes")

        return {
            "success": True,
            "collection": collection_name,
            "stage_count": len(pipeline),
            "execution_stats": self._extract_explain_stats(explain_result),
            "performance_notes": notes,
            "recommendations": recommendations,
        }

    def _estimate_complexity(self, stages: list[dict]) -> str:
        """Estimate pipeline complexity based on stages."""
        if not stages:
            return "minimal"

        stage_weights = {
            "$lookup": 3,
            "$group": 2,
            "$facet": 3,
            "$bucket": 2,
            "$sort": 1,
            "$match": 0,
            "$project": 0,
        }

        total_weight = 0
        for stage in stages:
            if isinstance(stage, dict):
                stage_name = list(stage.keys())[0]
                total_weight += stage_weights.get(stage_name, 1)

        if total_weight <= 2:
            return "minimal"
        elif total_weight <= 5:
            return "low"
        elif total_weight <= 10:
            return "medium"
        else:
            return "high"

    def _contains_collscan(self, explain_result: dict) -> bool:
        """Check if explain result contains collection scan."""
        if not explain_result:
            return False

        try:
            stages = explain_result.get("stages", [])
            for stage in stages:
                if "COLLSCAN" in str(stage):
                    return True
            return False
        except Exception:
            return False

    def _extract_explain_stats(self, explain_result: dict) -> dict[str, Any]:
        """Extract relevant statistics from explain output."""
        if not explain_result:
            return {}

        try:
            return {
                "stages": len(explain_result.get("stages", [])),
                "documents_examined": explain_result.get("executionStats", {}).get(
                    "totalDocsExamined", 0
                ),
                "documents_returned": explain_result.get("executionStats", {}).get("nReturned", 0),
            }
        except Exception:
            return {}
