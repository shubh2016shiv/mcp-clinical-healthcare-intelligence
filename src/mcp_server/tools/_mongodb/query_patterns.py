"""Reusable query pattern templates for common MongoDB operations.

This module provides reusable query templates and patterns
for common healthcare data query scenarios.

OBSERVABILITY: All query pattern operations are logged with pattern names
and parameter substitutions for verification.
"""

import logging
from datetime import datetime
from typing import Any

from ..base_tool import BaseTool
from ..utils import handle_mongo_errors

logger = logging.getLogger(__name__)


class QueryPatternsTools(BaseTool):
    """Tools for working with reusable MongoDB query patterns.
    
    This class provides methods for retrieving, building, and validating
    common query patterns for healthcare data operations.
    """

    def __init__(self):
        """Initialize query patterns tools."""
        super().__init__()

    @handle_mongo_errors
    async def get_query_pattern(
        self,
        pattern_name: str,
        parameters: dict[str, Any] | None = None,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Get a pre-built query pattern with optional parameter substitution.

        This method retrieves template queries for common healthcare operations
        and substitutes parameters if provided.

        Args:
            pattern_name: Name of the pattern (e.g., "active_conditions", "patient_medications")
            parameters: Dictionary of parameters to substitute in the pattern
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether pattern was found and built
                - pattern_name: Name of the pattern
                - query: The query template or final query
                - parameters_used: Parameters that were substituted
                - description: Description of what the pattern does
        """
        # Validate pattern name
        available_patterns = {
            "active_conditions",
            "patient_medications",
            "date_range",
            "text_search",
            "recent_encounters",
            "lab_results",
            "patient_demographics",
        }

        if pattern_name not in available_patterns:
            raise ValueError(
                f"Unknown pattern: {pattern_name}. Available: {', '.join(sorted(available_patterns))}"
            )

        # Observability: Log pattern retrieval
        logger.info(
            f"\n{'='*70}\n"
            f"RETRIEVING QUERY PATTERN:\n"
            f"  Pattern: {pattern_name}\n"
            f"  Parameters Provided: {len(parameters) if parameters else 0}\n"
            f"{'='*70}"
        )

        # Get pattern template
        pattern_templates = {
            "active_conditions": {
                "description": "Find all active conditions for patients",
                "query": {"status": "active"},
                "required_params": [],
                "optional_params": ["patient_id", "condition_type"],
            },
            "patient_medications": {
                "description": "Find all medications for a patient",
                "query": {"patient_id": "{{patient_id}}"},
                "required_params": ["patient_id"],
                "optional_params": ["status"],
            },
            "date_range": {
                "description": "Find records within a date range",
                "query": {
                    "{{date_field}}": {
                        "$gte": "{{start_date}}",
                        "$lte": "{{end_date}}"
                    }
                },
                "required_params": ["date_field", "start_date", "end_date"],
                "optional_params": [],
            },
            "text_search": {
                "description": "Search text fields",
                "query": {
                    "$or": [
                        {"{{field1}}": {"$regex": "{{search_term}}", "$options": "i"}},
                        {"{{field2}}": {"$regex": "{{search_term}}", "$options": "i"}},
                    ]
                },
                "required_params": ["search_term"],
                "optional_params": ["field1", "field2"],
            },
            "recent_encounters": {
                "description": "Find recent patient encounters",
                "query": {
                    "patient_id": "{{patient_id}}",
                    "encounter_date": {"$gte": "{{date_threshold}}"}
                },
                "required_params": ["patient_id", "date_threshold"],
                "optional_params": [],
            },
            "lab_results": {
                "description": "Find lab results for a patient",
                "query": {
                    "patient_id": "{{patient_id}}",
                    "result_type": "lab",
                    "status": "final",
                },
                "required_params": ["patient_id"],
                "optional_params": ["date_range"],
            },
            "patient_demographics": {
                "description": "Find patients by demographic criteria",
                "query": {
                    "gender": "{{gender}}",
                    "age_range": "{{age_range}}"
                },
                "required_params": [],
                "optional_params": ["gender", "age_range", "state"],
            },
        }

        template = pattern_templates[pattern_name]

        # Validate parameters
        parameters = parameters or {}
        
        for required_param in template["required_params"]:
            if required_param not in parameters:
                raise ValueError(
                    f"Missing required parameter for {pattern_name}: {required_param}. "
                    f"Required: {template['required_params']}"
                )

        # Build final query with parameter substitution
        query = self._substitute_parameters(template["query"], parameters)

        logger.info(f"✓ Query pattern retrieved: {pattern_name}")

        return {
            "success": True,
            "pattern_name": pattern_name,
            "query": query,
            "description": template["description"],
            "parameters_used": parameters,
            "template_query": template["query"],
        }

    @handle_mongo_errors
    async def build_date_range_query(
        self,
        field: str,
        start_date: str | None = None,
        end_date: str | None = None,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Build a date range filter query.

        This method constructs a MongoDB date range query for finding
        records within a specific time period.

        Args:
            field: The field name to filter on (e.g., "prescribed_date")
            start_date: ISO 8601 start date (e.g., "2024-01-01")
            end_date: ISO 8601 end date (e.g., "2024-12-31")
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether query was built successfully
                - query: The MongoDB date range query
                - field: The field used
                - date_range: The range parameters
        """
        # Validation: Check field name
        if not field or not isinstance(field, str):
            raise ValueError("Field name must be a non-empty string")

        if not field.isidentifier():
            raise ValueError(f"Invalid field name: {field}")

        # Validation: Check dates
        if start_date:
            try:
                datetime.fromisoformat(start_date)
            except ValueError:
                raise ValueError(f"Invalid start date format. Use ISO 8601: {start_date}")

        if end_date:
            try:
                datetime.fromisoformat(end_date)
            except ValueError:
                raise ValueError(f"Invalid end date format. Use ISO 8601: {end_date}")

        # Observability: Log query construction
        logger.info(
            f"\n{'='*70}\n"
            f"BUILDING DATE RANGE QUERY:\n"
            f"  Field: {field}\n"
            f"  Start Date: {start_date}\n"
            f"  End Date: {end_date}\n"
            f"{'='*70}"
        )

        # Build query
        query_conditions = {}

        if start_date:
            query_conditions["$gte"] = start_date

        if end_date:
            query_conditions["$lte"] = end_date

        query = {field: query_conditions}

        logger.info(f"✓ Date range query built: {field}")

        return {
            "success": True,
            "query": query,
            "field": field,
            "date_range": {
                "start_date": start_date,
                "end_date": end_date,
            },
        }

    @handle_mongo_errors
    async def build_text_search_query(
        self,
        fields: list[str],
        search_term: str,
        case_sensitive: bool = False,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Build a text search query across multiple fields.

        This method constructs a MongoDB regex-based text search query
        for finding records matching a search term.

        Args:
            fields: List of field names to search (e.g., ["name", "description"])
            search_term: The search term (will be used as regex pattern)
            case_sensitive: Whether search should be case sensitive (default: False)
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether query was built successfully
                - query: The MongoDB text search query
                - fields: The fields searched
                - search_term: The search term used
        """
        # Validation: Check fields
        if not isinstance(fields, list) or len(fields) == 0:
            raise ValueError("Fields must be a non-empty list")

        for field in fields:
            if not isinstance(field, str) or not field.isidentifier():
                raise ValueError(f"Invalid field name: {field}")

        # Validation: Check search term
        if not search_term or not isinstance(search_term, str):
            raise ValueError("Search term must be a non-empty string")

        # Sanitize search term to prevent regex injection
        search_term = self._sanitize_regex_term(search_term)

        # Observability: Log query construction
        logger.info(
            f"\n{'='*70}\n"
            f"BUILDING TEXT SEARCH QUERY:\n"
            f"  Fields: {fields}\n"
            f"  Search Term: {search_term}\n"
            f"  Case Sensitive: {case_sensitive}\n"
            f"{'='*70}"
        )

        # Build query using $or with regex
        regex_options = "" if case_sensitive else "i"
        
        or_conditions = [
            {field: {"$regex": search_term, "$options": regex_options}}
            for field in fields
        ]

        query = {"$or": or_conditions}

        logger.info(f"✓ Text search query built: {len(fields)} fields")

        return {
            "success": True,
            "query": query,
            "fields": fields,
            "search_term": search_term,
            "case_sensitive": case_sensitive,
        }

    @handle_mongo_errors
    async def build_patient_filter_query(
        self,
        patient_id: str,
        additional_filters: dict[str, Any] | None = None,
        security_context: Any = None,
    ) -> dict[str, Any]:
        """Build a patient-centric filter query.

        This method constructs a MongoDB query that filters by patient_id
        and optionally additional criteria.

        Args:
            patient_id: The patient ID to filter by
            additional_filters: Optional additional filter conditions
            security_context: Security context for access control

        Returns:
            Dict containing:
                - success: Whether query was built successfully
                - query: The MongoDB patient filter query
                - patient_id: The patient ID used
                - filters: Combined filters applied
        """
        # Validation: Check patient_id
        if not patient_id or not isinstance(patient_id, str):
            raise ValueError("Patient ID must be a non-empty string")

        if not patient_id.strip():
            raise ValueError("Patient ID cannot be whitespace only")

        patient_id = patient_id.strip()

        # Validation: Check additional filters
        if additional_filters is not None and not isinstance(additional_filters, dict):
            raise ValueError("Additional filters must be a dictionary")

        additional_filters = additional_filters or {}

        # Observability: Log query construction
        logger.info(
            f"\n{'='*70}\n"
            f"BUILDING PATIENT FILTER QUERY:\n"
            f"  Patient ID: {patient_id}\n"
            f"  Additional Filters: {len(additional_filters)}\n"
            f"{'='*70}"
        )

        # Build query
        query = {"patient_id": patient_id}
        query.update(additional_filters)

        logger.info(f"✓ Patient filter query built: {len(query)} conditions")

        return {
            "success": True,
            "query": query,
            "patient_id": patient_id,
            "filters": query,
            "additional_filters_count": len(additional_filters),
        }

    def _substitute_parameters(self, template: Any, parameters: dict[str, Any]) -> Any:
        """Substitute parameters in a query template (recursive)."""
        if isinstance(template, str):
            result = template
            for param_name, param_value in parameters.items():
                placeholder = f"{{{{param_name}}}}"
                if placeholder in result:
                    result = result.replace(placeholder, str(param_value))
            return result

        elif isinstance(template, dict):
            return {
                key: self._substitute_parameters(value, parameters)
                for key, value in template.items()
            }

        elif isinstance(template, list):
            return [
                self._substitute_parameters(item, parameters)
                for item in template
            ]

        else:
            return template

    def _sanitize_regex_term(self, term: str) -> str:
        """Sanitize a regex search term to prevent injection."""
        # Escape special regex characters
        special_chars = r"\.^$*+?{}[]|()"
        for char in special_chars:
            term = term.replace(char, f"\\{char}")
        return term
