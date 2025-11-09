#!/usr/bin/env python3
"""
Data Quality Framework - Enterprise Edition

Comprehensive data quality validation for ETL pipelines with schema validation,
completeness checks, consistency validation, and quality scoring.

Features:
- Pydantic-based schema validation
- Completeness and consistency checks
- Data quality scoring and reporting
- Duplicate detection and profiling
- Referential integrity validation
- Quality metrics collection
"""

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "pydantic is required for data quality validation. Install with: pip install pydantic"
    ) from None

from healthcare_data_pipeline.connection_manager import get_database
from healthcare_data_pipeline.metrics import update_data_quality_score
from healthcare_data_pipeline.structured_logging import get_logger

logger = get_logger(__name__)


class DataQualityRule(BaseModel):
    """Data quality validation rule."""

    name: str
    description: str
    severity: str = Field(default="warning", description="Rule severity: error, warning, info")
    enabled: bool = Field(default=True, description="Whether rule is enabled")

    def validate(self, data: Any, context: dict[str, Any]) -> list["ValidationResult"]:
        """Validate data against this rule.

        Args:
            data: Data to validate
            context: Validation context

        Returns:
            List of validation results
        """
        raise NotImplementedError("Subclasses must implement validate method")


class ValidationResult(BaseModel):
    """Result of a data quality validation."""

    rule_name: str
    severity: str
    message: str
    field_name: str | None = None
    field_value: Any | None = None
    record_id: str | None = None
    collection_name: str | None = None
    suggestion: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletenessRule(DataQualityRule):
    """Rule for checking data completeness."""

    required_fields: list[str]
    allow_empty_strings: bool = Field(default=False)

    def validate(self, data: Any, context: dict[str, Any]) -> list[ValidationResult]:
        """Check for required fields."""
        results = []

        if not isinstance(data, dict):
            return results

        for field in self.required_fields:
            value = data.get(field)

            if value is None:
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Required field '{field}' is missing",
                        field_name=field,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )
            elif not self.allow_empty_strings and isinstance(value, str) and not value.strip():
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Required field '{field}' is empty",
                        field_name=field,
                        field_value=value,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )

        return results


class FormatRule(DataQualityRule):
    """Rule for checking data format."""

    field_name: str
    pattern: str | None = None
    data_type: str | None = None
    min_length: int | None = None
    max_length: int | None = None

    def validate(self, data: Any, context: dict[str, Any]) -> list[ValidationResult]:
        """Check field format."""
        results = []

        if not isinstance(data, dict):
            return results

        value = data.get(self.field_name)
        if value is None:
            return results

        # Type checking
        if self.data_type:
            if self.data_type == "string" and not isinstance(value, str):
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Field '{self.field_name}' should be string, got {type(value).__name__}",
                        field_name=self.field_name,
                        field_value=value,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )
            elif self.data_type == "number" and not isinstance(value, int | float):
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Field '{self.field_name}' should be numeric, got {type(value).__name__}",
                        field_name=self.field_name,
                        field_value=value,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )

        # Pattern matching
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Field '{self.field_name}' does not match required pattern",
                        field_name=self.field_name,
                        field_value=value,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )

        # Length checking
        if isinstance(value, str):
            if self.min_length and len(value) < self.min_length:
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Field '{self.field_name}' is too short (min {self.min_length} chars)",
                        field_name=self.field_name,
                        field_value=value,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )
            if self.max_length and len(value) > self.max_length:
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Field '{self.field_name}' is too long (max {self.max_length} chars)",
                        field_name=self.field_name,
                        field_value=value,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )

        return results


class ReferentialIntegrityRule(DataQualityRule):
    """Rule for checking referential integrity."""

    field_name: str
    referenced_collection: str
    referenced_field: str = "patient_id"
    allow_missing_references: bool = Field(default=False)

    def validate(self, data: Any, context: dict[str, Any]) -> list[ValidationResult]:
        """Check referential integrity."""
        results = []

        if not isinstance(data, dict):
            return results

        reference_value = data.get(self.field_name)
        if reference_value is None:
            return results

        try:
            db = get_database()
            collection = db[self.referenced_collection]

            # Check if referenced record exists
            exists = collection.find_one({self.referenced_field: reference_value})
            if not exists and not self.allow_missing_references:
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Referenced {self.referenced_field} '{reference_value}' not found in {self.referenced_collection}",
                        field_name=self.field_name,
                        field_value=reference_value,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    rule_name=self.name,
                    severity="error",
                    message=f"Failed to check referential integrity: {e}",
                    field_name=self.field_name,
                    field_value=reference_value,
                    record_id=data.get("id")
                    or data.get("patient_id")
                    or data.get("source_fhir_id"),
                    collection_name=context.get("collection_name"),
                )
            )

        return results


class DateConsistencyRule(DataQualityRule):
    """Rule for checking date field consistency."""

    date_fields: list[str]
    max_future_days: int = Field(
        default=1, description="Allow dates up to this many days in the future"
    )
    max_past_years: int = Field(
        default=150, description="Allow dates up to this many years in the past"
    )

    def validate(self, data: Any, context: dict[str, Any]) -> list[ValidationResult]:
        """Check date field consistency."""
        results = []

        if not isinstance(data, dict):
            return results

        now = datetime.utcnow()

        for field_name in self.date_fields:
            value = data.get(field_name)
            if value is None:
                continue

            try:
                if isinstance(value, str):
                    # Try to parse date string
                    if "T" in value:
                        parsed_date = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    else:
                        parsed_date = datetime.strptime(value, "%Y-%m-%d")
                elif isinstance(value, datetime):
                    parsed_date = value
                else:
                    continue

                # Check if date is not too far in the future
                max_future = now + timedelta(days=self.max_future_days)
                if parsed_date > max_future:
                    results.append(
                        ValidationResult(
                            rule_name=self.name,
                            severity=self.severity,
                            message=f"Date in field '{field_name}' is too far in the future",
                            field_name=field_name,
                            field_value=value,
                            record_id=data.get("id")
                            or data.get("patient_id")
                            or data.get("source_fhir_id"),
                            collection_name=context.get("collection_name"),
                        )
                    )

                # Check if date is not too far in the past
                max_past = now - timedelta(days=self.max_past_years * 365)
                if parsed_date < max_past:
                    results.append(
                        ValidationResult(
                            rule_name=self.name,
                            severity=self.severity,
                            message=f"Date in field '{field_name}' is too far in the past",
                            field_name=field_name,
                            field_value=value,
                            record_id=data.get("id")
                            or data.get("patient_id")
                            or data.get("source_fhir_id"),
                            collection_name=context.get("collection_name"),
                        )
                    )

            except (ValueError, TypeError) as e:
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity="warning",
                        message=f"Invalid date format in field '{field_name}': {e}",
                        field_name=field_name,
                        field_value=value,
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=context.get("collection_name"),
                    )
                )

        return results


class DuplicateDetectionRule(DataQualityRule):
    """Rule for detecting duplicate records."""

    duplicate_fields: list[str]
    scope: str = Field(
        default="collection", description="Scope for duplicate detection: collection or global"
    )

    def validate(self, data: Any, context: dict[str, Any]) -> list[ValidationResult]:
        """Check for duplicates based on specified fields."""
        results = []

        if not isinstance(data, dict):
            return results

        try:
            db = get_database()
            collection_name = context.get("collection_name")
            if not collection_name:
                return results

            collection = db[collection_name]

            # Build query for duplicate check
            query = {}
            for field in self.duplicate_fields:
                value = data.get(field)
                if value is not None:
                    query[field] = value

            if not query:
                return results

            # Check if record with same field values exists
            existing = collection.find_one(query)
            if existing and str(existing.get("_id")) != str(data.get("_id", "")):
                results.append(
                    ValidationResult(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Duplicate record found based on fields: {', '.join(self.duplicate_fields)}",
                        field_name=", ".join(self.duplicate_fields),
                        field_value=str(query),
                        record_id=data.get("id")
                        or data.get("patient_id")
                        or data.get("source_fhir_id"),
                        collection_name=collection_name,
                    )
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    rule_name=self.name,
                    severity="error",
                    message=f"Failed to check for duplicates: {e}",
                    record_id=data.get("id")
                    or data.get("patient_id")
                    or data.get("source_fhir_id"),
                    collection_name=context.get("collection_name"),
                )
            )

        return results


class DataQualityValidator:
    """Data quality validator with configurable rules."""

    def __init__(self):
        """Initialize data quality validator."""
        self.rules: dict[str, list[DataQualityRule]] = defaultdict(list)
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up default data quality rules."""

        # Patient collection rules
        self.add_rule(
            "clean_patients",
            CompletenessRule(
                name="patient_required_fields",
                description="Patient records must have required demographic fields",
                required_fields=["patient_id", "first_name", "last_name", "birth_date", "gender"],
            ),
        )

        self.add_rule(
            "clean_patients",
            DateConsistencyRule(
                name="patient_date_consistency",
                description="Patient dates must be reasonable",
                date_fields=["birth_date"],
            ),
        )

        # Condition collection rules
        self.add_rule(
            "clean_conditions",
            CompletenessRule(
                name="condition_required_fields",
                description="Condition records must have essential fields",
                required_fields=["patient_id", "condition_name", "status"],
            ),
        )

        self.add_rule(
            "clean_conditions",
            ReferentialIntegrityRule(
                name="condition_patient_reference",
                description="Condition must reference valid patient",
                field_name="patient_id",
                referenced_collection="clean_patients",
            ),
        )

        # Observation collection rules
        self.add_rule(
            "clean_observations",
            CompletenessRule(
                name="observation_required_fields",
                description="Observation records must have essential fields",
                required_fields=["patient_id", "test_name", "value", "unit", "test_date"],
            ),
        )

        self.add_rule(
            "clean_observations",
            ReferentialIntegrityRule(
                name="observation_patient_reference",
                description="Observation must reference valid patient",
                field_name="patient_id",
                referenced_collection="clean_patients",
            ),
        )

        # Medication collection rules
        self.add_rule(
            "clean_medications",
            CompletenessRule(
                name="medication_required_fields",
                description="Medication records must have essential fields",
                required_fields=["patient_id", "medication_name", "status"],
            ),
        )

        self.add_rule(
            "clean_medications",
            ReferentialIntegrityRule(
                name="medication_patient_reference",
                description="Medication must reference valid patient",
                field_name="patient_id",
                referenced_collection="clean_patients",
            ),
        )

        # Format validation rules
        self.add_rule(
            "clean_patients",
            FormatRule(
                name="patient_name_format",
                description="Patient names should be reasonable length",
                field_name="first_name",
                min_length=1,
                max_length=50,
            ),
        )

        self.add_rule(
            "clean_patients",
            FormatRule(
                name="patient_gender_format",
                description="Gender should be valid enum value",
                field_name="gender",
                pattern="^(male|female|other|unknown)$",
            ),
        )

    def add_rule(self, collection_name: str, rule: DataQualityRule) -> None:
        """Add a data quality rule for a collection.

        Args:
            collection_name: Collection name
            rule: Data quality rule
        """
        self.rules[collection_name].append(rule)

    def validate_record(
        self, collection_name: str, record: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate a single record against all applicable rules.

        Args:
            collection_name: Collection name
            record: Record to validate

        Returns:
            List of validation results
        """
        results = []
        context = {"collection_name": collection_name}

        for rule in self.rules.get(collection_name, []):
            if rule.enabled:
                try:
                    rule_results = rule.validate(record, context)
                    results.extend(rule_results)
                except Exception as e:
                    logger.error(f"Rule validation failed for {rule.name}: {e}")
                    results.append(
                        ValidationResult(
                            rule_name=rule.name,
                            severity="error",
                            message=f"Rule validation failed: {e}",
                            record_id=record.get("id")
                            or record.get("patient_id")
                            or record.get("source_fhir_id"),
                            collection_name=collection_name,
                        )
                    )

        return results

    def validate_collection(
        self, collection_name: str, limit: int | None = None
    ) -> "DataQualityReport":
        """Validate all records in a collection.

        Args:
            collection_name: Collection name
            limit: Maximum records to validate (None for all)

        Returns:
            Data quality report
        """
        try:
            db = get_database()
            collection = db[collection_name]

            cursor = collection.find({})
            if limit:
                cursor = cursor.limit(limit)

            total_records = 0
            valid_records = 0
            all_results = []

            for record in cursor:
                total_records += 1
                results = self.validate_record(collection_name, record)

                if not results:
                    valid_records += 1
                else:
                    all_results.extend(results)

            return DataQualityReport(
                collection_name=collection_name,
                total_records=total_records,
                valid_records=valid_records,
                validation_results=all_results,
                quality_score=(valid_records / total_records * 100) if total_records > 0 else 0.0,
            )

        except Exception as e:
            logger.error(f"Failed to validate collection {collection_name}: {e}")
            return DataQualityReport(
                collection_name=collection_name,
                total_records=0,
                valid_records=0,
                validation_results=[],
                quality_score=0.0,
                error=str(e),
            )


class DataQualityReport(BaseModel):
    """Data quality validation report."""

    collection_name: str
    total_records: int
    valid_records: int
    validation_results: list[ValidationResult]
    quality_score: float
    error: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_error_summary(self) -> dict[str, int]:
        """Get summary of errors by type."""
        error_counts = Counter()
        for result in self.validation_results:
            error_counts[result.rule_name] += 1
        return dict(error_counts)

    def get_severity_summary(self) -> dict[str, int]:
        """Get summary of issues by severity."""
        severity_counts = Counter()
        for result in self.validation_results:
            severity_counts[result.severity] += 1
        return dict(severity_counts)

    def print_report(self) -> None:
        """Print a formatted quality report."""
        print(f"\n{'='*60}")
        print(f"DATA QUALITY REPORT - {self.collection_name.upper()}")
        print(f"{'='*60}")
        print(f"Total Records: {self.total_records:,}")
        print(f"Valid Records: {self.valid_records:,}")
        print(f"Quality Score: {self.quality_score:.1f}%")

        if self.error:
            print(f"Error: {self.error}")
            return

        severity_summary = self.get_severity_summary()
        if severity_summary:
            print("\nISSUES BY SEVERITY:")
            for severity, count in severity_summary.items():
                print(f"  {severity.upper()}: {count}")

        error_summary = self.get_error_summary()
        if error_summary:
            print("\nISSUES BY RULE:")
            for rule, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True):
                print(f"  {rule}: {count}")

        print(f"{'='*60}\n")


class DataQualityProfiler:
    """Data profiling and quality analysis tool."""

    def __init__(self):
        """Initialize data profiler."""
        self.profiles: dict[str, dict[str, Any]] = {}

    def profile_collection(self, collection_name: str, sample_size: int = 10000) -> dict[str, Any]:
        """Profile a collection for data quality insights.

        Args:
            collection_name: Collection name
            sample_size: Number of records to sample

        Returns:
            Profiling results
        """
        try:
            db = get_database()
            collection = db[collection_name]

            # Get sample of records
            cursor = collection.aggregate(
                [{"$sample": {"size": sample_size}}, {"$limit": sample_size}]
            )

            records = list(cursor)
            if not records:
                return {"error": "No records found in collection"}

            profile = {
                "collection_name": collection_name,
                "sample_size": len(records),
                "total_documents": collection.count_documents({}),
                "field_analysis": self._analyze_fields(records),
                "duplicate_analysis": self._analyze_duplicates(records),
                "data_types": self._analyze_data_types(records),
                "generated_at": datetime.utcnow().isoformat(),
            }

            self.profiles[collection_name] = profile
            return profile

        except Exception as e:
            logger.error(f"Failed to profile collection {collection_name}: {e}")
            return {"error": str(e)}

    def _analyze_fields(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze field completeness and patterns."""
        if not records:
            return {}

        field_stats = defaultdict(
            lambda: {"present": 0, "null": 0, "empty_strings": 0, "unique_values": set()}
        )

        for record in records:
            for key, value in record.items():
                field_stats[key]["present"] += 1

                if value is None:
                    field_stats[key]["null"] += 1
                elif isinstance(value, str) and not value.strip():
                    field_stats[key]["empty_strings"] += 1

                # Track unique values (limit to prevent memory issues)
                if len(field_stats[key]["unique_values"]) < 100:
                    field_stats[key]["unique_values"].add(str(value)[:100])

        # Convert to percentages and clean up
        total_records = len(records)
        analysis = {}

        for field, stats in field_stats.items():
            analysis[field] = {
                "completeness": (stats["present"] - stats["null"]) / total_records * 100,
                "null_percentage": stats["null"] / total_records * 100,
                "empty_string_percentage": stats["empty_strings"] / total_records * 100,
                "unique_value_count": len(stats["unique_values"]),
            }

        return analysis

    def _analyze_duplicates(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze potential duplicates."""
        if not records:
            return {}

        # Check for exact duplicates
        record_strings = [str(sorted(record.items())) for record in records]
        duplicate_count = len(record_strings) - len(set(record_strings))

        return {
            "exact_duplicates": duplicate_count,
            "duplicate_percentage": duplicate_count / len(records) * 100 if records else 0,
        }

    def _analyze_data_types(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze data types distribution."""
        if not records:
            return {}

        type_stats = defaultdict(lambda: defaultdict(int))

        for record in records:
            for key, value in record.items():
                type_name = type(value).__name__
                type_stats[key][type_name] += 1

        return dict(type_stats)


# Global instances
_data_quality_validator = DataQualityValidator()
_data_quality_profiler = DataQualityProfiler()


def get_data_quality_validator() -> DataQualityValidator:
    """Get the global data quality validator instance."""
    return _data_quality_validator


def get_data_quality_profiler() -> DataQualityProfiler:
    """Get the global data quality profiler instance."""
    return _data_quality_profiler


def validate_record(collection_name: str, record: dict[str, Any]) -> list[ValidationResult]:
    """Validate a single record."""
    return _data_quality_validator.validate_record(collection_name, record)


def validate_collection(collection_name: str, limit: int | None = None) -> DataQualityReport:
    """Validate a collection."""
    report = _data_quality_validator.validate_collection(collection_name, limit)
    update_data_quality_score(report.quality_score)
    return report


def profile_collection(collection_name: str, sample_size: int = 10000) -> dict[str, Any]:
    """Profile a collection."""
    return _data_quality_profiler.profile_collection(collection_name, sample_size)
