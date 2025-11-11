"""Pydantic models and enums for MCP tool requests and responses.

This module defines all the data models used by the healthcare MCP tools.
Using Pydantic provides automatic validation, serialization, and documentation.

Key Components:
    - CollectionNames enum for type safety
    - Request models for each tool
    - Response models for structured data
    - Field validation and descriptions

Design Principles:
    - Comprehensive field descriptions for MCP tool documentation
    - Reasonable defaults and limits
    - Type safety with enums where appropriate
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CollectionNames(str, Enum):
    """Enum for MongoDB collection names to ensure type safety and prevent typos.

    This enum maps to the collections created by the healthcare data pipeline.
    Each collection contains a specific type of FHIR resource.
    """

    PATIENTS = "patients"
    ENCOUNTERS = "encounters"
    CONDITIONS = "conditions"
    OBSERVATIONS = "observations"
    MEDICATIONS = "medications"  # medicationrequests collection
    ALLERGIES = "allergies"  # allergyintolerances collection
    PROCEDURES = "procedures"
    IMMUNIZATIONS = "immunizations"
    CARE_PLANS = "care_plans"
    DIAGNOSTIC_REPORTS = "diagnosticreports"
    CLAIMS = "claims"
    EOB = "explanationofbenefits"  # explanationofbenefits collection
    DRUGS = "drugs"


# =============================================================================
# PATIENT TOOLS MODELS
# =============================================================================


class SearchPatientsRequest(BaseModel):
    """Request model for searching patients with flexible criteria.

    This model supports flexible patient searches using various demographic
    and administrative fields. All text searches are case-insensitive partial matches.
    """

    first_name: str | None = Field(
        None, description="Patient's first name (case-insensitive partial match)"
    )
    last_name: str | None = Field(
        None, description="Patient's last name (case-insensitive partial match)"
    )
    patient_id: str | None = Field(None, description="Exact patient ID for precise lookup")
    city: str | None = Field(None, description="City from patient's address")
    state: str | None = Field(None, description="State code (e.g., 'TX', 'CA')")
    birth_date_start: str | None = Field(
        None, description="Filter patients born on or after this date (YYYY-MM-DD)"
    )
    birth_date_end: str | None = Field(
        None, description="Filter patients born on or before this date (YYYY-MM-DD)"
    )
    gender: Literal["male", "female", "other"] | None = Field(None, description="Patient's gender")
    limit: int = Field(20, ge=1, le=200, description="Maximum number of results (1-200)")

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value: str | None) -> str | None:
        """Validate patient ID format for security.

        Rationale: Patient IDs are high-risk inputs that can be used for injection attacks.
        Strict validation prevents malicious input while allowing legitimate IDs.

        Args:
            value: Patient ID to validate

        Returns:
            Validated patient ID

        Raises:
            ValueError: If patient ID format is invalid
        """
        if value is not None:
            from ..security import get_security_manager

            try:
                security_manager = get_security_manager()
                return security_manager.validator.validate_patient_id(value)
            except RuntimeError:
                # Security not initialized, skip validation for now
                pass
        return value

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_names(cls, value: str | None, info) -> str | None:
        """Validate name fields for security.

        Rationale: Names can contain dangerous characters or be used for injection.
        Validation ensures names follow expected patterns and don't contain exploits.

        Args:
            value: Name to validate
            info: Field validation info

        Returns:
            Validated name

        Raises:
            ValueError: If name validation fails
        """
        if value is not None:
            from ..security import get_security_manager

            try:
                security_manager = get_security_manager()
                field_name = info.field_name
                return security_manager.validator.validate_name_field(value, field_name)
            except RuntimeError:
                # Security not initialized, skip validation for now
                pass
        return value

    @model_validator(mode="after")
    def validate_date_range(self) -> "SearchPatientsRequest":
        """Validate date range logic for security.

        Rationale: Date ranges can be manipulated for data exfiltration.
        Validation ensures logical consistency and prevents abuse.

        Returns:
            Validated request

        Raises:
            ValueError: If date range is invalid
        """
        if self.birth_date_start and self.birth_date_end:
            from ..security import get_security_manager

            try:
                security_manager = get_security_manager()
                validated_start, validated_end = security_manager.validator.validate_date_range(
                    self.birth_date_start, self.birth_date_end
                )
                self.birth_date_start = validated_start
                self.birth_date_end = validated_end
            except RuntimeError:
                # Security not initialized, skip validation for now
                pass
        return self


class PatientSummary(BaseModel):
    """Summary representation of a patient record."""

    patient_id: str = Field(..., description="Unique patient identifier")
    first_name: str | None = Field(None, description="Patient's first name")
    last_name: str | None = Field(None, description="Patient's last name")
    birth_date: str | None = Field(None, description="Patient's birth date")
    gender: str | None = Field(None, description="Patient's gender")
    city: str | None = Field(None, description="Patient's city")
    state: str | None = Field(None, description="Patient's state")


class ClinicalTimelineRequest(BaseModel):
    """Request model for fetching a patient's complete clinical timeline."""

    patient_id: str = Field(
        ...,  # Required field
        description="Patient ID to fetch timeline for",
    )
    start_date: str | None = Field(
        None, description="Filter events from this date onwards (YYYY-MM-DD)"
    )
    end_date: str | None = Field(None, description="Filter events up to this date (YYYY-MM-DD)")
    event_types: list[str] | None = Field(
        None,
        description="Filter specific event types: encounters, conditions, medications, procedures, immunizations, observations",
    )
    limit: int = Field(100, ge=1, le=500, description="Maximum number of events to return")

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value: str) -> str:
        """Validate patient ID format for security.

        Rationale: Patient IDs in timeline requests are critical for PHI access control.
        Validation prevents unauthorized access to clinical data.

        Args:
            value: Patient ID to validate

        Returns:
            Validated patient ID

        Raises:
            ValueError: If patient ID format is invalid
        """
        from ..security import get_security_manager

        try:
            security_manager = get_security_manager()
            return security_manager.validator.validate_patient_id(value)
        except RuntimeError:
            # Security not initialized, return as-is for now
            return value

    @model_validator(mode="after")
    def validate_timeline_dates(self) -> "ClinicalTimelineRequest":
        """Validate timeline date range for security.

        Rationale: Timeline date ranges can be abused for bulk data extraction.
        Validation ensures logical date ranges and prevents abuse.

        Returns:
            Validated request

        Raises:
            ValueError: If date range is invalid
        """
        if self.start_date and self.end_date:
            from ..security import get_security_manager

            try:
                security_manager = get_security_manager()
                validated_start, validated_end = security_manager.validator.validate_date_range(
                    self.start_date, self.end_date
                )
                self.start_date = validated_start
                self.end_date = validated_end
            except RuntimeError:
                # Security not initialized, skip validation for now
                pass
        return self


class ClinicalEvent(BaseModel):
    """Represents a single clinical event in a patient's timeline."""

    event_type: str = Field(..., description="Type of clinical event")
    event_date: str | None = Field(None, description="Date of the event")
    event_name: str | None = Field(None, description="Name or description of the event")
    source_id: str = Field(..., description="Original document ID in MongoDB")
    details: dict[str, Any] = Field(..., description="Additional event details")


class ClinicalTimelineResponse(BaseModel):
    """Response model for clinical timeline queries."""

    patient_id: str = Field(..., description="Patient ID")
    total_events: int = Field(..., description="Total number of events found")
    events: list[ClinicalEvent] = Field(..., description="Chronological list of clinical events")
    date_range: dict[str, str | None] = Field(..., description="Applied date range filter")


# =============================================================================
# ANALYTICS TOOLS MODELS
# =============================================================================


class ConditionAnalysisRequest(BaseModel):
    """Request model for analyzing conditions across patient populations.

    PATIENT ID VALIDATION: Optional patient_id field. When provided, results
    are filtered to that patient's conditions only (patient-specific analysis).
    When omitted, returns population-level analysis across all patients.
    """

    patient_id: str | None = Field(
        None, description="Optional: Filter by specific patient ID for patient-centric analysis"
    )
    condition_name: str | None = Field(
        None, description="Partial match for condition name (case-insensitive)"
    )
    status: Literal["active", "resolved", "inactive"] | None = Field(
        None, description="Condition status filter"
    )
    onset_date_start: str | None = Field(
        None, description="Filter conditions that started on or after this date"
    )
    onset_date_end: str | None = Field(
        None, description="Filter conditions that started on or before this date"
    )
    group_by: Literal["condition", "patient_demographics", "time_period"] | None = Field(
        None, description="How to aggregate the results"
    )
    limit: int = Field(50, ge=1, le=200, description="Maximum number of results")


class ConditionRecord(BaseModel):
    """Individual condition record."""

    patient_id: str = Field(..., description="Patient who has this condition")
    condition_name: str | None = Field(None, description="Name of the condition")
    status: str | None = Field(None, description="Condition status")
    onset_date: str | None = Field(None, description="When the condition started")
    verification_status: str | None = Field(None, description="Verification status")


class ConditionGroup(BaseModel):
    """Grouped condition analysis result."""

    condition_name: str | None = Field(
        None, description="Condition name (when grouped by condition)"
    )
    demographics: dict[str, Any] | None = Field(None, description="Demographic breakdown")
    time_period: dict[str, Any] | None = Field(None, description="Time period information")
    condition_count: int = Field(..., description="Number of conditions in this group")
    patient_count: int | None = Field(None, description="Number of unique patients")
    unique_condition_count: int | None = Field(None, description="Number of unique conditions")


class ConditionAnalysisResponse(BaseModel):
    """Response model for condition analysis."""

    analysis_type: str = Field(..., description="Type of analysis performed")
    total_count: int | None = Field(None, description="Total number of records")
    conditions: list[ConditionRecord] | None = Field(
        None, description="Individual condition records"
    )
    groups: list[ConditionGroup] | None = Field(None, description="Grouped analysis results")


class FinancialSummaryRequest(BaseModel):
    """Request model for financial analysis across claims and EOBs."""

    patient_id: str | None = Field(None, description="Filter by specific patient")
    start_date: str | None = Field(None, description="Filter claims from this date (YYYY-MM-DD)")
    end_date: str | None = Field(None, description="Filter claims up to this date (YYYY-MM-DD)")
    insurance_provider: str | None = Field(
        None, description="Filter by insurance provider name (partial match)"
    )
    min_amount: float | None = Field(None, ge=0, description="Minimum claim amount in USD")
    max_amount: float | None = Field(None, ge=0, description="Maximum claim amount in USD")
    group_by: Literal["patient", "insurance", "facility", "time_period"] | None = Field(
        None, description="How to aggregate financial data"
    )


class FinancialRecord(BaseModel):
    """Individual financial record."""

    patient_id: str | None = Field(None, description="Patient ID")
    patient_name: str | None = Field(None, description="Patient name")
    claim_id: str | None = Field(None, description="Claim identifier")
    facility_name: str | None = Field(None, description="Healthcare facility")
    insurance_provider: str | None = Field(None, description="Insurance provider")
    billable_period: str | None = Field(None, description="Billing period")
    total_amount: float = Field(..., description="Total claim amount")


class FinancialGroup(BaseModel):
    """Grouped financial analysis result."""

    group_key: str = Field(
        ..., description="Grouping key (patient, insurance, facility, or time period)"
    )
    total_claims: int = Field(..., description="Total number of claims")
    total_amount: float = Field(..., description="Total amount across all claims")
    average_claim: float = Field(..., description="Average claim amount")
    patient_count: int | None = Field(None, description="Number of unique patients")


class FinancialSummaryResponse(BaseModel):
    """Response model for financial summary queries."""

    analysis_type: str = Field(..., description="Type of financial analysis")
    currency: str = Field(default="USD", description="Currency for amounts")
    summary: dict[str, Any] | None = Field(None, description="Overall summary statistics")
    records: list[FinancialRecord] | None = Field(None, description="Individual financial records")
    groups: list[FinancialGroup] | None = Field(None, description="Grouped analysis results")


# =============================================================================
# MEDICATION TOOLS MODELS
# =============================================================================


class MedicationHistoryRequest(BaseModel):
    """Request model for medication and drug information retrieval."""

    patient_id: str | None = Field(None, description="Filter by specific patient")
    medication_name: str | None = Field(
        None, description="Partial match for medication name (case-insensitive)"
    )
    drug_class: str | None = Field(
        None, description="Filter by drug class (L3 level from ATC classification)"
    )
    status: Literal["active", "completed", "stopped"] | None = Field(
        None, description="Medication status filter"
    )
    prescribed_date_start: str | None = Field(
        None, description="Filter medications prescribed on or after this date"
    )
    include_drug_details: bool = Field(
        True, description="Whether to enrich with drug class and therapeutic information"
    )
    limit: int = Field(50, ge=1, le=200, description="Maximum number of results")


class MedicationRecord(BaseModel):
    """Individual medication record."""

    patient_id: str = Field(..., description="Patient ID")
    medication_name: str | None = Field(None, description="Name of the medication")
    prescriber: str | None = Field(None, description="Prescribing provider")
    status: str | None = Field(None, description="Medication status")
    prescribed_date: str | None = Field(None, description="Date medication was prescribed")
    dosage_instruction: str | None = Field(None, description="Dosage instructions")
    drug_classification: dict[str, Any] | None = Field(
        None, description="Drug classification information from RxNorm"
    )


class MedicationHistoryResponse(BaseModel):
    """Response model for medication history queries."""

    total_medications: int = Field(..., description="Total number of medication records")
    medications: list[MedicationRecord] = Field(..., description="List of medication records")
    enriched_with_drug_data: bool = Field(
        ..., description="Whether drug classification was included"
    )


# =============================================================================
# DRUG TOOLS MODELS
# =============================================================================


class DrugRecord(BaseModel):
    """Individual drug record from the drugs collection."""

    ingredient_rxcui: str = Field(..., description="RxNorm Concept Unique Identifier (primary key)")
    primary_drug_name: str | None = Field(None, description="Standardized drug ingredient name")
    therapeutic_class_l2: str | None = Field(None, description="Therapeutic class (level 2 ATC)")
    drug_class_l3: str | None = Field(None, description="Drug class (level 3 ATC)")
    drug_subclass_l4: str | None = Field(
        None, description="Drug subclass (level 4 ATC - most specific)"
    )
    ingestion_metadata: dict[str, Any] | None = Field(
        None, description="Ingestion timestamp and version info"
    )


class SearchDrugsRequest(BaseModel):
    """Request model for searching drugs in the drugs collection."""

    drug_name: str | None = Field(
        None, description="Partial match for drug name (case-insensitive)"
    )
    therapeutic_class: str | None = Field(
        None, description="Filter by therapeutic class (L2 ATC classification)"
    )
    drug_class: str | None = Field(None, description="Filter by drug class (L3 ATC classification)")
    drug_subclass: str | None = Field(
        None, description="Filter by drug subclass (L4 ATC classification)"
    )
    rxcui: str | None = Field(None, description="Exact RxCUI match")
    limit: int = Field(50, ge=1, le=200, description="Maximum number of results")


class DrugClassAnalysisRequest(BaseModel):
    """Request model for analyzing drug classifications."""

    group_by: Literal["therapeutic_class", "drug_class", "drug_subclass"] = Field(
        ..., description="How to group the analysis"
    )
    min_count: int = Field(1, ge=1, description="Minimum number of drugs in a class to include")
    limit: int = Field(50, ge=1, le=100, description="Maximum number of groups to return")


class DrugSearchResponse(BaseModel):
    """Response model for drug search queries."""

    total_drugs: int = Field(..., description="Total number of drug records found")
    drugs: list[DrugRecord] = Field(..., description="List of matching drug records")


class DrugClassGroup(BaseModel):
    """Grouped drug classification result."""

    class_name: str = Field(..., description="Name of the drug class")
    drug_count: int = Field(..., description="Number of drugs in this class")
    example_drugs: list[str] = Field(..., description="Sample drug names in this class")


class DrugClassAnalysisResponse(BaseModel):
    """Response model for drug classification analysis."""

    analysis_type: str = Field(..., description="Type of classification analysis")
    total_classes: int = Field(..., description="Total number of classes found")
    classes: list[DrugClassGroup] = Field(..., description="Grouped classification results")


# =============================================================================
# COMMON RESPONSE MODELS
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error message")
    details: str | None = Field(None, description="Additional error details")
    operation: str | None = Field(None, description="Operation that failed")
