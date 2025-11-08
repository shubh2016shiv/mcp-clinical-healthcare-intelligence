"""Data models for healthcare pipeline ingestion.

This module defines Pydantic models for validating data before ingestion into MongoDB.
Models ensure data quality and consistency across different data sources.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RxNavDrugModel(BaseModel):
    """RxNav drug information model for ATC classification data.

    Represents drug ingredient data extracted from RxNav API with ATC
    (Anatomical Therapeutic Chemical) classification hierarchy. This model
    is specifically designed for RxNorm standardized drug terminology with
    therapeutic classification information.

    Attributes:
        ingredient_rxcui: RxNorm Concept Unique Identifier for the drug ingredient.
            This is the primary key linking to RxNorm database.
        primary_drug_name: Standard/preferred name of the drug ingredient as
            maintained by RxNorm.
        therapeutic_class_l2: ATC Level 2 classification - the therapeutic
            subgroup category (e.g., "Antibacterials for systemic use").
        drug_class_l3: ATC Level 3 classification - the pharmacological
            subgroup providing more specific therapeutic information.
        drug_subclass_l4: ATC Level 4 classification - the chemical subgroup,
            the most specific ATC level indicating the chemical structure/action.
        ingestion_metadata: Metadata dictionary containing ingestion timestamp
            and version information for tracking data provenance.
    """

    ingredient_rxcui: str = Field(
        ...,
        description="RxNorm Concept Unique Identifier (RXCUI) for the drug ingredient",
    )
    primary_drug_name: str = Field(
        ..., description="Standard preferred name of the drug ingredient from RxNorm"
    )
    therapeutic_class_l2: str | None = Field(
        None, description="ATC Level 2 - Therapeutic subgroup classification"
    )
    drug_class_l3: str | None = Field(
        None, description="ATC Level 3 - Pharmacological subgroup classification"
    )
    drug_subclass_l4: str | None = Field(
        None, description="ATC Level 4 - Chemical subgroup classification"
    )
    ingestion_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata including ingestion timestamp and version"
    )

    @field_validator("ingredient_rxcui", mode="before")
    @classmethod
    def validate_rxcui(cls, value: Any) -> str:
        """Validate RXCUI is present and non-empty.

        The RXCUI is the unique identifier linking to RxNorm and must not be empty.

        Args:
            value: The RXCUI value to validate

        Returns:
            Stripped and validated RXCUI as string

        Raises:
            ValueError: If RXCUI is empty or None
        """
        if not value or not str(value).strip():
            raise ValueError("ingredient_rxcui cannot be empty or whitespace-only")
        return str(value).strip()

    @field_validator("primary_drug_name", mode="before")
    @classmethod
    def validate_drug_name(cls, value: Any) -> str:
        """Validate drug name is present and non-empty.

        The drug name provides human-readable identification and must not be empty.

        Args:
            value: The drug name value to validate

        Returns:
            Stripped and validated drug name as string

        Raises:
            ValueError: If drug name is empty or None
        """
        if not value or not str(value).strip():
            raise ValueError("primary_drug_name cannot be empty or whitespace-only")
        return str(value).strip()

    def add_ingestion_metadata(self) -> None:
        """Add or update ingestion metadata with current timestamp.

        This method should be called before database insertion to ensure
        all records have consistent metadata tracking for audit purposes.
        """
        if not self.ingestion_metadata:
            self.ingestion_metadata = {}

        self.ingestion_metadata.update(
            {
                "ingested_at": datetime.now(UTC).isoformat(),
                "ingestion_version": "1.0",
                "source": "rxnav",
            }
        )

    class Config:
        """Pydantic model configuration.

        Allows extra fields to be added without validation errors, providing
        flexibility for future extensions to the data schema.
        """

        extra = "allow"
        json_schema_extra = {
            "example": {
                "ingredient_rxcui": "5640",
                "primary_drug_name": "Aspirin",
                "therapeutic_class_l2": "Antithrombotic agents",
                "drug_class_l3": "Platelet aggregation inhibitors excl. heparin",
                "drug_subclass_l4": "Salicylic acid and derivatives",
                "ingestion_metadata": {
                    "ingested_at": "2024-01-15T10:30:00+00:00",
                    "ingestion_version": "1.0",
                    "source": "rxnav",
                },
            }
        }
