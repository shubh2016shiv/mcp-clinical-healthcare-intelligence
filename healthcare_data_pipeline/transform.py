#!/usr/bin/env python3
"""
FHIR Data Transformation Pipeline

Transforms raw FHIR data from MongoDB into clean, medically relevant data
using Pydantic models. Integrates with Gemini LLM for data enrichment and
validation of medical relevance.

This script reads from existing MongoDB collections and writes transformed
data to new 'clean_*' collections.
"""

import base64
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from tqdm import tqdm

# Configure logging first (before .env loading to avoid issues)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to load .env file from project root
try:
    from dotenv import load_dotenv

    # Load .env from project root (parent of healthcare_data_pipeline directory)
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded .env file from {env_path}")
except ImportError:
    # python-dotenv not installed, skip .env loading
    logger.debug("python-dotenv not installed, skipping .env file loading")
except Exception as e:
    logger.debug(f"Error loading .env file: {e}")

# Conditional import for Gemini (only needed if using enrichment)
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


# ============================================================================
# MAIN TRANSFORMATION PIPELINE
# ============================================================================


class FHIRTransformationPipeline:
    """Main pipeline for transforming FHIR data to clean medical records."""

    def __init__(
        self,
        mongo_client: MongoClient,
        db_name: str = "text_to_mongo_db",
        gemini_api_key: str | None = None,
        use_gemini: bool = True,
    ):
        """Initialize transformation pipeline.

        Args:
            mongo_client: MongoDB client instance
            db_name: Database name
            gemini_api_key: Google AI API key for Gemini
            use_gemini: Whether to use Gemini for enrichment
        """
        self.client = mongo_client
        self.db = mongo_client[db_name]
        self.gemini = None

        if use_gemini and gemini_api_key:
            try:
                self.gemini = GeminiEnricher(gemini_api_key)
                logger.info("Gemini enrichment enabled")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}")
                logger.info("Continuing without Gemini enrichment")

        # Transformation mapping will be resolved lazily when first accessed
        # (functions are defined later in this file)
        self._transformations = None

    @property
    def transformations(self):
        """Lazy-load transformations mapping after all functions are defined."""
        if self._transformations is None:
            import sys

            # When script is run directly, __name__ is '__main__', so use that
            # Otherwise use the actual module name
            module_name = self.__class__.__module__
            if module_name == "__main__":
                # Script is run directly, functions are in __main__ namespace
                current_module = sys.modules["__main__"]
            else:
                current_module = sys.modules[module_name]

            transformations_map = {
                "allergyintolerances": ("clean_allergies", "transform_allergy"),
                "conditions": ("clean_conditions", "transform_condition"),
                "observations": ("clean_observations", "transform_observation"),
                "medicationrequests": ("clean_medications", "transform_medication_request"),
                "immunizations": ("clean_immunizations", "transform_immunization"),
                "procedures": ("clean_procedures", "transform_procedure"),
                "encounters": ("clean_encounters", "transform_encounter"),
                "careplans": ("clean_care_plans", "transform_care_plan"),
                "patients": ("clean_patients", "transform_patient"),
                "diagnosticreports": ("clean_diagnostic_reports", "transform_diagnostic_report"),
                "claims": ("clean_claims", "transform_claim"),
                "explanationofbenefits": (
                    "clean_explanation_of_benefits",
                    "transform_explanation_of_benefit",
                ),
            }

            self._transformations = {}
            for source_coll, (target_coll, func_name) in transformations_map.items():
                func = getattr(current_module, func_name, None)
                if func is None or not callable(func):
                    # Try to find what's actually available
                    available = [
                        k
                        for k in dir(current_module)
                        if k.startswith("transform_") and callable(getattr(current_module, k, None))
                    ]
                    raise RuntimeError(
                        f"Transformation function {func_name} not found in {module_name}. "
                        f"Available transform functions: {available}"
                    )
                self._transformations[source_coll] = (target_coll, func)

        return self._transformations

    def transform_collection(
        self,
        source_collection: str,
        target_collection: str,
        transform_func: callable,
        batch_size: int = 500,
    ) -> dict[str, int]:
        """Transform a single collection.

        Args:
            source_collection: Source collection name
            target_collection: Target collection name
            transform_func: Transformation function
            batch_size: Batch size for bulk operations

        Returns:
            Statistics dictionary
        """
        stats = {"processed": 0, "transformed": 0, "skipped": 0, "errors": 0}

        source_coll = self.db[source_collection]
        target_coll = self.db[target_collection]

        # Get total count
        total_docs = source_coll.count_documents({})

        if total_docs == 0:
            logger.info(f"No documents in {source_collection}, skipping")
            return stats

        logger.info(f"Transforming {source_collection} -> {target_collection} ({total_docs} docs)")

        # Process in batches
        batch = []

        with tqdm(total=total_docs, desc=f"Processing {source_collection}") as pbar:
            cursor = source_coll.find({})

            for doc in cursor:
                stats["processed"] += 1

                try:
                    # Transform document
                    transformed = transform_func(doc, self.gemini)

                    if transformed:
                        # Add transformation metadata
                        transformed["transformed_at"] = datetime.utcnow().isoformat()
                        transformed["transformation_version"] = "1.0"

                        batch.append(transformed)
                        stats["transformed"] += 1

                        # Bulk insert when batch is full
                        if len(batch) >= batch_size:
                            self._bulk_upsert(target_coll, batch)
                            batch = []
                    else:
                        stats["skipped"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    logger.debug(f"Error transforming document: {e}")

                pbar.update(1)

            # Insert remaining batch
            if batch:
                self._bulk_upsert(target_coll, batch)

        return stats

    def _bulk_upsert(self, collection, documents: list[dict]) -> None:
        """Bulk upsert documents to collection.

        Args:
            collection: Target collection
            documents: List of documents to upsert
        """
        if not documents:
            return

        try:
            # Create upsert operations
            operations = []

            for doc in documents:
                # Use source_fhir_id as unique key
                fhir_id = doc.get("source_fhir_id")
                patient_id = doc.get("patient_id")

                if fhir_id:
                    filter_query = {"source_fhir_id": fhir_id}
                elif patient_id:
                    filter_query = {"patient_id": patient_id}
                else:
                    # Fallback: use entire document as filter (will insert)
                    filter_query = {"_id": doc.get("_id", {})}

                operations.append(UpdateOne(filter_query, {"$set": doc}, upsert=True))

            if operations:
                collection.bulk_write(operations, ordered=False)

        except BulkWriteError as e:
            logger.warning(f"Bulk write error: {e.details.get('nInserted', 0)} inserted")
        except Exception as e:
            logger.error(f"Bulk upsert error: {e}")

    def transform_all(self) -> dict[str, dict[str, int]]:
        """Transform all collections.

        Returns:
            Dictionary of statistics for each collection
        """
        all_stats = {}

        logger.info("Starting full transformation pipeline")

        for source_coll, (target_coll, transform_func) in self.transformations.items():
            # Check if source collection exists
            if source_coll not in self.db.list_collection_names():
                logger.info(f"Collection {source_coll} not found, skipping")
                continue

            try:
                stats = self.transform_collection(source_coll, target_coll, transform_func)

                all_stats[source_coll] = stats

                logger.info(
                    f"{source_coll}: {stats['transformed']} transformed, "
                    f"{stats['skipped']} skipped, {stats['errors']} errors"
                )

            except Exception as e:
                logger.error(f"Error transforming {source_coll}: {e}")
                all_stats[source_coll] = {"error": str(e)}

        return all_stats

    def create_indexes(self) -> None:
        """Create indexes on clean collections."""
        logger.info("Creating indexes on clean collections...")

        index_definitions = {
            "clean_allergies": [
                ("patient_id", {}),
                ("allergy_name", {}),
                ("category", {}),
            ],
            "clean_conditions": [
                ("patient_id", {}),
                ("condition_name", {}),
                ("status", {}),
            ],
            "clean_observations": [
                ("patient_id", {}),
                ("test_name", {}),
                ("observation_type", {}),
            ],
            "clean_medications": [
                ("patient_id", {}),
                ("medication_name", {}),
                ("status", {}),
            ],
            "clean_immunizations": [
                ("patient_id", {}),
                ("vaccine_name", {}),
            ],
            "clean_procedures": [
                ("patient_id", {}),
                ("procedure_name", {}),
            ],
            "clean_encounters": [
                ("patient_id", {}),
                ("encounter_type", {}),
            ],
            "clean_care_plans": [
                ("patient_id", {}),
                ("plan_name", {}),
            ],
            "clean_patients": [
                ("patient_id", {"unique": True}),
                ("last_name", {}),
            ],
            "clean_diagnostic_reports": [
                ("patient_id", {}),
                ("report_type", {}),
            ],
            "clean_claims": [
                ("patient_id", {}),
                ("claim_date", {}),
                ("status", {}),
            ],
            "clean_explanation_of_benefits": [
                ("patient_id", {}),
                ("created_date", {}),
                ("status", {}),
            ],
        }

        indexes_created = 0

        for collection_name, indexes in index_definitions.items():
            if collection_name in self.db.list_collection_names():
                collection = self.db[collection_name]

                for index_field, index_options in indexes:
                    try:
                        collection.create_index(index_field, **index_options)
                        indexes_created += 1
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Could not create index {index_field}: {e}")

        logger.info(f"Created {indexes_created} indexes")

    def print_summary(self, stats: dict[str, dict[str, int]]) -> None:
        """Print transformation summary with enrichment statistics.

        Args:
            stats: Statistics dictionary from transform_all()
        """
        print("\n" + "=" * 80)
        print("TRANSFORMATION SUMMARY".center(80))
        print("=" * 80 + "\n")

        total_processed = 0
        total_transformed = 0
        total_skipped = 0
        total_errors = 0

        for collection, coll_stats in stats.items():
            if "error" in coll_stats:
                print(f"❌ {collection}: ERROR - {coll_stats['error']}")
            else:
                processed = coll_stats.get("processed", 0)
                transformed = coll_stats.get("transformed", 0)
                skipped = coll_stats.get("skipped", 0)
                errors = coll_stats.get("errors", 0)

                total_processed += processed
                total_transformed += transformed
                total_skipped += skipped
                total_errors += errors

                print(f"✓ {collection}:")
                print(f"  Processed: {processed}")
                print(f"  Transformed: {transformed}")
                print(f"  Skipped: {skipped}")
                print(f"  Errors: {errors}")
                print()

        print("=" * 80)
        print("TOTALS:")
        print(f"  Processed: {total_processed:,}")
        print(f"  Transformed: {total_transformed:,}")
        print(f"  Skipped: {total_skipped:,}")
        print(f"  Errors: {total_errors:,}")
        print("=" * 80 + "\n")

        if self.gemini:
            print("GEMINI ENRICHMENT STATISTICS:")
            print("=" * 80)
            gemini_stats = self.gemini.get_stats()
            print(f"Total API requests made: {gemini_stats['total_requests']}")
            print(f"Conditions cached: {gemini_stats['conditions_cached']}")
            print(f"Medications cached: {gemini_stats['medications_cached']}")
            print(f"Procedures cached: {gemini_stats['procedures_cached']}")
            print(f"Observations cached: {gemini_stats['observations_cached']}")
            print("=" * 80 + "\n")
        print()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform FHIR data to clean medical records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform with Gemini enrichment (API key from .env file)
  python transform.py

  # Transform with Gemini enrichment (API key from command line)
  python transform.py --gemini-key YOUR_API_KEY

  # Transform without Gemini (faster, no enrichment)
  python transform.py --no-gemini

  # Transform specific collections only
  python transform.py --collections conditions observations

  # Custom database
  python transform.py --db-name my_healthcare_db

Note: Gemini API key is automatically loaded from .env file in project root
      (GEMINI_API_KEY or GOOGLE_API_KEY environment variable)
        """,
    )

    parser.add_argument("--host", default="localhost", help="MongoDB host (default: localhost)")
    parser.add_argument("--port", type=int, default=27017, help="MongoDB port (default: 27017)")
    parser.add_argument("--user", default="admin", help="MongoDB username (default: admin)")
    parser.add_argument("--password", default="mongopass123", help="MongoDB password")
    parser.add_argument(
        "--db-name", default="text_to_mongo_db", help="Database name (default: text_to_mongo_db)"
    )
    parser.add_argument(
        "--gemini-key", help="Google AI API key for Gemini (optional, also reads from .env file)"
    )
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini enrichment")
    parser.add_argument(
        "--collections", nargs="+", help="Specific collections to transform (optional)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Batch size for bulk operations (default: 500)"
    )

    args = parser.parse_args()

    # Load Gemini API key from .env file if not provided via command line
    if not args.gemini_key and not args.no_gemini:
        args.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if args.gemini_key:
            logger.info("Using Gemini API key from environment variable (.env file)")

    # Connect to MongoDB
    connection_string = (
        f"mongodb://{args.user}:{args.password}@{args.host}:{args.port}/"
        f"{args.db_name}?authSource=admin"
    )

    logger.info(f"Connecting to MongoDB at {args.host}:{args.port}/{args.db_name}")

    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)

        # Test connection
        client.admin.command("ping")
        logger.info("Connected to MongoDB")

    except Exception as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        sys.exit(1)

    # Initialize pipeline
    use_gemini = not args.no_gemini

    pipeline = FHIRTransformationPipeline(
        mongo_client=client,
        db_name=args.db_name,
        gemini_api_key=args.gemini_key,
        use_gemini=use_gemini,
    )

    # Transform collections
    try:
        if args.collections:
            # Transform specific collections
            all_stats = {}

            for collection in args.collections:
                if collection in pipeline.transformations:
                    target_coll, transform_func = pipeline.transformations[collection]

                    stats = pipeline.transform_collection(
                        collection, target_coll, transform_func, batch_size=args.batch_size
                    )

                    all_stats[collection] = stats
                else:
                    logger.warning(f"Unknown collection: {collection}")
        else:
            # Transform all collections
            all_stats = pipeline.transform_all()

        # Create indexes
        pipeline.create_indexes()

        # Print summary
        pipeline.print_summary(all_stats)

        logger.info("Transformation complete!")

    except KeyboardInterrupt:
        logger.warning("Transformation cancelled by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        client.close()


# HELPER FUNCTIONS FOR DATA EXTRACTION
# ============================================================================


def extract_text_from_coding(coding_obj: Any, prefer_display: bool = True) -> str:
    """Extract human-readable text from FHIR coding object.

    Args:
        coding_obj: FHIR coding object (can be nested)
        prefer_display: Prefer 'display' over 'text' field

    Returns:
        Human-readable text or empty string
    """
    if not isinstance(coding_obj, dict):
        return ""

    # Try 'text' field first (often most readable)
    if not prefer_display and coding_obj.get("text"):
        return str(coding_obj["text"]).strip()

    # Try 'coding' array
    coding_array = coding_obj.get("coding", [])
    if isinstance(coding_array, list) and coding_array:
        first_coding = coding_array[0]
        if isinstance(first_coding, dict):
            display = first_coding.get("display", "")
            if display:
                return str(display).strip()

    # Fallback to 'text' field
    if coding_obj.get("text"):
        return str(coding_obj["text"]).strip()

    return ""


def extract_reference_id(reference_str: str) -> str:
    """Extract ID from FHIR reference string.

    Args:
        reference_str: FHIR reference (e.g., "urn:uuid:123" or "Patient/123")

    Returns:
        Extracted ID or empty string
    """
    if not isinstance(reference_str, str):
        return ""

    # Handle urn:uuid: format
    if "urn:uuid:" in reference_str:
        return reference_str.split("urn:uuid:")[-1]

    # Handle Resource/id format
    if "/" in reference_str:
        return reference_str.split("/")[-1]

    return reference_str


def extract_display_from_reference(reference_obj: Any) -> str:
    """Extract display name from FHIR reference object.

    Args:
        reference_obj: FHIR reference object

    Returns:
        Display name or empty string
    """
    if not isinstance(reference_obj, dict):
        return ""

    return str(reference_obj.get("display", "")).strip()


def decode_base64_content(base64_str: str) -> str:
    """Decode base64 encoded content.

    Args:
        base64_str: Base64 encoded string

    Returns:
        Decoded text or empty string on error
    """
    try:
        decoded_bytes = base64.b64decode(base64_str)
        return decoded_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.debug(f"Base64 decode error: {e}")
        return ""


def clean_medical_text(text: str) -> str:
    """Clean and normalize medical text.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_medically_relevant(text: str) -> bool:
    """Quick check if text contains medically relevant information.

    Args:
        text: Text to check

    Returns:
        True if likely medically relevant
    """
    if not text or len(text) < 3:
        return False

    # Filter out pure codes/IDs
    if re.match(r"^[A-Z0-9\-_]+$", text):
        return False

    # Filter out URLs
    if text.startswith(("http://", "https://", "urn:")):
        return False

    # Filter out system identifiers
    if "terminology.hl7.org" in text or "snomed.info" in text:
        return False

    return True


# ============================================================================
# GEMINI LLM INTEGRATION - ENHANCED
# ============================================================================


class EnhancedGeminiEnricher:
    """Enhanced Gemini LLM client with comprehensive medical data enrichment."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """Initialize enhanced Gemini client with caching.

        Args:
            api_key: Google AI API key
            model: Gemini model name

        Raises:
            ImportError: If google.generativeai is not installed
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google.generativeai is not installed. "
                "Install it with: pip install google-generativeai"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.request_count = 0

        # Caching for common medical terms
        self.condition_cache: dict[str, dict[str, Any]] = {}
        self.medication_cache: dict[str, dict[str, Any]] = {}
        self.procedure_cache: dict[str, dict[str, Any]] = {}
        self.observation_cache: dict[str, dict[str, Any]] = {}

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time

        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    def _call_gemini(self, prompt: str) -> str | None:
        """Make rate-limited Gemini API call.

        Args:
            prompt: Prompt to send to Gemini

        Returns:
            Response text or None
        """
        try:
            self._rate_limit()
            response = self.model.generate_content(prompt)
            self.request_count += 1

            if response and response.text:
                return response.text.strip()

        except Exception as e:
            logger.debug(f"Gemini API error: {e}")

        return None

    # ========================================================================
    # PHASE 1: CRITICAL ENRICHMENTS (HIGHEST VALUE)
    # ========================================================================

    def extract_clinical_summary(self, base64_content: str) -> dict[str, Any] | None:
        """Extract comprehensive clinical information from diagnostic report.

        Returns:
            {
                'clinical_summary': str,
                'key_diagnoses': List[str],
                'medications_mentioned': List[str],
                'key_findings': List[str]
            }
        """
        decoded_text = decode_base64_content(base64_content)

        if not decoded_text or len(decoded_text) < 20:
            return None

        # For short reports, just clean and return
        if len(decoded_text) < 500:
            return {
                "clinical_summary": clean_medical_text(decoded_text),
                "key_diagnoses": [],
                "medications_mentioned": [],
                "key_findings": [],
            }

        prompt = f"""Extract structured clinical information from this medical report.

Report:
{decoded_text[:2500]}

Provide response in this EXACT JSON format (nothing else):
{{
  "clinical_summary": "Brief summary in 2-3 sentences",
  "key_diagnoses": ["diagnosis1", "diagnosis2"],
  "medications_mentioned": ["med1", "med2"],
  "key_findings": ["finding1", "finding2"]
}}

If any section has no information, use empty array. Keep summary under 200 words."""

        response = self._call_gemini(prompt)

        if response:
            try:
                # Try to parse JSON response
                cleaned = response.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Fallback: just return as summary
                return {
                    "clinical_summary": clean_medical_text(response[:500]),
                    "key_diagnoses": [],
                    "medications_mentioned": [],
                    "key_findings": [],
                }

        return None

    def enrich_condition(self, condition_name: str) -> dict[str, str] | None:
        """Enrich condition with description, category, and patient explanation.

        Returns:
            {
                'description': str,
                'category': str,
                'patient_explanation': str,
                'severity_indicator': str
            }
        """
        if not condition_name or len(condition_name) < 3:
            return None

        # Check cache
        if condition_name in self.condition_cache:
            return self.condition_cache[condition_name]

        prompt = f"""Provide clinical information about: {condition_name}

Return EXACT JSON format (nothing else):
{{
  "description": "One sentence clinical description (max 25 words)",
  "category": "Medical category (e.g., Cardiovascular, Metabolic, Respiratory)",
  "patient_explanation": "Simple explanation for patients (max 30 words)",
  "severity_indicator": "mild, moderate, or severe"
}}

Condition: {condition_name}"""

        response = self._call_gemini(prompt)

        if response:
            try:
                cleaned = response.replace("```json", "").replace("```", "").strip()
                enrichment = json.loads(cleaned)

                # Cache result
                self.condition_cache[condition_name] = enrichment
                return enrichment

            except json.JSONDecodeError:
                # Fallback: create simple enrichment
                enrichment = {
                    "description": response[:150] if response else None,
                    "category": "General",
                    "patient_explanation": response[:150] if response else None,
                    "severity_indicator": "moderate",
                }
                self.condition_cache[condition_name] = enrichment
                return enrichment

        return None

    def enrich_medication(self, medication_name: str) -> dict[str, Any] | None:
        """Enrich medication with indication, drug class, and side effects.

        Returns:
            {
                'indication': str,
                'drug_class': str,
                'common_side_effects': List[str],
                'patient_friendly_name': str
            }
        """
        if not medication_name or len(medication_name) < 3:
            return None

        # Check cache
        if medication_name in self.medication_cache:
            return self.medication_cache[medication_name]

        prompt = f"""Provide medication information for: {medication_name}

Return EXACT JSON format (nothing else):
{{
  "indication": "What condition(s) it treats (one sentence)",
  "drug_class": "Pharmacological class (3-5 words)",
  "common_side_effects": ["effect1", "effect2", "effect3"],
  "patient_friendly_name": "Common name or simple description"
}}

Medication: {medication_name}"""

        response = self._call_gemini(prompt)

        if response:
            try:
                cleaned = response.replace("```json", "").replace("```", "").strip()
                enrichment = json.loads(cleaned)

                self.medication_cache[medication_name] = enrichment
                return enrichment

            except json.JSONDecodeError:
                enrichment = {
                    "indication": response[:150] if response else None,
                    "drug_class": "Unknown",
                    "common_side_effects": [],
                    "patient_friendly_name": medication_name,
                }
                self.medication_cache[medication_name] = enrichment
                return enrichment

        return None

    # ========================================================================
    # PHASE 2: HIGH VALUE ENRICHMENTS
    # ========================================================================

    def interpret_observation(
        self, test_name: str, value: float, unit: str
    ) -> dict[str, str] | None:
        """Interpret lab result with clinical context.

        Returns:
            {
                'interpretation': str,
                'clinical_significance': str,
                'risk_level': str
            }
        """
        prompt = f"""Interpret this lab result:

Test: {test_name}
Value: {value} {unit}

Provide EXACT JSON format (nothing else):
{{
  "interpretation": "Normal/Elevated/Low and brief explanation (max 15 words)",
  "clinical_significance": "What this means clinically (max 25 words)",
  "risk_level": "low, moderate, or high"
}}

Test: {test_name}
Value: {value} {unit}"""

        response = self._call_gemini(prompt)

        if response:
            try:
                cleaned = response.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {
                    "interpretation": response[:100] if response else None,
                    "clinical_significance": response[:150] if response else None,
                    "risk_level": "moderate",
                }

        return None

    def enrich_procedure(self, procedure_name: str) -> dict[str, str] | None:
        """Enrich procedure with purpose and explanation.

        Returns:
            {
                'purpose': str,
                'category': str,
                'patient_explanation': str,
                'complexity': str
            }
        """
        if not procedure_name or len(procedure_name) < 3:
            return None

        # Check cache
        if procedure_name in self.procedure_cache:
            return self.procedure_cache[procedure_name]

        prompt = f"""Provide procedure information for: {procedure_name}

Return EXACT JSON format (nothing else):
{{
  "purpose": "Why this procedure is done (max 20 words)",
  "category": "Medical specialty (e.g., Cardiovascular, Orthopedic)",
  "patient_explanation": "Simple explanation (max 30 words)",
  "complexity": "low, moderate, or high"
}}

Procedure: {procedure_name}"""

        response = self._call_gemini(prompt)

        if response:
            try:
                cleaned = response.replace("```json", "").replace("```", "").strip()
                enrichment = json.loads(cleaned)
                self.procedure_cache[procedure_name] = enrichment
                return enrichment
            except json.JSONDecodeError:
                enrichment = {
                    "purpose": response[:150] if response else None,
                    "category": "General",
                    "patient_explanation": response[:150] if response else None,
                    "complexity": "moderate",
                }
                self.procedure_cache[procedure_name] = enrichment
                return enrichment

        return None

    def enrich_allergy(self, allergy_name: str) -> dict[str, Any] | None:
        """Enrich allergy with avoidance tips and cross-reactions.

        Returns:
            {
                'avoidance_tips': str,
                'cross_reactions': List[str],
                'common_symptoms': List[str]
            }
        """
        prompt = f"""Provide allergy management information for: {allergy_name}

Return EXACT JSON format (nothing else):
{{
  "avoidance_tips": "Brief guidance (max 30 words)",
  "cross_reactions": ["substance1", "substance2"],
  "common_symptoms": ["symptom1", "symptom2", "symptom3"]
}}

Allergy: {allergy_name}"""

        response = self._call_gemini(prompt)

        if response:
            try:
                cleaned = response.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {
                    "avoidance_tips": response[:150] if response else None,
                    "cross_reactions": [],
                    "common_symptoms": [],
                }

        return None

    # ========================================================================
    # PHASE 3: OPTIONAL ENRICHMENTS
    # ========================================================================

    def enrich_care_plan(self, plan_name: str, activities: list[str]) -> dict[str, Any] | None:
        """Enrich care plan with goals and outcomes."""
        prompt = f"""Provide care plan information:

Plan: {plan_name}
Activities: {", ".join(activities[:5])}

Return EXACT JSON format (nothing else):
{{
  "plan_description": "Brief description (max 25 words)",
  "expected_outcomes": ["outcome1", "outcome2", "outcome3"]
}}"""

        response = self._call_gemini(prompt)

        if response:
            try:
                cleaned = response.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {
                    "plan_description": response[:150] if response else None,
                    "expected_outcomes": [],
                }

        return None

    def enrich_immunization(self, vaccine_name: str) -> dict[str, str] | None:
        """Enrich vaccine with purpose and prevention info."""
        prompt = f"""Provide vaccine information for: {vaccine_name}

Return EXACT JSON format (nothing else):
{{
  "purpose": "What it prevents (max 15 words)",
  "prevents": ["disease1", "disease2"]
}}

Vaccine: {vaccine_name}"""

        response = self._call_gemini(prompt)

        if response:
            try:
                cleaned = response.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {"purpose": response[:100] if response else None, "prevents": []}

        return None

    def get_stats(self) -> dict[str, int]:
        """Get enrichment statistics."""
        return {
            "total_requests": self.request_count,
            "conditions_cached": len(self.condition_cache),
            "medications_cached": len(self.medication_cache),
            "procedures_cached": len(self.procedure_cache),
            "observations_cached": len(self.observation_cache),
        }

    # Legacy method for backward compatibility
    def validate_medical_relevance(self, text: str) -> bool:
        """Validate if text is medically relevant using Gemini.

        Args:
            text: Text to validate

        Returns:
            True if medically relevant
        """
        # Quick pre-filter
        if not is_medically_relevant(text):
            return False

        # For short, clear medical terms, skip LLM validation
        if len(text) < 100 and any(
            term in text.lower()
            for term in [
                "diabetes",
                "hypertension",
                "infection",
                "allergy",
                "medication",
                "procedure",
                "surgery",
                "diagnosis",
                "treatment",
                "symptom",
            ]
        ):
            return True

        return True  # Default to accepting if basic filters pass

    # Legacy method for backward compatibility
    def enrich_condition_description(self, condition_name: str) -> str | None:
        """Legacy method: Enrich condition with brief clinical description.

        Args:
            condition_name: Name of medical condition

        Returns:
            Brief clinical description or None
        """
        enrichment = self.enrich_condition(condition_name)
        if enrichment and enrichment.get("description"):
            return enrichment["description"]
        return None


# Alias for backward compatibility
GeminiEnricher = EnhancedGeminiEnricher


# ============================================================================
# TRANSFORMATION FUNCTIONS FOR EACH RESOURCE TYPE
# ============================================================================


def transform_allergy(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform AllergyIntolerance to clean Allergy model with Gemini enrichment.

    Args:
        doc: Raw FHIR AllergyIntolerance document
        gemini: Optional Gemini enricher

    Returns:
        Clean allergy dictionary or None
    """
    allergy_name = extract_text_from_coding(doc.get("code", {}))

    if not allergy_name or not is_medically_relevant(allergy_name):
        return None

    # Extract category
    categories = doc.get("category", [])
    category = categories[0] if categories else "unknown"

    # Map category
    category_map = {
        "food": "food",
        "medication": "medication",
        "environment": "environment",
        "biologic": "biologic",
    }
    category = category_map.get(category, "environment")

    # Extract severity
    criticality = doc.get("criticality", "low")
    severity_map = {"low": "low", "high": "high", "unable-to-assess": "moderate"}
    severity = severity_map.get(criticality, "moderate")

    # Extract status
    status_obj = doc.get("clinicalStatus", {})
    status_code = "active"
    if isinstance(status_obj, dict):
        coding = status_obj.get("coding", [])
        if coding and isinstance(coding, list):
            status_code = coding[0].get("code", "active")

    status_map = {"active": "active", "inactive": "inactive", "resolved": "resolved"}
    status = status_map.get(status_code, "active")

    # Extract dates
    recorded_date = doc.get("recordedDate", "")

    # Extract patient reference
    patient_ref = doc.get("patient", {})
    patient_id = extract_reference_id(patient_ref.get("reference", ""))

    result = {
        "allergy_name": allergy_name,
        "category": category,
        "severity": severity,
        "status": status,
        "recorded_date": recorded_date,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }

    # Add Gemini enrichment if available
    if gemini and len(allergy_name) < 200:
        enrichment = gemini.enrich_allergy(allergy_name)
        if enrichment:
            result.update(enrichment)  # Merge enrichment fields

    return result


def transform_condition(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform Condition to clean Condition model with Gemini enrichment."""
    condition_name = extract_text_from_coding(doc.get("code", {}))

    if not condition_name or not is_medically_relevant(condition_name):
        return None

    # Extract status
    status_obj = doc.get("clinicalStatus", {})
    status_code = "active"
    if isinstance(status_obj, dict):
        coding = status_obj.get("coding", [])
        if coding and isinstance(coding, list):
            status_code = coding[0].get("code", "active")

    status_map = {
        "active": "active",
        "inactive": "inactive",
        "resolved": "resolved",
        "recurrence": "active",
        "relapse": "active",
        "remission": "inactive",
    }
    status = status_map.get(status_code, "active")

    # Extract verification status
    verif_obj = doc.get("verificationStatus", {})
    verif_code = "confirmed"
    if isinstance(verif_obj, dict):
        coding = verif_obj.get("coding", [])
        if coding and isinstance(coding, list):
            verif_code = coding[0].get("code", "confirmed")

    # Extract dates
    onset_date = doc.get("onsetDateTime", "")
    recorded_date = doc.get("recordedDate", onset_date)

    # Extract patient reference
    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    result = {
        "condition_name": condition_name,
        "status": status,
        "onset_date": onset_date,
        "recorded_date": recorded_date,
        "verification_status": verif_code,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }

    # Add Gemini enrichment if available
    if gemini and len(condition_name) < 200:
        enrichment = gemini.enrich_condition(condition_name)
        if enrichment:
            result.update(enrichment)  # Merge enrichment fields

    return result


def transform_observation(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform Observation to clean LabResult or VitalSign model with Gemini enrichment."""
    test_name = extract_text_from_coding(doc.get("code", {}))

    if not test_name or not is_medically_relevant(test_name):
        return None

    # Extract value
    value_qty = doc.get("valueQuantity", {})
    if not isinstance(value_qty, dict):
        return None  # No quantitative value

    value = value_qty.get("value")
    unit = value_qty.get("unit", "")

    if value is None:
        return None

    # Extract status
    status = doc.get("status", "final")

    # Extract test date
    test_date = doc.get("effectiveDateTime", doc.get("issued", ""))

    # Extract patient reference
    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    # Determine if vital sign or lab result
    vital_keywords = [
        "blood pressure",
        "heart rate",
        "temperature",
        "weight",
        "height",
        "bmi",
        "respiratory rate",
        "oxygen",
    ]

    is_vital = any(keyword in test_name.lower() for keyword in vital_keywords)

    result = {
        "test_name": test_name,
        "value": float(value) if isinstance(value, int | float) else value,
        "unit": unit,
        "status": status,
        "test_date": test_date,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
        "observation_type": "vital_sign" if is_vital else "lab_result",
    }

    # Add Gemini enrichment if available (interpret lab results)
    if gemini and isinstance(value, int | float):
        interpretation = gemini.interpret_observation(test_name, value, unit)
        if interpretation:
            result.update(interpretation)  # Merge interpretation fields

    return result


def transform_medication_request(
    doc: dict, gemini: EnhancedGeminiEnricher | None = None
) -> dict | None:
    """Transform MedicationRequest to clean Medication model with Gemini enrichment.

    Extracts medication name from either coded or referenced medication.
    Rejects documents without valid medication names.

    Args:
        doc: Raw FHIR MedicationRequest document
        gemini: Optional Gemini enricher

    Returns:
        Clean medication dictionary or None
    """
    # Extract medication name from reference or code
    medication_name = ""

    # Try medicationCodeableConcept first (most common)
    med_code = doc.get("medicationCodeableConcept", {})
    if isinstance(med_code, dict):
        medication_name = extract_text_from_coding(med_code)

    # If not found, try medicationReference display field
    if not medication_name:
        med_ref = doc.get("medicationReference", {})
        if isinstance(med_ref, dict):
            medication_name = extract_display_from_reference(med_ref)

    # REJECT if no human-readable medication name (per design principles)
    # Do NOT use UUIDs or IDs as medication names - reject documents without meaningful names
    if not medication_name or not is_medically_relevant(medication_name):
        return None

    # Additional validation: reject if medication_name looks like a UUID/ID
    # This ensures we only store human-readable medication names
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
    )
    if uuid_pattern.match(medication_name):
        return None  # Reject UUIDs as medication names - violates "Human-Readable Fields" principle

    # Extract status
    status = doc.get("status", "active")
    status_map = {
        "active": "active",
        "completed": "completed",
        "stopped": "stopped",
        "on-hold": "active",
        "cancelled": "stopped",
        "entered-in-error": "stopped",
    }
    status = status_map.get(status, "active")

    # Extract dates
    prescribed_date = doc.get("authoredOn", "")

    # Extract prescriber
    requester = doc.get("requester", {})
    prescriber = extract_display_from_reference(requester)

    # Extract patient reference
    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    result = {
        "medication_name": medication_name,
        "status": status,
        "prescribed_date": prescribed_date,
        "prescriber": prescriber if prescriber else None,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }

    # Add Gemini enrichment if available
    if gemini and len(medication_name) < 200:
        enrichment = gemini.enrich_medication(medication_name)
        if enrichment:
            result.update(enrichment)  # Merge enrichment fields

    return result


def transform_immunization(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform Immunization to clean Immunization model with Gemini enrichment."""
    vaccine_code = doc.get("vaccineCode", {})
    vaccine_name = extract_text_from_coding(vaccine_code)

    if not vaccine_name or not is_medically_relevant(vaccine_name):
        return None

    # Extract dates
    admin_date = doc.get("occurrenceDateTime", "")

    # Extract location
    location_ref = doc.get("location", {})
    location = extract_display_from_reference(location_ref)

    # Extract status
    status = doc.get("status", "completed")

    # Extract patient reference
    patient_ref = doc.get("patient", {})
    patient_id = extract_reference_id(patient_ref.get("reference", ""))

    result = {
        "vaccine_name": vaccine_name,
        "administration_date": admin_date,
        "location": location if location else None,
        "status": status,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }

    # Add Gemini enrichment if available
    if gemini and len(vaccine_name) < 200:
        enrichment = gemini.enrich_immunization(vaccine_name)
        if enrichment:
            result.update(enrichment)  # Merge enrichment fields

    return result


def transform_procedure(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform Procedure to clean Procedure model with Gemini enrichment."""
    code = doc.get("code", {})
    procedure_name = extract_text_from_coding(code)

    if not procedure_name or not is_medically_relevant(procedure_name):
        return None

    # Extract status
    status = doc.get("status", "completed")

    # Extract dates
    performed_period = doc.get("performedPeriod", {})
    performed_date = performed_period.get("start", doc.get("performedDateTime", ""))
    end_date = performed_period.get("end")

    # Extract location
    location_ref = doc.get("location", {})
    location = extract_display_from_reference(location_ref)

    # Extract patient reference
    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    result = {
        "procedure_name": procedure_name,
        "status": status,
        "performed_date": performed_date,
        "end_date": end_date if end_date else None,
        "location": location if location else None,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }

    # Add Gemini enrichment if available
    if gemini and len(procedure_name) < 200:
        enrichment = gemini.enrich_procedure(procedure_name)
        if enrichment:
            result.update(enrichment)  # Merge enrichment fields

    return result


def transform_encounter(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform Encounter to clean Encounter model."""
    # Extract encounter type/reason
    type_array = doc.get("type", [])
    visit_reason = ""
    if type_array and isinstance(type_array, list):
        visit_reason = extract_text_from_coding(type_array[0])

    # Extract class code
    class_obj = doc.get("class", {})
    encounter_type = (
        class_obj.get("code", "ambulatory") if isinstance(class_obj, dict) else "ambulatory"
    )

    # Map encounter types
    type_map = {
        "AMB": "ambulatory",
        "EMER": "emergency",
        "IMP": "inpatient",
        "ACUTE": "inpatient",
        "NONAC": "ambulatory",
        "VR": "virtual",
    }
    encounter_type = type_map.get(encounter_type, "ambulatory")

    # Extract dates
    period = doc.get("period", {})
    start_date = period.get("start", "")
    end_date = period.get("end")

    # Extract location
    location_array = doc.get("location", [])
    location = ""
    if location_array and isinstance(location_array, list):
        location_obj = location_array[0].get("location", {})
        location = extract_display_from_reference(location_obj)

    # Extract provider
    participant_array = doc.get("participant", [])
    provider = ""
    if participant_array and isinstance(participant_array, list):
        individual = participant_array[0].get("individual", {})
        provider = extract_display_from_reference(individual)

    # Extract status
    status = doc.get("status", "finished")

    # Extract patient reference
    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    return {
        "encounter_type": encounter_type,
        "visit_reason": visit_reason if visit_reason else None,
        "start_date": start_date,
        "end_date": end_date if end_date else None,
        "location": location if location else None,
        "provider": provider if provider else None,
        "status": status,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


def transform_care_plan(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform CarePlan to clean CarePlan model with Gemini enrichment."""
    # Extract plan name from category
    category_array = doc.get("category", [])
    plan_name = ""

    for category in category_array:
        if isinstance(category, dict):
            text = extract_text_from_coding(category)
            if text and is_medically_relevant(text) and len(text) > 10:
                plan_name = text
                break

    if not plan_name:
        return None

    # Extract status
    status = doc.get("status", "active")
    status_map = {
        "active": "active",
        "completed": "inactive",
        "revoked": "inactive",
        "on-hold": "active",
    }
    status = status_map.get(status, "active")

    # Extract activities
    activities = []
    activity_array = doc.get("activity", [])

    for activity in activity_array:
        if not isinstance(activity, dict):
            continue

        detail = activity.get("detail", {})
        if not isinstance(detail, dict):
            continue

        activity_code = detail.get("code", {})
        activity_name = extract_text_from_coding(activity_code)

        if activity_name and is_medically_relevant(activity_name):
            activity_status = detail.get("status", "in-progress")
            location_obj = detail.get("location", {})
            location = extract_display_from_reference(location_obj)

            activities.append(
                {
                    "activity_name": activity_name,
                    "status": activity_status,
                    "location": location if location else None,
                }
            )

    if not activities:
        return None

    # Extract dates
    period = doc.get("period", {})
    start_date = period.get("start", "")
    end_date = period.get("end")

    # Extract patient reference
    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    result = {
        "plan_name": plan_name,
        "status": status,
        "activities": activities,
        "start_date": start_date,
        "end_date": end_date if end_date else None,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }

    # Add Gemini enrichment if available
    if gemini:
        activity_names = [a.get("activity_name") for a in activities]
        enrichment = gemini.enrich_care_plan(plan_name, activity_names)
        if enrichment:
            result.update(enrichment)  # Merge enrichment fields

    return result


def transform_patient(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform Patient to clean Patient model."""
    # Extract name
    name_array = doc.get("name", [])
    first_name = ""
    last_name = ""

    if name_array and isinstance(name_array, list):
        for name_obj in name_array:
            if not isinstance(name_obj, dict):
                continue

            use = name_obj.get("use", "official")
            if use == "official":
                given_names = name_obj.get("given", [])
                if given_names and isinstance(given_names, list):
                    first_name = given_names[0]

                last_name = name_obj.get("family", "")
                break

    if not first_name or not last_name:
        return None

    # Extract birth date
    birth_date = doc.get("birthDate", "")

    # Extract gender
    gender = doc.get("gender", "unknown")
    gender_map = {"male": "male", "female": "female", "other": "other", "unknown": "unknown"}
    gender = gender_map.get(gender, "unknown")

    # Extract address
    address_array = doc.get("address", [])
    address = None

    if address_array and isinstance(address_array, list):
        addr = address_array[0]
        if isinstance(addr, dict):
            line_array = addr.get("line", [])
            street = line_array[0] if line_array else None

            address = {
                "street": street,
                "city": addr.get("city", ""),
                "state": addr.get("state", ""),
                "postal_code": addr.get("postalCode", ""),
                "country": addr.get("country", "US"),
            }

    # Extract phone
    telecom_array = doc.get("telecom", [])
    phone = None

    for telecom in telecom_array:
        if isinstance(telecom, dict) and telecom.get("system") == "phone":
            phone = telecom.get("value")
            break

    # Extract race/ethnicity from extensions
    race = None
    ethnicity = None

    extensions = doc.get("extension", [])
    for ext in extensions:
        if not isinstance(ext, dict):
            continue

        url = ext.get("url", "")

        if "us-core-race" in url:
            text_ext = next((e for e in ext.get("extension", []) if e.get("url") == "text"), None)
            if text_ext:
                race = text_ext.get("valueString")

        elif "us-core-ethnicity" in url:
            text_ext = next((e for e in ext.get("extension", []) if e.get("url") == "text"), None)
            if text_ext:
                ethnicity = text_ext.get("valueString")

    # Extract marital status
    marital_status_obj = doc.get("maritalStatus", {})
    marital_status = extract_text_from_coding(marital_status_obj)

    # Extract language
    communication = doc.get("communication", [])
    language = "English"
    if communication and isinstance(communication, list):
        lang_obj = communication[0].get("language", {})
        lang_text = extract_text_from_coding(lang_obj)
        if lang_text:
            language = lang_text

    patient_id = doc.get("fhir_id", "")

    return {
        "patient_id": patient_id,
        "first_name": first_name,
        "last_name": last_name,
        "birth_date": birth_date,
        "gender": gender,
        "address": address,
        "phone": phone,
        "race": race,
        "ethnicity": ethnicity,
        "language": language,
        "marital_status": marital_status if marital_status else None,
        "source_fhir_id": patient_id,
    }


def transform_diagnostic_report(
    doc: dict, gemini: EnhancedGeminiEnricher | None = None
) -> dict | None:
    """Transform DiagnosticReport - extract clinical summary with comprehensive Gemini enrichment."""
    # Extract presented form (base64 encoded report)
    presented_form = doc.get("presentedForm", [])

    if not presented_form or not isinstance(presented_form, list):
        return None

    extracted_data = None

    for form in presented_form:
        if not isinstance(form, dict):
            continue

        base64_data = form.get("data")
        if not base64_data:
            continue

        # Use Gemini to extract comprehensive clinical information
        if gemini:
            extracted_data = gemini.extract_clinical_summary(base64_data)
        else:
            # Fallback: just decode basic text
            decoded_text = decode_base64_content(base64_data)
            decoded_text = clean_medical_text(decoded_text)
            if decoded_text and len(decoded_text) >= 20:
                extracted_data = {
                    "clinical_summary": decoded_text,
                    "key_diagnoses": [],
                    "medications_mentioned": [],
                    "key_findings": [],
                }

        if extracted_data and extracted_data.get("clinical_summary"):
            break

    if (
        not extracted_data
        or not extracted_data.get("clinical_summary")
        or len(extracted_data["clinical_summary"]) < 20
    ):
        return None

    # Extract report type
    code = doc.get("code", {})
    report_type = extract_text_from_coding(code)

    # Extract date
    report_date = doc.get("effectiveDateTime", doc.get("issued", ""))

    # Extract patient reference
    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    return {
        "report_type": report_type if report_type else "Clinical Report",
        "clinical_summary": extracted_data["clinical_summary"],
        "key_diagnoses": extracted_data.get("key_diagnoses", []),
        "medications_mentioned": extracted_data.get("medications_mentioned", []),
        "key_findings": extracted_data.get("key_findings", []),
        "report_date": report_date,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


def transform_claim(doc: dict, gemini: EnhancedGeminiEnricher | None = None) -> dict | None:
    """Transform Claim to clean Claim model.

    Args:
        doc: Raw FHIR Claim document
        gemini: Optional Gemini enricher (not used for claims)

    Returns:
        Clean claim dictionary or None
    """
    # Extract and validate patient reference
    patient_ref = doc.get("patient", {})
    patient_id = extract_reference_id(patient_ref.get("reference", ""))

    if not patient_id:
        return None

    # Extract claim date
    claim_date = doc.get("created", "")

    # Extract total cost
    total_obj = doc.get("total", {})
    total_cost = None
    if isinstance(total_obj, dict):
        value = total_obj.get("value")
        if value is not None:
            total_cost = float(value) if isinstance(value, int | float) else value

    if total_cost is None:
        return None  # Reject claims without valid cost

    # Extract claim status
    status = doc.get("status", "active")

    # Extract services from items
    services = []
    items = doc.get("item", [])

    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue

            # Extract service name from productOrService
            service_code = item.get("productOrService", {})
            service_name = extract_text_from_coding(service_code)

            if service_name and is_medically_relevant(service_name):
                services.append({"service_name": service_name, "sequence": item.get("sequence", 0)})

    # Require at least one valid service
    if not services:
        return None

    # Extract claim type
    claim_type_obj = doc.get("type", {})
    claim_type = extract_text_from_coding(claim_type_obj)

    return {
        "patient_id": patient_id,
        "claim_date": claim_date,
        "total_cost": total_cost,
        "status": status,
        "services": services,
        "claim_type": claim_type if claim_type else "professional",
        "source_fhir_id": doc.get("fhir_id", ""),
    }


def transform_explanation_of_benefit(
    doc: dict, gemini: EnhancedGeminiEnricher | None = None
) -> dict | None:
    """Transform ExplanationOfBenefit to clean EOB model.

    Args:
        doc: Raw FHIR ExplanationOfBenefit document
        gemini: Optional Gemini enricher (not used for EOB)

    Returns:
        Clean EOB dictionary or None
    """
    # Extract and validate patient reference
    patient_ref = doc.get("patient", {})
    patient_id = extract_reference_id(patient_ref.get("reference", ""))

    if not patient_id:
        return None

    # Extract EOB date
    created_date = doc.get("created", "")

    # Extract total amounts from totals array
    total_amount = None
    totals = doc.get("total", [])

    if isinstance(totals, list) and totals:
        # Get first total amount (submitted amount typically)
        first_total = totals[0]
        if isinstance(first_total, dict):
            amount_obj = first_total.get("amount", {})
            if isinstance(amount_obj, dict):
                value = amount_obj.get("value")
                if value is not None:
                    total_amount = float(value) if isinstance(value, int | float) else value

    if total_amount is None:
        return None  # Reject EOB without valid amount

    # Extract EOB status
    status = doc.get("status", "active")

    # Extract services from items
    services = []
    items = doc.get("item", [])

    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue

            # Extract service name from productOrService
            service_code = item.get("productOrService", {})
            service_name = extract_text_from_coding(service_code)

            if service_name and is_medically_relevant(service_name):
                services.append({"service_name": service_name, "sequence": item.get("sequence", 0)})

    # Require at least one valid service
    if not services:
        return None

    # Extract EOB type
    eob_type_obj = doc.get("type", {})
    eob_type = extract_text_from_coding(eob_type_obj)

    # Extract insurer information
    insurer = doc.get("insurer", {})
    insurer_name = extract_display_from_reference(insurer) if isinstance(insurer, dict) else ""

    return {
        "patient_id": patient_id,
        "created_date": created_date,
        "total_amount": total_amount,
        "status": status,
        "services": services,
        "eob_type": eob_type if eob_type else "professional",
        "insurer_name": insurer_name if insurer_name else None,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
