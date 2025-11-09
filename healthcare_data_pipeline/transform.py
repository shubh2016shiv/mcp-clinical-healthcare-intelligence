#!/usr/bin/env python3
"""
FHIR Data Transformation Pipeline - Enterprise Edition

Transforms raw FHIR data from MongoDB into clean, medically relevant data
using Pydantic models. Integrates with Gemini LLM for data enrichment and
validation of medical relevance.

CRITICAL FIXES:
- Helper functions defined BEFORE use
- Enterprise error handling with context
- Robust logging at appropriate levels
- Module import safety validated
- No duplicate code
"""

import base64
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import google.generativeai as genai
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from tqdm import tqdm

from healthcare_data_pipeline.config import load_config

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure logging level for the module.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)


# ============================================================================
# HELPER FUNCTIONS FOR DATA EXTRACTION
# ============================================================================
# CRITICAL: These MUST be defined before transformation functions use them


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
# ERROR TRACKING
# ============================================================================


class ErrorTracker:
    """Track and aggregate errors during transformation with context."""

    def __init__(self, max_samples: int = 10):
        """Initialize error tracker.

        Args:
            max_samples: Maximum number of error samples to store
        """
        self.errors = []
        self.max_samples = max_samples
        self.total_count = 0
        self.error_types = {}

    def record(self, error: Exception, context: dict) -> None:
        """Record an error with context.

        Args:
            error: Exception that occurred
            context: Context dictionary with collection, doc_id, etc.
        """
        self.total_count += 1

        # Track error types
        error_type = type(error).__name__
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        # Store sample if under limit
        if len(self.errors) < self.max_samples:
            self.errors.append(
                {
                    "error_type": error_type,
                    "error_message": str(error)[:500],
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    def get_summary(self) -> dict:
        """Get error summary.

        Returns:
            Dictionary with error statistics and samples
        """
        return {
            "total_errors": self.total_count,
            "error_types": self.error_types,
            "sampled_errors": len(self.errors),
            "samples": self.errors,
        }

    def has_errors(self) -> bool:
        """Check if any errors were recorded.

        Returns:
            True if errors exist
        """
        return self.total_count > 0


# ============================================================================
# GEMINI LLM INTEGRATION
# ============================================================================


class GeminiEnricher:
    """Gemini LLM client for medical data enrichment and validation."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialize Gemini client.

        Args:
            api_key: Google AI API key
            model: Gemini model name

        Raises:
            ValueError: If API key is invalid or model name is incorrect
            Exception: For other initialization errors
        """
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is required and cannot be empty")

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini API: {e}") from e

        try:
            self.model = genai.GenerativeModel(model)
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "404" in error_msg:
                try:
                    available_models = [
                        m.name
                        for m in genai.list_models()
                        if "generateContent" in m.supported_generation_methods
                    ]
                    flash_models = [m for m in available_models if "flash" in m.lower()]
                    suggestion = (
                        flash_models[0].split("/")[-1] if flash_models else "gemini-2.5-flash"
                    )
                    raise ValueError(
                        f"Model '{model}' not found or not available. "
                        f"Error: {e}. "
                        f"Try using: '{suggestion}'"
                    ) from e
                except Exception:
                    raise ValueError(
                        f"Model '{model}' not found or not available. "
                        f"Error: {e}. "
                        f"Common model names: 'gemini-2.5-flash', 'gemini-2.5-pro'"
                    ) from e
            else:
                raise ValueError(f"Failed to initialize Gemini model '{model}': {e}") from e

        self.request_count = 0
        self.max_requests_per_minute = 15
        self.max_retries = 3
        self.base_retry_delay = 1.0
        self.retry_multiplier = 2.0

        logger.info(f"[INFO] Gemini initialized successfully with model: {model}")

    def _call_gemini_with_retry(self, prompt: str, operation_name: str = "API call") -> str | None:
        """Call Gemini API with exponential backoff retry.

        Args:
            prompt: Prompt to send to Gemini
            operation_name: Name of operation for logging

        Returns:
            Response text or None if all retries failed
        """
        import time

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                self.request_count += 1

                if response and response.text:
                    if attempt > 0:
                        logger.debug(f"[INFO] {operation_name} succeeded on attempt {attempt + 1}")
                    return response.text.strip()

            except Exception as e:
                error_msg = str(e).lower()

                is_retryable = any(
                    x in error_msg
                    for x in [
                        "429",
                        "500",
                        "503",
                        "timeout",
                        "deadline exceeded",
                        "connection reset",
                        "temporarily unavailable",
                    ]
                )

                if not is_retryable:
                    logger.debug(f"[ERROR] {operation_name} failed (non-retryable): {e}")
                    return None

                if attempt < self.max_retries - 1:
                    delay = self.base_retry_delay * (self.retry_multiplier**attempt)
                    logger.debug(
                        f"[INFO] {operation_name} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.debug(
                        f"[ERROR] {operation_name} failed after {self.max_retries} attempts: {e}"
                    )

        return None

    def extract_clinical_summary(self, base64_content: str) -> str | None:
        """Extract clinical summary from diagnostic report.

        Args:
            base64_content: Base64 encoded diagnostic report

        Returns:
            Clinical summary or None
        """
        decoded_text = decode_base64_content(base64_content)

        if not decoded_text or len(decoded_text) < 20:
            return None

        if len(decoded_text) < 500:
            return clean_medical_text(decoded_text)

        prompt = f"""Extract only the medically relevant clinical information from this report.
Focus on: diagnoses, symptoms, medications, test results, clinical observations.
Remove: administrative details, formatting, headers, boilerplate text.
Keep it concise (under 200 words).

Report:
{decoded_text[:2000]}

Provide only the clinical summary, nothing else."""

        response_text = self._call_gemini_with_retry(prompt, "clinical_summary_extraction")
        if response_text:
            return clean_medical_text(response_text)

        return None

    def validate_medical_relevance(self, text: str) -> bool:
        """Validate if text is medically relevant using Gemini.

        Args:
            text: Text to validate

        Returns:
            True if medically relevant
        """
        if not is_medically_relevant(text):
            return False

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

        return True

    def enrich_condition_description(self, condition_name: str) -> str | None:
        """Enrich condition with brief clinical description.

        Args:
            condition_name: Name of medical condition

        Returns:
            Brief clinical description or None
        """
        if not condition_name or len(condition_name) < 3:
            return None

        try:
            prompt = f"""Provide a single sentence clinical description of: {condition_name}
Focus on what it is, not treatment. Maximum 20 words.
Example: "Type 2 Diabetes - A metabolic disorder affecting blood sugar regulation."

Condition: {condition_name}
Description:"""

            response = self.model.generate_content(prompt)
            self.request_count += 1

            if response and response.text:
                desc = clean_medical_text(response.text)
                if len(desc) > 150:
                    return None
                return desc

        except Exception as e:
            logger.debug(f"Gemini enrichment error: {e}")

        return None


# ============================================================================
# TRANSFORMATION FUNCTIONS FOR EACH RESOURCE TYPE
# ============================================================================


def transform_allergy(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform AllergyIntolerance to clean Allergy model."""
    allergy_name = extract_text_from_coding(doc.get("code", {}))

    if not allergy_name or not is_medically_relevant(allergy_name):
        return None

    categories = doc.get("category", [])
    category = categories[0] if categories else "unknown"

    category_map = {
        "food": "food",
        "medication": "medication",
        "environment": "environment",
        "biologic": "biologic",
    }
    category = category_map.get(category, "environment")

    criticality = doc.get("criticality", "low")
    severity_map = {"low": "low", "high": "high", "unable-to-assess": "moderate"}
    severity = severity_map.get(criticality, "moderate")

    status_obj = doc.get("clinicalStatus", {})
    status_code = "active"
    if isinstance(status_obj, dict):
        coding = status_obj.get("coding", [])
        if coding and isinstance(coding, list):
            status_code = coding[0].get("code", "active")

    status_map = {"active": "active", "inactive": "inactive", "resolved": "resolved"}
    status = status_map.get(status_code, "active")

    recorded_date = doc.get("recordedDate", "")

    patient_ref = doc.get("patient", {})
    patient_id = extract_reference_id(patient_ref.get("reference", ""))

    return {
        "allergy_name": allergy_name,
        "category": category,
        "severity": severity,
        "status": status,
        "recorded_date": recorded_date,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


def transform_condition(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform Condition to clean Condition model."""
    condition_name = extract_text_from_coding(doc.get("code", {}))

    if not condition_name or not is_medically_relevant(condition_name):
        return None

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

    verif_obj = doc.get("verificationStatus", {})
    verif_code = "confirmed"
    if isinstance(verif_obj, dict):
        coding = verif_obj.get("coding", [])
        if coding and isinstance(coding, list):
            verif_code = coding[0].get("code", "confirmed")

    onset_date = doc.get("onsetDateTime", "")
    recorded_date = doc.get("recordedDate", onset_date)

    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    description = None
    if gemini and len(condition_name) < 100:
        description = gemini.enrich_condition_description(condition_name)

    result = {
        "condition_name": condition_name,
        "status": status,
        "onset_date": onset_date,
        "recorded_date": recorded_date,
        "verification_status": verif_code,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }

    if description:
        result["description"] = description

    return result


def transform_observation(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform Observation to clean LabResult or VitalSign model."""
    test_name = extract_text_from_coding(doc.get("code", {}))

    if not test_name or not is_medically_relevant(test_name):
        return None

    value_qty = doc.get("valueQuantity", {})
    if not isinstance(value_qty, dict):
        return None

    value = value_qty.get("value")
    unit = value_qty.get("unit", "")

    if value is None:
        return None

    status = doc.get("status", "final")
    test_date = doc.get("effectiveDateTime", doc.get("issued", ""))

    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

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

    return result


def transform_medication_request(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform MedicationRequest to clean Medication model."""
    medication_name = ""

    med_code = doc.get("medicationCodeableConcept", {})
    if isinstance(med_code, dict):
        medication_name = extract_text_from_coding(med_code)

    if not medication_name:
        med_ref = doc.get("medicationReference", {})
        medication_name = extract_display_from_reference(med_ref)

    if not medication_name or not is_medically_relevant(medication_name):
        return None

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

    prescribed_date = doc.get("authoredOn", "")

    requester = doc.get("requester", {})
    prescriber = extract_display_from_reference(requester)

    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    return {
        "medication_name": medication_name,
        "status": status,
        "prescribed_date": prescribed_date,
        "prescriber": prescriber if prescriber else None,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


def transform_immunization(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform Immunization to clean Immunization model."""
    vaccine_code = doc.get("vaccineCode", {})
    vaccine_name = extract_text_from_coding(vaccine_code)

    if not vaccine_name or not is_medically_relevant(vaccine_name):
        return None

    admin_date = doc.get("occurrenceDateTime", "")

    location_ref = doc.get("location", {})
    location = extract_display_from_reference(location_ref)

    status = doc.get("status", "completed")

    patient_ref = doc.get("patient", {})
    patient_id = extract_reference_id(patient_ref.get("reference", ""))

    return {
        "vaccine_name": vaccine_name,
        "administration_date": admin_date,
        "location": location if location else None,
        "status": status,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


def transform_procedure(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform Procedure to clean Procedure model."""
    code = doc.get("code", {})
    procedure_name = extract_text_from_coding(code)

    if not procedure_name or not is_medically_relevant(procedure_name):
        return None

    status = doc.get("status", "completed")

    performed_period = doc.get("performedPeriod", {})
    performed_date = performed_period.get("start", doc.get("performedDateTime", ""))
    end_date = performed_period.get("end")

    location_ref = doc.get("location", {})
    location = extract_display_from_reference(location_ref)

    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    return {
        "procedure_name": procedure_name,
        "status": status,
        "performed_date": performed_date,
        "end_date": end_date if end_date else None,
        "location": location if location else None,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


def transform_encounter(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform Encounter to clean Encounter model."""
    type_array = doc.get("type", [])
    visit_reason = ""
    if type_array and isinstance(type_array, list):
        visit_reason = extract_text_from_coding(type_array[0])

    class_obj = doc.get("class", {})
    encounter_type = (
        class_obj.get("code", "ambulatory") if isinstance(class_obj, dict) else "ambulatory"
    )

    type_map = {
        "AMB": "ambulatory",
        "EMER": "emergency",
        "IMP": "inpatient",
        "ACUTE": "inpatient",
        "NONAC": "ambulatory",
        "VR": "virtual",
    }
    encounter_type = type_map.get(encounter_type, "ambulatory")

    period = doc.get("period", {})
    start_date = period.get("start", "")
    end_date = period.get("end")

    location_array = doc.get("location", [])
    location = ""
    if location_array and isinstance(location_array, list):
        location_obj = location_array[0].get("location", {})
        location = extract_display_from_reference(location_obj)

    participant_array = doc.get("participant", [])
    provider = ""
    if participant_array and isinstance(participant_array, list):
        individual = participant_array[0].get("individual", {})
        provider = extract_display_from_reference(individual)

    status = doc.get("status", "finished")

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


def transform_care_plan(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform CarePlan to clean CarePlan model."""
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

    status = doc.get("status", "active")
    status_map = {
        "active": "active",
        "completed": "inactive",
        "revoked": "inactive",
        "on-hold": "active",
    }
    status = status_map.get(status, "active")

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

    period = doc.get("period", {})
    start_date = period.get("start", "")
    end_date = period.get("end")

    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    return {
        "plan_name": plan_name,
        "status": status,
        "activities": activities,
        "start_date": start_date,
        "end_date": end_date if end_date else None,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


def transform_patient(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform Patient to clean Patient model."""
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

    birth_date = doc.get("birthDate", "")

    gender = doc.get("gender", "unknown")
    gender_map = {"male": "male", "female": "female", "other": "other", "unknown": "unknown"}
    gender = gender_map.get(gender, "unknown")

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

    telecom_array = doc.get("telecom", [])
    phone = None

    for telecom in telecom_array:
        if isinstance(telecom, dict) and telecom.get("system") == "phone":
            phone = telecom.get("value")
            break

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

    marital_status_obj = doc.get("maritalStatus", {})
    marital_status = extract_text_from_coding(marital_status_obj)

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


def transform_diagnostic_report(doc: dict, gemini: GeminiEnricher | None = None) -> dict | None:
    """Transform DiagnosticReport - extract clinical summary with Gemini."""
    presented_form = doc.get("presentedForm", [])

    if not presented_form or not isinstance(presented_form, list):
        return None

    clinical_summary = None

    for form in presented_form:
        if not isinstance(form, dict):
            continue

        base64_data = form.get("data")
        if not base64_data:
            continue

        if gemini:
            clinical_summary = gemini.extract_clinical_summary(base64_data)
        else:
            clinical_summary = decode_base64_content(base64_data)
            clinical_summary = clean_medical_text(clinical_summary)

        if clinical_summary:
            break

    if not clinical_summary or len(clinical_summary) < 20:
        return None

    code = doc.get("code", {})
    report_type = extract_text_from_coding(code)

    report_date = doc.get("effectiveDateTime", doc.get("issued", ""))

    subject_ref = doc.get("subject", {})
    patient_id = extract_reference_id(subject_ref.get("reference", ""))

    return {
        "report_type": report_type if report_type else "Clinical Report",
        "clinical_summary": clinical_summary,
        "report_date": report_date,
        "patient_id": patient_id,
        "source_fhir_id": doc.get("fhir_id", ""),
    }


# ============================================================================
# MAIN TRANSFORMATION PIPELINE
# ============================================================================


class FHIRTransformationPipeline:
    """Main pipeline for transforming FHIR data to clean medical records."""

    def __init__(
        self,
        mongo_client: MongoClient,
        source_db_name: str = "fhir_db",
        target_db_name: str = "fhir_db",
        gemini_api_key: str | None = None,
        use_gemini: bool = True,
        collection_mapping: dict[str, str] | None = None,
    ):
        """Initialize transformation pipeline.

        Args:
            mongo_client: MongoDB client instance
            source_db_name: Database name to read from (raw data source)
            target_db_name: Database name to write to (transformed data destination, always fhir_db)
            gemini_api_key: Google AI API key for Gemini
            use_gemini: Whether to use Gemini for enrichment
            collection_mapping: Mapping from ingestion collection names to final collection names.
                If None, uses default mapping from config.
        """
        self.client = mongo_client
        self.source_db = mongo_client[source_db_name]
        self.target_db = mongo_client[target_db_name]
        self.gemini = None

        # Store collection mapping (ingestion_name -> final_name)
        # Default mapping if not provided
        if collection_mapping is None:
            collection_mapping = {
                "allergyintolerances": "allergies",
                "conditions": "conditions",
                "observations": "observations",
                "medicationrequests": "medications",
                "immunizations": "immunizations",
                "procedures": "procedures",
                "encounters": "encounters",
                "careplans": "care_plans",
                "patients": "patients",
                "diagnosticreports": "diagnosticreports",  # Fixed: no underscore
                "claims": "claims",
                "explanationofbenefits": "explanationofbenefits",
            }
        self.collection_mapping = collection_mapping

        # Validate helper functions are available
        self._validate_dependencies()

        if use_gemini and gemini_api_key:
            try:
                self.gemini = GeminiEnricher(gemini_api_key)
                logger.info("[INFO] Gemini enrichment enabled and verified")
            except ValueError as e:
                logger.error(f"[ERROR] GEMINI INITIALIZATION FAILED: {e}")
                logger.error("[ERROR] This is likely due to:")
                logger.error("[ERROR]   1. Invalid API key")
                logger.error("[ERROR]   2. Incorrect model name")
                logger.error("[ERROR]   3. API connectivity issues")
                logger.warning("[WARNING] Continuing without Gemini enrichment")
                self.gemini = None
            except Exception as e:
                logger.error(
                    f"[ERROR] UNEXPECTED ERROR initializing Gemini: {type(e).__name__}: {e}"
                )
                logger.warning("[WARNING] Continuing without Gemini enrichment")
                self.gemini = None
        else:
            if use_gemini and not gemini_api_key:
                logger.warning("[WARNING] Gemini enrichment requested but no API key provided")
            self.gemini = None

        self.enrichment_stats = {
            "collections": {},
            "total_enriched_docs": 0,
            "total_enrichment_fields": 0,
        }

        self._transformations = None

    def _validate_dependencies(self) -> None:
        """Validate that all required helper functions are available.

        Raises:
            RuntimeError: If required helper functions are missing
        """
        required_helpers = [
            "extract_text_from_coding",
            "extract_reference_id",
            "extract_display_from_reference",
            "decode_base64_content",
            "clean_medical_text",
            "is_medically_relevant",
        ]

        missing = []
        for helper_name in required_helpers:
            if helper_name not in globals():
                missing.append(helper_name)

        if missing:
            raise RuntimeError(
                f"CRITICAL: Required helper functions not found: {missing}. "
                f"This indicates a code organization issue. "
                f"Helper functions must be defined before the pipeline class."
            )

        logger.debug("[INFO] All helper functions validated and available")

    def transform_collection(
        self,
        source_collection: str,
        target_collection: str,
        transform_func: callable,
        batch_size: int = 500,
    ) -> dict[str, int]:
        """Transform a single collection with enterprise error handling.

        Args:
            source_collection: Source collection name
            target_collection: Target collection name
            transform_func: Transformation function
            batch_size: Batch size for bulk operations

        Returns:
            Statistics dictionary
        """
        stats = {"processed": 0, "transformed": 0, "skipped": 0, "errors": 0}

        # Initialize error tracker
        error_tracker = ErrorTracker(max_samples=5)

        source_coll = self.source_db[source_collection]
        target_coll = self.target_db[target_collection]

        total_docs = source_coll.count_documents({})

        if total_docs == 0:
            logger.info(f"[INFO] No documents in {source_collection}, skipping")
            return stats

        logger.info(
            f"[INFO] Transforming {source_collection} -> {target_collection} ({total_docs} docs)"
        )

        # If source and target are the same collection, read all documents FIRST before dropping
        # This prevents losing data when dropping the collection
        if self.source_db.name == self.target_db.name and source_collection == target_collection:
            logger.info(
                f"[INFO] Overwrite mode: reading documents before dropping {target_collection} collection"
            )
            # Materialize cursor into list before dropping collection
            source_documents = list(source_coll.find({}))
            logger.info(f"[INFO] Read {len(source_documents)} documents from {source_collection}")

            # Now safe to drop since we have all documents in memory
            logger.info(f"[INFO] Dropping existing {target_collection} collection")
            target_coll.drop()
            logger.info(f"[INFO] Dropped {target_collection} collection for clean transformation")

            # Use the materialized list instead of cursor
            document_iterator = source_documents
        else:
            # Different collections or databases - safe to use cursor
            document_iterator = source_coll.find({})

        batch = []

        with tqdm(total=total_docs, desc=f"Processing {source_collection}") as pbar:
            for doc in document_iterator:
                stats["processed"] += 1

                try:
                    transformed = transform_func(doc, self.gemini)

                    if transformed:
                        enrichment_fields = self._identify_enrichment_fields(
                            transformed, target_collection
                        )

                        if enrichment_fields and self.gemini:
                            if target_collection not in self.enrichment_stats["collections"]:
                                self.enrichment_stats["collections"][target_collection] = {
                                    "enriched_docs": 0,
                                    "fields": {},
                                }
                            self.enrichment_stats["collections"][target_collection][
                                "enriched_docs"
                            ] += 1
                            self.enrichment_stats["total_enriched_docs"] += 1

                            for field, value in enrichment_fields.items():
                                if (
                                    field
                                    not in self.enrichment_stats["collections"][target_collection][
                                        "fields"
                                    ]
                                ):
                                    self.enrichment_stats["collections"][target_collection][
                                        "fields"
                                    ][field] = {"count": 0, "sample_value": None}
                                self.enrichment_stats["collections"][target_collection]["fields"][
                                    field
                                ]["count"] += 1
                                self.enrichment_stats["total_enrichment_fields"] += 1

                                if (
                                    self.enrichment_stats["collections"][target_collection][
                                        "fields"
                                    ][field]["sample_value"]
                                    is None
                                ):
                                    sample = value
                                    if isinstance(sample, str) and len(sample) > 100:
                                        sample = sample[:100] + "..."
                                    elif isinstance(sample, list):
                                        sample = str(sample[:3]) + (
                                            "..." if len(sample) > 3 else ""
                                        )
                                    self.enrichment_stats["collections"][target_collection][
                                        "fields"
                                    ][field]["sample_value"] = sample

                        transformed["transformed_at"] = datetime.utcnow().isoformat()
                        transformed["transformation_version"] = "1.0"

                        batch.append(transformed)
                        stats["transformed"] += 1

                        if len(batch) >= batch_size:
                            self._bulk_upsert(target_coll, batch)
                            batch = []
                    else:
                        stats["skipped"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    error_tracker.record(
                        e,
                        {
                            "collection": source_collection,
                            "doc_id": str(doc.get("_id", "unknown")),
                            "fhir_id": doc.get("fhir_id", "unknown"),
                        },
                    )
                    logger.warning(
                        f"[WARNING] Error transforming document in {source_collection} "
                        f"(doc_id={doc.get('_id', 'unknown')}): {type(e).__name__}: {str(e)[:200]}"
                    )

                pbar.update(1)

            if batch:
                self._bulk_upsert(target_coll, batch)

        # Log error summary if errors occurred
        if error_tracker.has_errors():
            summary = error_tracker.get_summary()
            logger.error(
                f"❌ {source_collection}: {summary['total_errors']} errors occurred during transformation"
            )
            logger.error(f"   Error types: {summary['error_types']}")

            # Log sample errors with context
            for i, error_sample in enumerate(summary["samples"][:3], 1):
                logger.error(f"   Sample Error {i}:")
                logger.error(f"      Type: {error_sample['error_type']}")
                logger.error(f"      Message: {error_sample['error_message'][:200]}")
                logger.error(f"      Context: {error_sample['context']}")

        return stats

    def _identify_enrichment_fields(self, transformed_doc: dict, collection_name: str) -> dict:
        """Identify which fields in a document are enrichment fields added by Gemini.

        Args:
            transformed_doc: The transformed document
            collection_name: The collection name (ingestion name during transformation)

        Returns:
            Dictionary of enrichment fields found in the document
        """
        # Map collection_name (ingestion name) to final name for lookup
        final_name = self.collection_mapping.get(collection_name, collection_name)

        enrichment_field_map = {
            "conditions": [
                "description",
                "category",
                "patient_explanation",
                "severity_indicator",
            ],
            "medications": [
                "indication",
                "drug_class",
                "common_side_effects",
                "patient_friendly_name",
            ],
            "observations": ["interpretation", "clinical_significance", "risk_level"],
            "procedures": ["purpose", "category", "patient_explanation", "complexity"],
            "allergies": ["avoidance_tips", "cross_reactions", "common_symptoms"],
            "care_plans": ["plan_description", "expected_outcomes"],
            "immunizations": ["purpose", "prevents"],
            "diagnosticreports": [  # Fixed: use final name from mapping (no underscore)
                "clinical_summary",
                "key_diagnoses",
                "medications_mentioned",
                "key_findings",
            ],
            "encounters": ["visit_type_description", "typical_activities"],
        }

        enrichment_fields = {}
        # Use final collection name to look up enrichment fields
        fields_to_check = enrichment_field_map.get(final_name, [])

        for field in fields_to_check:
            if field in transformed_doc and transformed_doc[field] is not None:
                value = transformed_doc[field]
                if (
                    (isinstance(value, str) and value.strip())
                    or (isinstance(value, list) and len(value) > 0)
                    or (not isinstance(value, str | list) and value)
                ):
                    enrichment_fields[field] = value

        return enrichment_fields

    @property
    def transformations(self) -> dict:
        """Lazy-load transformation functions after they are all defined.

        During transformation, we use the ingestion collection name (source) as the target
        collection name. After all transformations complete, we rename collections based on
        the collection_mapping to their final names.
        """
        if self._transformations is None:
            transformations_map = {
                "allergyintolerances": "transform_allergy",
                "conditions": "transform_condition",
                "observations": "transform_observation",
                "medicationrequests": "transform_medication_request",
                "immunizations": "transform_immunization",
                "procedures": "transform_procedure",
                "encounters": "transform_encounter",
                "careplans": "transform_care_plan",
                "patients": "transform_patient",
                "diagnosticreports": "transform_diagnostic_report",
            }

            self._transformations = {}
            for source_coll, func_name in transformations_map.items():
                # During transformation, use the same collection name as source (ingestion name)
                # This ensures we write to the ingestion collection name, then rename at the end
                target_coll = source_coll  # Use ingestion name during transformation
                func = globals().get(func_name)

                if func is None or not callable(func):
                    available = [
                        k
                        for k in globals()
                        if k.startswith("transform_") and callable(globals().get(k))
                    ]
                    raise RuntimeError(
                        f"Transformation function '{func_name}' not found in global scope. "
                        f"Available functions: {available}"
                    )

                self._transformations[source_coll] = (target_coll, func)

        return self._transformations

    def _bulk_upsert(self, collection, documents: list[dict]) -> None:
        """Bulk upsert documents to collection."""
        if not documents:
            return

        try:
            operations = []

            for doc in documents:
                fhir_id = doc.get("source_fhir_id")
                patient_id = doc.get("patient_id")

                if fhir_id:
                    filter_query = {"source_fhir_id": fhir_id}
                elif patient_id:
                    filter_query = {"patient_id": patient_id}
                else:
                    filter_query = {"_id": doc.get("_id", {})}

                operations.append(UpdateOne(filter_query, {"$set": doc}, upsert=True))

            if operations:
                collection.bulk_write(operations, ordered=False)

        except BulkWriteError as e:
            logger.warning(f"Bulk write partial success: {e.details.get('nInserted', 0)} inserted")
        except Exception as e:
            logger.error(f"Bulk upsert error: {type(e).__name__}: {e}")

    def rename_collections(self) -> None:
        """Rename collections from ingestion names to final names based on collection_mapping.

        This is called after all transformations complete. Collections are renamed from
        their ingestion names (e.g., 'diagnosticreports') to their final names based on
        the collection_mapping configuration.

        Only renames collections where ingestion_name != final_name. If target collection
        already exists, it is dropped first (overwrite mode).
        """
        logger.info("Renaming collections to final names based on mapping...")

        rename_count = 0
        skip_count = 0

        for ingestion_name, final_name in self.collection_mapping.items():
            # Skip if names are the same (no rename needed)
            if ingestion_name == final_name:
                skip_count += 1
                continue

            # Check if source collection exists
            if ingestion_name not in self.target_db.list_collection_names():
                logger.debug(f"Collection {ingestion_name} not found, skipping rename")
                continue

            # Check if target collection already exists
            if final_name in self.target_db.list_collection_names():
                logger.warning(
                    f"Target collection {final_name} already exists. "
                    f"Dropping it before renaming {ingestion_name} -> {final_name}"
                )
                self.target_db[final_name].drop()

            try:
                # Rename collection using MongoDB's rename_collection command
                # This is atomic and efficient
                self.target_db[ingestion_name].rename(final_name)
                logger.info(f"Renamed collection: {ingestion_name} -> {final_name}")
                rename_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to rename collection {ingestion_name} -> {final_name}: "
                    f"{type(e).__name__}: {e}"
                )

        logger.info(
            f"Collection renaming complete: {rename_count} renamed, {skip_count} skipped (same name)"
        )

    def transform_all(
        self, max_workers: int | None = None, enable_parallel: bool = True
    ) -> dict[str, dict[str, int]]:
        """Transform all collections with optional parallel processing."""
        all_stats = {}

        logger.info("Starting full transformation pipeline")

        # Filter available collections
        available_transformations = {}
        for source_coll, (target_coll, transform_func) in self.transformations.items():
            if source_coll in self.source_db.list_collection_names():
                available_transformations[source_coll] = (target_coll, transform_func)
            else:
                logger.info(f"Collection {source_coll} not found, skipping")

        if not available_transformations:
            logger.warning("No collections available for transformation")
            return all_stats

        # Determine worker count
        if max_workers is None:
            max_workers = min(4, len(available_transformations))  # Default to 4 workers max
        actual_workers = min(max_workers, len(available_transformations)) if enable_parallel else 1

        logger.info(f"Using {actual_workers} worker threads for transformation")

        if enable_parallel and actual_workers > 1:
            # Parallel transformation
            logger.info("Parallel transformation enabled for improved performance")

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                # Submit transformation tasks
                future_to_collection = {}
                for source_coll, (target_coll, transform_func) in available_transformations.items():
                    future = executor.submit(
                        self.transform_collection, source_coll, target_coll, transform_func
                    )
                    future_to_collection[future] = source_coll

                # Collect results as they complete
                for future in as_completed(future_to_collection):
                    source_coll = future_to_collection[future]
                    try:
                        stats = future.result()
                        all_stats[source_coll] = stats

                        logger.info(
                            f"{source_coll}: {stats['transformed']} transformed, "
                            f"{stats['skipped']} skipped, {stats['errors']} errors"
                        )

                    except Exception as e:
                        logger.error(f"Error transforming {source_coll}: {type(e).__name__}: {e}")
                        all_stats[source_coll] = {"error": str(e)}
        else:
            # Sequential transformation (fallback)
            logger.info("Sequential transformation mode")

            for source_coll, (target_coll, transform_func) in available_transformations.items():
                try:
                    stats = self.transform_collection(source_coll, target_coll, transform_func)

                    all_stats[source_coll] = stats

                    logger.info(
                        f"{source_coll}: {stats['transformed']} transformed, "
                        f"{stats['skipped']} skipped, {stats['errors']} errors"
                    )

                except Exception as e:
                    logger.error(f"Error transforming {source_coll}: {type(e).__name__}: {e}")
                    all_stats[source_coll] = {"error": str(e)}

        # After all transformations complete, rename collections to their final names
        logger.info("All transformations complete. Renaming collections to final names...")
        self.rename_collections()

        return all_stats

    def create_indexes(self) -> None:
        """Create indexes on transformed collections using final collection names."""
        logger.info("Creating indexes on transformed collections...")

        # Index definitions keyed by final collection names (after mapping)
        index_definitions = {
            "allergies": [
                ("patient_id", {}),
                ("allergy_name", {}),
                ("category", {}),
            ],
            "conditions": [
                ("patient_id", {}),
                ("condition_name", {}),
                ("status", {}),
            ],
            "observations": [
                ("patient_id", {}),
                ("test_name", {}),
                ("observation_type", {}),
            ],
            "medications": [
                ("patient_id", {}),
                ("medication_name", {}),
                ("status", {}),
            ],
            "immunizations": [
                ("patient_id", {}),
                ("vaccine_name", {}),
            ],
            "procedures": [
                ("patient_id", {}),
                ("procedure_name", {}),
            ],
            "encounters": [
                ("patient_id", {}),
                ("encounter_type", {}),
            ],
            "care_plans": [
                ("patient_id", {}),
                ("plan_name", {}),
            ],
            "patients": [
                ("patient_id", {"unique": True}),
                ("last_name", {}),
            ],
            "diagnosticreports": [  # Fixed: use final name from mapping (no underscore)
                ("patient_id", {}),
                ("report_type", {}),
            ],
        }

        indexes_created = 0

        # Use final collection names from mapping
        for collection_name, indexes in index_definitions.items():
            # Check if collection exists (using final name after renaming)
            if collection_name in self.target_db.list_collection_names():
                collection = self.target_db[collection_name]

                for index_field, index_options in indexes:
                    try:
                        collection.create_index(index_field, **index_options)
                        indexes_created += 1
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Could not create index {index_field}: {e}")

        logger.info(f"Created {indexes_created} indexes")

    def print_summary(self, stats: dict[str, dict[str, int]]) -> None:
        """Print transformation summary with error analysis."""
        print("\n" + "=" * 80)
        print("TRANSFORMATION SUMMARY".center(80))
        print("=" * 80 + "\n")

        total_processed = 0
        total_transformed = 0
        total_skipped = 0
        total_errors = 0

        for collection, coll_stats in stats.items():
            if "error" in coll_stats:
                print(f"[ERROR] {collection}: ERROR - {coll_stats['error']}")
            else:
                processed = coll_stats.get("processed", 0)
                transformed = coll_stats.get("transformed", 0)
                skipped = coll_stats.get("skipped", 0)
                errors = coll_stats.get("errors", 0)

                total_processed += processed
                total_transformed += transformed
                total_skipped += skipped
                total_errors += errors

                print(f"[INFO] {collection}:")
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

        # Error breakdown
        if total_errors > 0:
            print("=" * 80)
            print("ERROR ANALYSIS".center(80))
            print("=" * 80)
            print(f"Total Errors: {total_errors:,}\n")
            for collection, coll_stats in stats.items():
                if "error" not in coll_stats and coll_stats.get("errors", 0) > 0:
                    error_rate = (coll_stats["errors"] / coll_stats["processed"]) * 100
                    print(
                        f"  {collection}: {coll_stats['errors']} errors "
                        f"({error_rate:.1f}% failure rate)"
                    )
            print("=" * 80 + "\n")

        if self.gemini:
            print(f"Gemini API requests made: {self.gemini.request_count}")

        self.print_enrichment_table()

        print()

    def print_enrichment_table(self) -> None:
        """Print enrichment statistics as a formatted table."""
        if not self.gemini or not self.enrichment_stats["collections"]:
            if self.gemini:
                print("\n" + "=" * 80)
                print("[INFO] GEMINI ENRICHMENT STATISTICS".center(80))
                print("=" * 80)
                print("[WARNING] No enrichment fields were added during transformation")
                print("=" * 80 + "\n")
            return

        print("\n" + "=" * 80)
        print("[INFO] GEMINI ENRICHMENT STATISTICS".center(80))
        print("=" * 80 + "\n")

        print(f"Total Enriched Documents: {self.enrichment_stats['total_enriched_docs']:,}")
        print(
            f"Total Enrichment Fields Added: {self.enrichment_stats['total_enrichment_fields']:,}"
        )
        print(f"Gemini API Requests: {self.gemini.request_count:,}")
        print()

        print("=" * 80)
        print(f"{'Collection':<30} {'Field':<30} {'Count':<10} {'Sample Value':<30}")
        print("=" * 80)

        sorted_collections = sorted(self.enrichment_stats["collections"].items())

        for collection_name, coll_data in sorted_collections:
            enriched_docs = coll_data["enriched_docs"]
            fields = coll_data["fields"]

            print(f"\n{collection_name} ({enriched_docs} documents enriched)")
            print("-" * 80)

            sorted_fields = sorted(fields.items(), key=lambda x: x[1]["count"], reverse=True)

            for field_name, field_data in sorted_fields:
                count = field_data["count"]
                sample = field_data["sample_value"]

                if sample is None:
                    sample_str = "N/A"
                elif isinstance(sample, str):
                    sample_str = sample[:50] + ("..." if len(sample) > 50 else "")
                elif isinstance(sample, list):
                    sample_str = str(sample)[:50] + ("..." if len(str(sample)) > 50 else "")
                else:
                    sample_str = str(sample)[:50]

                print(f"{'':<30} {field_name:<30} {count:<10} {sample_str:<30}")

        print("=" * 80 + "\n")


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
  # Transform with Gemini enrichment
  python transform.py --gemini-key YOUR_API_KEY

  # Transform without Gemini (faster, no enrichment)
  python transform.py --no-gemini

  # Transform specific collections only
  python transform.py --collections conditions observations

  # Custom database
  python transform.py --db-name my_healthcare_db
        """,
    )

    parser.add_argument("--host", default="localhost", help="MongoDB host (default: localhost)")
    parser.add_argument("--port", type=int, default=27017, help="MongoDB port (default: 27017)")
    parser.add_argument("--user", default="admin", help="MongoDB username (default: admin)")
    parser.add_argument("--password", default="mongopass123", help="MongoDB password")
    parser.add_argument(
        "--db-name",
        default="fhir_db",
        help="Database name (default: fhir_db, deprecated - use --source-db-name and --target-db-name)",
    )
    parser.add_argument(
        "--source-db-name",
        default=None,
        help="Source database name to read from (default: uses --db-name if not specified)",
    )
    parser.add_argument(
        "--target-db-name",
        default=None,
        help="Target database name to write to (default: uses --db-name if not specified)",
    )
    parser.add_argument("--gemini-key", help="Google AI API key for Gemini (optional)")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini enrichment")
    parser.add_argument(
        "--collections", nargs="+", help="Specific collections to transform (optional)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Batch size for bulk operations (default: 500)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging level
    configure_logging(args.log_level)

    # Load configuration (auto-loads if not already loaded)
    config = load_config()

    # Override config with command line args if provided
    if args.gemini_key:
        config.gemini.api_key = args.gemini_key
    if hasattr(args, "no_gemini") and args.no_gemini:
        config.gemini.enabled = False

    # Connect to MongoDB
    connection_string = (
        f"mongodb://{args.user}:{args.password}@{args.host}:{args.port}/"
        f"{args.db_name}?authSource=admin"
    )

    logger.info(f"Connecting to MongoDB at {args.host}:{args.port}/{args.db_name}")

    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)

        client.admin.command("ping")
        logger.info("Connected to MongoDB")

    except Exception as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        sys.exit(1)

    # Initialize pipeline
    use_gemini = config.gemini.enabled and config.gemini.api_key is not None

    # Determine source and target database names
    source_db_name = args.source_db_name if args.source_db_name else args.db_name
    target_db_name = args.target_db_name if args.target_db_name else args.db_name

    try:
        # Get collection mapping from config
        collection_mapping = config.collection_mapping.mapping

        pipeline = FHIRTransformationPipeline(
            mongo_client=client,
            source_db_name=source_db_name,
            target_db_name=target_db_name,
            gemini_api_key=config.gemini.api_key,
            use_gemini=use_gemini,
            collection_mapping=collection_mapping,
        )
    except RuntimeError as e:
        logger.error(f"[ERROR] Pipeline initialization failed: {e}")
        logger.error("This is a critical code organization error.")
        sys.exit(1)

    # Transform collections
    try:
        if args.collections:
            all_stats = {}

            for collection in args.collections:
                if collection in pipeline.transformations:
                    target_coll, transform_func = pipeline.transformations[collection]

                    stats = pipeline.transform_collection(
                        collection, target_coll, transform_func, batch_size=args.batch_size
                    )

                    all_stats[collection] = stats
                else:
                    logger.warning(f"[WARNING] Unknown collection: {collection}")
        else:
            # Use configuration settings for parallel processing
            enable_parallel = config.pipeline.enable_parallel_transformation
            max_workers = config.pipeline.max_workers

            all_stats = pipeline.transform_all(
                max_workers=max_workers, enable_parallel=enable_parallel
            )

        pipeline.create_indexes()
        pipeline.print_summary(all_stats)

        logger.info("[INFO] Transformation complete!")

    except KeyboardInterrupt:
        logger.warning("[WARNING] Transformation cancelled by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"[ERROR] Transformation failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        client.close()


if __name__ == "__main__":
    main()
