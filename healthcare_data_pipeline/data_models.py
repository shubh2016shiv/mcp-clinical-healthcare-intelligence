from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# ============================================================================
# ENUMS FOR CONTROLLED VOCABULARIES
# ============================================================================


class AllergyCategory(str, Enum):
    FOOD = "food"
    MEDICATION = "medication"
    ENVIRONMENT = "environment"
    BIOLOGIC = "biologic"


class AllergySeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    LIFE_THREATENING = "life_threatening"


class ClinicalStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RESOLVED = "resolved"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


# ============================================================================
# CORE MEDICAL ENTITIES
# ============================================================================


class Allergy(BaseModel):
    """Simplified allergy/intolerance record with only medically relevant data"""

    allergy_name: str = Field(description="Name of the allergen")
    category: AllergyCategory
    severity: AllergySeverity
    status: ClinicalStatus
    recorded_date: datetime
    patient_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "allergy_name": "Allergic disposition",
                "category": "environment",
                "severity": "low",
                "status": "active",
                "recorded_date": "1976-03-22T19:27:59Z",
                "patient_id": "patient_001",
            }
        }


class CarePlanActivity(BaseModel):
    """Individual care plan activity"""

    activity_name: str
    status: str
    location: str | None = None


class CarePlan(BaseModel):
    """Simplified care plan with essential treatment information"""

    plan_name: str
    status: ClinicalStatus
    activities: list[CarePlanActivity]
    start_date: datetime
    end_date: datetime | None = None
    condition_treated: str | None = None
    patient_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "plan_name": "Diabetes self management plan",
                "status": "active",
                "activities": [
                    {
                        "activity_name": "Diabetic diet",
                        "status": "in-progress",
                        "location": "Clinic",
                    },
                    {
                        "activity_name": "Exercise therapy",
                        "status": "in-progress",
                        "location": "Clinic",
                    },
                ],
                "start_date": "2004-02-02T14:13:15Z",
                "condition_treated": "Prediabetes",
                "patient_id": "patient_001",
            }
        }


class Condition(BaseModel):
    """Medical condition/diagnosis"""

    condition_name: str
    status: ClinicalStatus
    onset_date: datetime
    recorded_date: datetime
    verification_status: str = Field(description="confirmed, provisional, refuted")
    patient_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "condition_name": "Risk activity involvement",
                "status": "active",
                "onset_date": "1999-01-18T15:29:44Z",
                "recorded_date": "1999-01-18T15:29:44Z",
                "verification_status": "confirmed",
                "patient_id": "patient_001",
            }
        }


class LabResult(BaseModel):
    """Laboratory observation result"""

    test_name: str
    value: float
    unit: str
    reference_range: str | None = None
    status: str = Field(description="final, preliminary, corrected")
    test_date: datetime
    patient_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "test_name": "Hemoglobin A1c",
                "value": 5.96,
                "unit": "%",
                "reference_range": "4.0-6.0%",
                "status": "final",
                "test_date": "2016-02-15T14:13:15Z",
                "patient_id": "patient_001",
            }
        }


class VitalSign(BaseModel):
    """Vital signs observation"""

    vital_type: str = Field(description="blood_pressure, heart_rate, temperature, weight, etc.")
    value: float
    unit: str
    measurement_date: datetime
    patient_id: str


class Immunization(BaseModel):
    """Vaccination record"""

    vaccine_name: str
    administration_date: datetime
    location: str | None = None
    status: str = Field(description="completed, not-done")
    patient_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "vaccine_name": "Influenza, seasonal, injectable, preservative free",
                "administration_date": "2016-02-15T14:13:15Z",
                "location": "Family Health Clinic",
                "status": "completed",
                "patient_id": "patient_001",
            }
        }


class Medication(BaseModel):
    """Medication prescription/administration"""

    medication_name: str
    status: str = Field(description="active, completed, stopped")
    prescribed_date: datetime
    prescriber: str | None = None
    patient_id: str

    # Optional detailed drug information if available
    drug_class: str | None = None
    drug_subclass: str | None = None
    therapeutic_class: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "medication_name": "Metformin 500mg",
                "status": "active",
                "prescribed_date": "2017-08-09T08:30:32Z",
                "prescriber": "Dr. Smith",
                "drug_class": "Antidiabetic Agents",
                "patient_id": "patient_001",
            }
        }


class Procedure(BaseModel):
    """Medical procedure performed"""

    procedure_name: str
    status: str = Field(description="completed, in-progress, not-done")
    performed_date: datetime
    end_date: datetime | None = None
    location: str | None = None
    patient_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "procedure_name": "Intramuscular injection",
                "status": "completed",
                "performed_date": "2015-12-28T14:13:15Z",
                "end_date": "2015-12-28T14:29:31Z",
                "location": "Methodist Hospital",
                "patient_id": "patient_001",
            }
        }


class Encounter(BaseModel):
    """Healthcare encounter/visit"""

    encounter_type: str = Field(description="ambulatory, emergency, inpatient, etc.")
    visit_reason: str | None = None
    start_date: datetime
    end_date: datetime | None = None
    location: str | None = None
    provider: str | None = None
    status: str = Field(description="finished, in-progress, cancelled")
    patient_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "encounter_type": "ambulatory",
                "visit_reason": "Well child visit",
                "start_date": "1999-01-18T14:13:15Z",
                "end_date": "1999-01-18T14:32:15Z",
                "location": "Family Health Clinic",
                "provider": "Dr. Haag",
                "status": "finished",
                "patient_id": "patient_001",
            }
        }


class ClaimItem(BaseModel):
    """Individual service/item in a claim"""

    service_name: str
    cost: float | None = None


class Claim(BaseModel):
    """Insurance claim information"""

    claim_date: datetime
    total_cost: float
    currency: str = "USD"
    insurance_type: str
    services: list[ClaimItem]
    status: str
    patient_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "claim_date": "1999-01-18T14:32:15Z",
                "total_cost": 783.46,
                "currency": "USD",
                "insurance_type": "NO_INSURANCE",
                "services": [
                    {"service_name": "Well child visit", "cost": 500.00},
                    {"service_name": "Risk assessment", "cost": 283.46},
                ],
                "status": "active",
                "patient_id": "patient_001",
            }
        }


class Address(BaseModel):
    """Patient address"""

    street: str | None = None
    city: str
    state: str
    postal_code: str
    country: str = "US"


class Patient(BaseModel):
    """Core patient demographic and identification information"""

    patient_id: str
    first_name: str
    last_name: str
    birth_date: datetime
    gender: Gender

    # Contact information
    address: Address | None = None
    phone: str | None = None

    # Demographics
    race: str | None = None
    ethnicity: str | None = None
    language: str = "English"
    marital_status: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "patient_001",
                "first_name": "Agueda",
                "last_name": "Quitzon",
                "birth_date": "1981-11-30",
                "gender": "female",
                "address": {
                    "street": "288 Armstrong Junction Suite 6",
                    "city": "Dallas",
                    "state": "TX",
                    "postal_code": "75220",
                    "country": "US",
                },
                "phone": "555-751-3789",
                "race": "White",
                "ethnicity": "Not Hispanic or Latino",
                "language": "English",
                "marital_status": "Married",
            }
        }


# ============================================================================
# CONSOLIDATED PATIENT RECORD
# ============================================================================


class MedicalRecord(BaseModel):
    """Complete medical record for a patient with all relevant clinical data"""

    patient: Patient
    allergies: list[Allergy] = []
    conditions: list[Condition] = []
    medications: list[Medication] = []
    lab_results: list[LabResult] = []
    vital_signs: list[VitalSign] = []
    immunizations: list[Immunization] = []
    procedures: list[Procedure] = []
    care_plans: list[CarePlan] = []
    encounters: list[Encounter] = []
    claims: list[Claim] = []

    last_updated: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "description": "Comprehensive patient medical record with all clinical data"
        }


# ============================================================================
# DATA TRANSFORMATION RULES
# ============================================================================

"""
TRANSFORMATION RULES FROM RAW FHIR TO CLEAN MODEL:

1. ALLERGYINTOLERANCES → Allergy
   - Extract: code.text → allergy_name
   - Extract: category[0] → category
   - Extract: criticality → severity
   - Extract: clinicalStatus.coding[0].code → status
   - Extract: recordedDate → recorded_date
   - REJECT: All coding systems, URLs, meta, ingestion_metadata, nested references

2. CAREPLANS → CarePlan
   - Extract: category[1].text → plan_name
   - Extract: activity[].detail.code.text → activities[].activity_name
   - Extract: activity[].detail.location.display → activities[].location
   - Extract: period.start → start_date
   - REJECT: All URN references, coding systems, HTML div content

3. CONDITIONS → Condition
   - Extract: code.text → condition_name
   - Extract: clinicalStatus.coding[0].code → status
   - Extract: onsetDateTime → onset_date
   - REJECT: All coding systems, encounter references, meta profiles

4. OBSERVATIONS → LabResult / VitalSign
   - Extract: code.text → test_name
   - Extract: valueQuantity.value → value
   - Extract: valueQuantity.unit → unit
   - Extract: effectiveDateTime → test_date
   - REJECT: All LOINC codes, system URLs, issued timestamps

5. IMMUNIZATIONS → Immunization
   - Extract: vaccineCode.text → vaccine_name
   - Extract: occurrenceDateTime → administration_date
   - Extract: location.display → location
   - REJECT: CVX codes, encounter references, meta profiles

6. MEDICATIONREQUESTS → Medication
   - Extract medication name from reference or code
   - Extract: authoredOn → prescribed_date
   - Extract: requester.display → prescriber
   - REJECT: Category coding, meta profiles, encounter references

7. PROCEDURES → Procedure
   - Extract: code.text → procedure_name
   - Extract: performedPeriod.start → performed_date
   - Extract: performedPeriod.end → end_date
   - Extract: location.display → location
   - REJECT: SNOMED codes, meta profiles

8. ENCOUNTERS → Encounter
   - Extract: type[0].text → visit_reason
   - Extract: class.code → encounter_type (map AMB → ambulatory)
   - Extract: period.start → start_date
   - Extract: period.end → end_date
   - Extract: participant[0].individual.display → provider
   - REJECT: Identifier systems, meta profiles, location references

9. PATIENTS → Patient
   - Extract: name[0].given[0] → first_name
   - Extract: name[0].family → last_name
   - Extract: birthDate → birth_date
   - Extract: address[0] → address (flatten)
   - Extract: extension[race].text → race
   - Extract: extension[ethnicity].text → ethnicity
   - REJECT: SSN, driver's license, passport, multipleBirthBoolean, HTML div

10. CLAIMS → Claim
    - Extract: created → claim_date
    - Extract: total.value → total_cost
    - Extract: item[].productOrService.text → services[].service_name
    - REJECT: billablePeriod, diagnosis references, facility references

11. DRUGS → Medication (supplement data)
    - Use primary_drug_name, drug_class_l3, drug_subclass_l4, therapeutic_class_l2
    - Match with medication requests by drug name
    - REJECT: ingredient_rxcui, ingestion_metadata

12. DIAGNOSTICREPORTS → Extract text from presentedForm
    - Decode base64 data
    - Parse meaningful clinical notes
    - REJECT: Raw base64, LOINC codes, meta profiles

GENERAL RULES:
- NO URLs or system identifiers (http://, urn:uuid:)
- NO coding system references (http://snomed.info/sct, http://loinc.org)
- NO meta.profile arrays
- NO ingestion_metadata
- Extract human-readable text from deepest nested "text" or "display" fields
- Convert status codes to simple strings (active, completed, etc.)
- Flatten all nested structures
- Use LLM to generate missing meaningful descriptions if needed
"""
