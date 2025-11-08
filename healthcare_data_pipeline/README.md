# Healthcare Natural Language to MongoDB Query System

A portfolio project demonstrating synthetic healthcare data generation, ingestion into MongoDB, and natural language querying using MCP (Model Context Protocol) technologies with agentic AI workflows.

## 🎯 Project Overview

This project creates a complete healthcare data analytics pipeline:

1. **Data Generation**: Uses Synthea to generate realistic, correlated synthetic patient data (FHIR R4 format)
2. **Data Ingestion**: Automated ETL pipeline to load FHIR bundles into MongoDB collections
3. **Natural Language Queries**: MCP-based system to convert natural language to MongoDB queries
4. **Agentic AI Workflows**: AI agents that can explore and analyze healthcare data relationships

## 📊 Generated Healthcare Data

The system generates comprehensive, correlated patient data in FHIR R4 format. During ingestion, these resources are **automatically separated into dedicated MongoDB collections** based on their resource type for optimal organization and querying.

### Data Types Generated

- **Patients**: Demographics, contact information, identifiers → `patients` collection
- **Encounters**: Hospital visits, outpatient visits, emergency visits → `encounters` collection
- **Conditions**: Diagnoses (diabetes, hypertension, etc.) → `conditions` collection
- **Observations**: Lab results, vital signs, measurements → `observations` collection
- **Medications**: Prescriptions and medication requests → `medicationrequests` collection
- **Allergies**: Patient allergies and intolerances → `allergyintolerances` collection
- **Procedures**: Medical procedures performed → `procedures` collection
- **Immunizations**: Vaccination records → `immunizations` collection
- **CarePlans**: Treatment plans → `careplans` collection
- **DiagnosticReports**: Lab and diagnostic test reports → `diagnosticreports` collection
- **Claims**: Insurance claims data → `claims` collection
- **ExplanationOfBenefits**: Insurance benefit explanations → `explanationofbenefits` collection

### Data Correlation

All data is naturally correlated (e.g., diabetic patients have glucose observations and metformin prescriptions). Resources maintain relationships through FHIR references, allowing you to join data across collections using MongoDB's `$lookup` operator.

See the [MongoDB Collections](#-mongodb-collections) section below for detailed information about each collection's purpose and contents.

## 🏗️ Architecture

```
┌─────────────┐
│   Synthea   │ ──> Generates FHIR R4 Bundles
│  (Docker)   │     (Patient data with all resources)
└─────────────┘
       │
       ↓
┌─────────────────┐
│ Data Ingestion  │ ──> Parses & Separates by Resource Type
│   Pipeline      │     (Groups resources by resourceType)
└─────────────────┘
       │
       ↓
┌─────────────────────────────────────┐
│         MongoDB Collections          │
│  ┌──────────┐  ┌──────────┐         │
│  │ patients │  │encounters │  ...    │
│  └──────────┘  └──────────┘         │
│  ┌──────────┐  ┌──────────┐         │
│  │conditions│  │observations│        │
│  └──────────┘  └──────────┘         │
│  ┌──────────┐  ┌──────────┐         │
│  │medication│  │procedures│  ...   │
│  │ requests │  └──────────┘         │
│  └──────────┘                       │
│  (12+ Collections, Auto-Separated)  │
└─────────────────────────────────────┘
       │
       ↓
┌─────────────────┐
│ MCP AI Agents   │ ──> NL to MongoDB Query
└─────────────────┘
```

### Data Flow Details

1. **Synthea Generation**: Synthea Docker generates realistic patient data in FHIR R4 format. Each patient bundle contains multiple resource types (Patient, Encounter, Condition, Observation, MedicationRequest, etc.) all in a single JSON file.

2. **Automatic Separation**: The ingestion script (`ingest.py`) automatically:
   - Reads each FHIR bundle file
   - Extracts all resources from the bundle
   - Groups resources by their `resourceType` field
   - Maps each resource type to its corresponding MongoDB collection
   - Performs bulk inserts for optimal performance

3. **Collection Organization**: Resources are stored in separate collections based on their type, enabling:
   - Efficient querying (query only what you need)
   - Optimized indexing per collection
   - Clear data organization
   - Maintained FHIR resource relationships

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- Python 3.8+
- 4GB+ free disk space
- 8GB+ RAM recommended

### Option 1: Python-Based Pipeline (Recommended)

The easiest way to get started - generates data and ingests into MongoDB with a single command.

```bash
# Generate 100 synthetic patients and ingest into MongoDB
python healthcare_data_pipeline/pipeline.py 100 Massachusetts

# Generate different amounts:
python healthcare_data_pipeline/pipeline.py 500 California
python healthcare_data_pipeline/pipeline.py 50 Texas
```

This single command will:
1. Start MongoDB and Redis infrastructure
2. Generate synthetic patient data using Synthea Docker
3. Create FHIR R4 bundles in `synthea_output/output/fhir/`
4. **Automatically separate and ingest** FHIR resources into dedicated MongoDB collections:
   - `patients`, `encounters`, `conditions`, `observations`
   - `medicationrequests`, `allergyintolerances`, `procedures`
   - `immunizations`, `careplans`, `diagnosticreports`, `claims`, `explanationofbenefits`
5. Create optimized indexes for each collection
6. Verify data was loaded successfully
7. Display connection details and sample queries

**Note**: Resources are automatically grouped by type during ingestion. A single FHIR bundle containing multiple resource types (Patient, Condition, Observation, etc.) is automatically separated into the appropriate collections.

**Helpful Options:**
```bash
# Skip infrastructure startup (if already running)
python healthcare_data_pipeline/pipeline.py 100 Massachusetts --skip-infra

# Skip Synthea generation (use existing data)
python healthcare_data_pipeline/pipeline.py --skip-synthea

# Skip data verification
python healthcare_data_pipeline/pipeline.py 100 Massachusetts --skip-verify
```

### Option 2: Step-by-Step Manual Process

If you prefer more control or need to reuse infrastructure:

```bash
# 1. Start infrastructure (MongoDB and Redis)
python infrastructure/manage_infrastructure.py --start

# 2. Generate Synthea data using Docker
cd infrastructure
docker compose --profile data-generation run --rm synthea -p 100 Massachusetts

# 3. Ingest FHIR data into MongoDB
python healthcare_data_pipeline/ingest.py ./synthea_output/output/fhir

# 4. Verify and explore the data
docker exec -it text_to_mongo_db mongosh \
  -u admin -p mongopass123 \
  --authenticationDatabase admin \
  text_to_mongo_db
```

### Explore the Data

```bash
# Connect to MongoDB
docker exec -it text_to_mongo_db mongosh \
  -u admin -p mongopass123 \
  --authenticationDatabase admin \
  text_to_mongo_db

# Sample queries
db.patients.countDocuments()
db.conditions.find({"code.coding.display": /Diabetes/i}).limit(5)
db.observations.find().sort({effectiveDateTime: -1}).limit(10)
db.medicationrequests.aggregate([
  {$group: {_id: "$medicationCodeableConcept.coding.0.display", count: {$sum: 1}}},
  {$sort: {count: -1}},
  {$limit: 10}
])
```

## 📚 MongoDB Collections

After ingestion, FHIR resources are automatically separated into dedicated MongoDB collections based on their resource type. This organization enables efficient querying and maintains FHIR compliance.

### Collection Overview

| Collection | Purpose | Key Data | Example Count |
|------------|---------|----------|---------------|
| **`patients`** | Patient demographics and identifiers | Name, DOB, gender, address, identifiers | 100 |
| **`encounters`** | Healthcare visits and interactions | Hospital stays, clinic visits, ER visits, dates, types | 800+ |
| **`conditions`** | Medical diagnoses and health conditions | Diagnosis codes, clinical status, onset dates, severity | 500+ |
| **`observations`** | Lab results, vital signs, measurements | Blood pressure, glucose, temperature, lab values, timestamps | 5000+ |
| **`medicationrequests`** | Prescriptions and medication orders | Medication names, dosages, frequencies, prescribing dates | 1000+ |
| **`allergyintolerances`** | Patient allergies and intolerances | Allergen types, severity, reaction descriptions | 200+ |
| **`procedures`** | Medical procedures and surgeries | Procedure codes, dates performed, outcomes | 400+ |
| **`immunizations`** | Vaccination records | Vaccine types, administration dates, lot numbers | 800+ |
| **`careplans`** | Treatment plans and care coordination | Goals, activities, status, care team members | 300+ |
| **`diagnosticreports`** | Lab and diagnostic test reports | Test results, interpretations, ordering physicians | 600+ |
| **`claims`** | Insurance claims data | Claim numbers, billing codes, amounts, status | Variable |
| **`explanationofbenefits`** | Insurance benefit explanations | Coverage details, payments, adjustments | Variable |

### Detailed Collection Purposes

#### `patients` Collection
**Purpose**: Central repository for all patient demographic information.

**Contains**:
- Patient identifiers (MRN, SSN, etc.)
- Demographics (name, date of birth, gender)
- Contact information (address, phone, email)
- Administrative data (deceased status, managing organization)

**Use Cases**: Patient lookup, demographic analysis, population studies

#### `encounters` Collection
**Purpose**: Tracks all healthcare interactions and visits.

**Contains**:
- Encounter types (inpatient, outpatient, emergency, ambulatory)
- Visit dates and durations
- Location information (hospital, clinic, department)
- Classifications (admission type, discharge disposition)
- Associated providers and organizations

**Use Cases**: Utilization analysis, visit frequency, care coordination tracking

#### `conditions` Collection
**Purpose**: Stores all medical diagnoses and health conditions.

**Contains**:
- Condition codes (ICD-10, SNOMED CT)
- Clinical status (active, resolved, remission)
- Severity and category
- Onset dates and verification status
- Related body systems

**Use Cases**: Disease prevalence studies, chronic condition management, population health analytics

#### `observations` Collection
**Purpose**: Captures all clinical measurements, lab results, and vital signs.

**Contains**:
- Vital signs (blood pressure, heart rate, temperature, respiratory rate)
- Laboratory results (glucose, cholesterol, complete blood count, etc.)
- Physical measurements (height, weight, BMI)
- Assessment findings
- Timestamps and effective dates

**Use Cases**: Clinical monitoring, trend analysis, alert generation, quality metrics

#### `medicationrequests` Collection
**Purpose**: Records all medication prescriptions and orders.

**Contains**:
- Medication names and codes (RxNorm, NDC)
- Dosage instructions and frequencies
- Prescribing dates and physicians
- Medication status (active, completed, stopped)
- Patient instructions

**Use Cases**: Medication adherence tracking, drug interaction analysis, prescribing patterns

#### `allergyintolerances` Collection
**Purpose**: Documents patient allergies and medication intolerances.

**Contains**:
- Allergen types (medications, foods, environmental)
- Reaction descriptions and severity
- Onset dates and verification status
- Clinical significance

**Use Cases**: Allergy alerts, medication safety, clinical decision support

#### `procedures` Collection
**Purpose**: Tracks all medical procedures and surgical interventions.

**Contains**:
- Procedure codes (CPT, ICD-10-PCS)
- Procedure dates and locations
- Performing providers
- Outcomes and complications
- Procedure categories

**Use Cases**: Surgical outcomes analysis, procedure utilization, quality reporting

#### `immunizations` Collection
**Purpose**: Maintains comprehensive vaccination records.

**Contains**:
- Vaccine types and codes (CVX)
- Administration dates and lot numbers
- Route and site of administration
- Vaccine status and reactions
- Series information

**Use Cases**: Immunization compliance, public health reporting, outbreak management

#### `careplans` Collection
**Purpose**: Documents treatment plans and care coordination activities.

**Contains**:
- Care plan goals and objectives
- Planned activities and interventions
- Care team members and roles
- Status and categories
- Time periods and milestones

**Use Cases**: Care coordination, quality improvement, care gap analysis

#### `diagnosticreports` Collection
**Purpose**: Stores structured diagnostic test results and interpretations.

**Contains**:
- Report types (lab, imaging, pathology)
- Test results and values
- Clinical interpretations
- Ordering and performing providers
- Report dates and status

**Use Cases**: Test result tracking, diagnostic accuracy, clinical decision support

#### `claims` Collection
**Purpose**: Insurance claim information for billing and reimbursement.

**Contains**:
- Claim numbers and types
- Billing codes (CPT, HCPCS, ICD-10)
- Claim amounts and status
- Service dates and providers
- Insurance information

**Use Cases**: Billing analysis, reimbursement tracking, financial reporting

#### `explanationofbenefits` Collection
**Purpose**: Detailed insurance benefit explanations and payment information.

**Contains**:
- Coverage details and benefits
- Payment amounts and adjustments
- Claim adjudication results
- Patient responsibility
- Insurance plan information

**Use Cases**: Benefits analysis, cost tracking, insurance verification

### How Data Separation Works

The ingestion process automatically separates FHIR resources into these collections:

#### Process Flow

```
FHIR Bundle Files (JSON)
    ↓
Read Each Bundle File
    ↓
Extract All Resources from Bundle
    ↓
Group Resources by resourceType
    ↓
Map to MongoDB Collections
    ↓
Bulk Insert into Collections
```

#### FHIR Bundle Structure

Each Synthea-generated file is a **FHIR Bundle** containing multiple resources:

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "patient-123",
        "name": [...],
        ...
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "id": "condition-456",
        "code": {...},
        ...
      }
    },
    {
      "resource": {
        "resourceType": "MedicationRequest",
        "id": "med-789",
        ...
      }
    }
    // ... more resources
  ]
}
```

#### Resource Type Mapping

The ingestion script uses this mapping to automatically route resources to collections:

```python
resource_collections = {
    "Patient": "patients",
    "Encounter": "encounters",
    "Condition": "conditions",
    "Observation": "observations",
    "MedicationRequest": "medicationrequests",
    "AllergyIntolerance": "allergyintolerances",
    "Procedure": "procedures",
    "Immunization": "immunizations",
    "CarePlan": "careplans",
    "DiagnosticReport": "diagnosticreports",
    "Claim": "claims",
    "ExplanationOfBenefit": "explanationofbenefits"
}
```

#### Example: One Bundle → Multiple Collections

A single FHIR bundle file might contain:
- 1 Patient resource
- 5 Encounter resources
- 3 Condition resources
- 20 Observation resources
- 8 MedicationRequest resources
- 2 AllergyIntolerance resources

After processing, these are automatically inserted into:
- `patients` collection: 1 document
- `encounters` collection: 5 documents
- `conditions` collection: 3 documents
- `observations` collection: 20 documents
- `medicationrequests` collection: 8 documents
- `allergyintolerances` collection: 2 documents

#### Benefits of Collection Separation

1. **Organized Data**: Each type of healthcare data is in its own collection
2. **Efficient Queries**: Query only the collection you need (e.g., only `observations` for lab results)
3. **Scalable**: Easy to add indexes per collection for optimal performance
4. **FHIR Compliant**: Maintains FHIR resource structure and standards
5. **Relationships Preserved**: Resources reference each other via IDs for data integrity

#### Data Relationships

Resources are linked via references. For example:

- A `Condition` has `subject.reference: "Patient/123"` → links to patient
- An `Observation` has `subject.reference: "Patient/123"` → links to patient
- A `MedicationRequest` has `subject.reference: "Patient/123"` → links to patient
- An `Encounter` has `subject.reference: "Patient/123"` → links to patient

You can use MongoDB's `$lookup` to join related data across collections:

```javascript
// Get patient with all their conditions
db.patients.aggregate([
  {
    $lookup: {
      from: "conditions",
      localField: "id",
      foreignField: "subject.reference",
      as: "conditions"
    }
  },
  {
    $lookup: {
      from: "medicationrequests",
      localField: "id",
      foreignField: "subject.reference",
      as: "medications"
    }
  }
])
```

This gives you a complete patient record with all their conditions, medications, and other related data!

## 🔍 Sample Use Cases

### 1. Find Diabetic Patients

```javascript
db.conditions.find({
  "code.coding.display": /Diabetes/i
})
```

### 2. Patients with Multiple Chronic Conditions

```javascript
db.conditions.aggregate([
  { $group: {
      _id: "$subject.reference",
      conditionCount: { $sum: 1 },
      conditions: { $push: "$code.coding.display" }
  }},
  { $match: { conditionCount: { $gte: 3 } } },
  { $sort: { conditionCount: -1 } }
])
```

### 3. Most Prescribed Medications

```javascript
db.medicationrequests.aggregate([
  { $unwind: "$medicationCodeableConcept.coding" },
  { $group: {
      _id: "$medicationCodeableConcept.coding.display",
      count: { $sum: 1 }
  }},
  { $sort: { count: -1 } },
  { $limit: 20 }
])
```

### 4. Patient Health Summary

```javascript
db.patients.aggregate([
  {
    $lookup: {
      from: "conditions",
      localField: "id",
      foreignField: "subject.reference",
      as: "conditions"
    }
  },
  {
    $lookup: {
      from: "medicationrequests",
      localField: "id",
      foreignField: "subject.reference",
      as: "medications"
    }
  },
  {
    $project: {
      name: 1,
      gender: 1,
      birthDate: 1,
      conditionCount: { $size: "$conditions" },
      medicationCount: { $size: "$medications" }
    }
  }
])
```

## 🤖 Natural Language Query Examples (Future MCP Integration)

Once you integrate MCP, users can query using natural language:

- "Show me all diabetic patients"
- "What are the most common diagnoses?"
- "Find patients taking blood pressure medication"
- "Show recent lab results for glucose tests"
- "Which patients have multiple chronic conditions?"
- "List all medications prescribed for hypertension"

## 🛠️ Management Commands

### Infrastructure Management

```bash
# Start infrastructure (MongoDB and Redis)
python infrastructure/manage_infrastructure.py --start

# Stop infrastructure
python infrastructure/manage_infrastructure.py --stop

# Restart infrastructure
python infrastructure/manage_infrastructure.py --restart

# Check container status
docker compose ps
```

### Data Generation & Ingestion

```bash
# Full pipeline: Generate 100 patients and ingest
python healthcare_data_pipeline/pipeline.py 100 Massachusetts

# Generate 500 patients in California
python healthcare_data_pipeline/pipeline.py 500 California

# Generate data without starting infrastructure
python healthcare_data_pipeline/pipeline.py 100 Massachusetts --skip-infra

# Ingest existing FHIR data (manual ingestion)
python healthcare_data_pipeline/ingest.py ./synthea_output/output/fhir

# Ingest with custom MongoDB connection
python healthcare_data_pipeline/ingest.py ./synthea_output/output/fhir \
  --host mongodb.example.com \
  --port 27017 \
  --user admin \
  --password mongopass123
```

### Cleanup & Maintenance

```bash
# View logs
docker compose logs mongodb
docker compose logs -f redis

# Clean up all data
docker exec -it text_to_mongo_db mongosh \
  -u admin -p mongopass123 \
  --authenticationDatabase admin \
  text_to_mongo_db \
  --eval "db.dropDatabase()"

# Remove all containers and volumes
docker compose down -v
```

## 📁 Project Structure

```
.
├── infrastructure/
│   ├── docker-compose.yml              # Main infrastructure definition
│   └── manage_infrastructure.py        # Infrastructure management
├── healthcare_data_pipeline/
│   ├── pipeline.py                     # Full pipeline orchestrator (RECOMMENDED)
│   ├── ingest.py                       # FHIR to MongoDB ingestion script
│   ├── README.md                       # This documentation
│   └── synthea_output/                 # Generated FHIR data (created)
│       └── fhir/                       # FHIR R4 bundle files
└── data/
    └── synthea/                        # Additional data files
```

### Key Python Scripts

- **`healthcare_data_pipeline/pipeline.py`** - Orchestrates the entire pipeline (recommended entry point)
  - Manages infrastructure startup
  - Triggers Synthea Docker for data generation
  - Calls ingestion script
  - Verifies data was loaded
  - Provides connection details

- **`healthcare_data_pipeline/ingest.py`** - Handles FHIR data ingestion into MongoDB
  - **Automatic collection separation**: Groups resources by type and routes to appropriate collections
  - Supports all major FHIR resource types (12+ collections)
  - Creates optimized indexes for each collection
  - Provides bulk upsert operations for performance
  - Maintains FHIR resource relationships and references
  - Can be run standalone for existing FHIR data

## 🔌 Connection Details

### MongoDB
- **Host**: localhost
- **Port**: 27017
- **Username**: admin
- **Password**: mongopass123
- **Database**: text_to_mongo_db
- **Connection String**: 
  ```
  mongodb://admin:mongopass123@localhost:27017/text_to_mongo_db?authSource=admin
  ```

### Redis
- **Host**: localhost
- **Port**: 6379
- **Password**: redispass123

## 🎓 Learning Objectives

This project demonstrates:

1. **Healthcare Data Standards**: Working with FHIR R4 format
2. **Docker Orchestration**: Multi-container applications with Docker Compose
3. **ETL Pipelines**: Extract, Transform, Load healthcare data
4. **NoSQL Database Design**: MongoDB schema design for healthcare
5. **Data Relationships**: Managing correlated healthcare data
6. **Query Optimization**: Indexing strategies for performance
7. **MCP Integration**: Natural language to database queries (next phase)
8. **Agentic AI**: Building AI agents for data exploration (next phase)

## 🚦 Next Steps (MCP Integration)

To complete your portfolio project:

1. **MCP Server Setup**: Create an MCP server that connects to MongoDB
2. **Query Translation**: Build NL-to-MongoDB query translator using Claude
3. **Tool Definitions**: Define MCP tools for common healthcare queries
4. **Agent Workflows**: Create multi-step agentic workflows for complex analysis
5. **Results Visualization**: Build a UI to display query results

Example MCP tool structure:
```json
{
  "name": "find_patients_by_condition",
  "description": "Find patients with a specific medical condition",
  "parameters": {
    "condition": "string - the medical condition to search for"
  }
}
```

## 🐛 Troubleshooting

### Issue: Python script says Docker is not installed
**Solution**: Ensure Docker Desktop is installed and in your PATH. Restart your terminal after installation.

### Issue: MongoDB connection timeout
**Solution**: 
1. Ensure Docker is running: `docker ps`
2. Verify MongoDB container is healthy: `docker ps | grep mongo`
3. Try manually starting infrastructure: `python infrastructure/manage_infrastructure.py --start`

### Issue: Synthea data generation fails
**Solution**:
1. Check Docker logs: `docker compose logs synthea`
2. Ensure adequate disk space (4GB+ recommended)
3. Verify output directory is writable: `ls -la synthea_output/`
4. Try with fewer patients: `python healthcare_data_pipeline/pipeline.py 50 Massachusetts`

### Issue: No data in MongoDB collections
**Solution**:
1. Verify FHIR files exist: `ls synthea_output/fhir/`
2. Run ingestion manually: `python healthcare_data_pipeline/ingest.py ./synthea_output/output/fhir`
3. Check logs for errors
4. Verify MongoDB is running: `docker exec text_to_mongo_db mongosh -u admin -p mongopass123 --authenticationDatabase admin --eval "db.adminCommand('ping')"`

### Issue: Port 27017 (MongoDB) or 6379 (Redis) already in use
**Solution**: 
1. Option A: Stop existing containers: `docker compose down`
2. Option B: Change ports in `infrastructure/docker-compose.yml`
3. Option C: Use `--skip-infra` flag if you have another MongoDB running

### Issue: "Permission denied" on Python scripts
**Solution**: Make scripts executable:
```bash
chmod +x healthcare_data_pipeline/pipeline.py
chmod +x healthcare_data_pipeline/ingest.py
```

### Issue: Data generation is very slow
**Solution**: 
1. Synthea generation time increases with patient count. 100 patients ≈ 5-10 minutes
2. For development, use smaller datasets: `python healthcare_data_pipeline/pipeline.py 10 Massachusetts`
3. Once data exists, use `--skip-synthea` to reuse it

## 📊 Performance Tips

- **Indexes**: Already created automatically during ingestion
- **Batch Size**: For large datasets (1000+ patients), increase Docker memory
- **Query Optimization**: Use `explain()` to analyze query performance
- **Aggregation**: Use aggregation pipelines for complex analytics

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome!

## 📄 License

MIT License - feel free to use this for learning and portfolio purposes.

## 🙏 Acknowledgments

- [Synthea](https://github.com/synthetichealth/synthea) - Synthetic patient data generator
- [FHIR](https://www.hl7.org/fhir/) - Healthcare data standards
- [MongoDB](https://www.mongodb.com/) - NoSQL database
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

---

**Built with ❤️ for learning healthcare data engineering and AI integration**
