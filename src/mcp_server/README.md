# Healthcare MCP Server

A Model Context Protocol (MCP) server for querying healthcare data stored in MongoDB. This server provides AI assistants with tools to search patients, analyze conditions, review medications, and explore drug information.

## Features

- **Patient Search**: Flexible patient lookup by demographics and identifiers
- **Clinical Timelines**: Complete chronological medical history for patients
- **Condition Analysis**: Population-level disease analysis and trends
- **Financial Analysis**: Insurance claims and billing data insights
- **Medication History**: Patient medication records with drug classification
- **Drug Database**: RxNorm drug information with ATC classifications

## Architecture

The server is built with a modular architecture:

```
src/mcp_server/
├── config.py              # Centralized configuration
├── database/
│   └── connection.py      # MongoDB connection management
├── tools/
│   ├── models.py          # Pydantic data models
│   ├── utils.py           # Shared utilities
│   ├── patient_tools.py   # Patient-focused tools
│   ├── analytics_tools.py # Population analysis tools
│   ├── medication_tools.py # Medication and drug tools
│   └── drug_tools.py      # Drug database tools
└── server.py              # Main FastMCP server
```

## Available Tools

### Patient Tools
- `search_patients`: Search patients by name, demographics, location
- `get_patient_clinical_timeline`: Complete medical history for a patient

### Analytics Tools
- `analyze_conditions`: Analyze conditions across populations
- `get_financial_summary`: Financial analysis of claims and billing

### Medication Tools
- `get_medication_history`: Patient medication records with drug details

### Drug Tools
- `search_drugs`: Search RxNorm drug database
- `analyze_drug_classes`: Analyze drug classifications and distributions

## Configuration

Configure the server using environment variables:

```bash
# MongoDB settings
export MONGODB_URI="mongodb://admin:mongopass123@localhost:27017/"
export MONGODB_DATABASE="fhir_db"

# Logging
export LOG_LEVEL="INFO"
```

## Usage

1. **Start the server**:
   ```bash
   python src/mcp_server/server.py
   ```

2. **Connect MCP clients** (Claude Desktop, etc.) to query healthcare data

3. **Example queries**:
   - "Find patients named John in Texas"
   - "Show me patient XYZ's clinical timeline"
   - "Analyze diabetes cases by demographics"
   - "What medications is patient ABC taking?"
   - "Find all drugs in the antithrombotic class"

## Data Collections

The server queries these MongoDB collections:

- `patients`: Patient demographics and identifiers
- `encounters`: Healthcare visits and interactions
- `conditions`: Medical diagnoses and health conditions
- `observations`: Lab results, vital signs, measurements
- `medications`: Prescriptions and medication orders
- `claims`: Insurance claims data
- `drugs`: RxNorm drug ingredients with ATC classifications

## Development

The server uses:
- **FastMCP**: Modern MCP server framework
- **Pydantic**: Data validation and serialization
- **MongoDB**: Document database for healthcare data
- **Python 3.11+**: Modern Python async features

## Error Handling

All tools include comprehensive error handling with:
- Connection retry logic
- Input validation
- Detailed error messages
- Graceful degradation

## Security Considerations

- Patient data is sensitive - handle appropriately
- Implement proper authentication in production
- Use TLS for MongoDB connections
- Audit tool access patterns
- Limit query result sizes



