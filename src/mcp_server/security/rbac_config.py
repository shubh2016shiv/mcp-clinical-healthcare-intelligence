"""Role-Based Access Control Configuration

This module centralizes all RBAC (Role-Based Access Control) rules for the MCP server.
It maps user roles to accessible MongoDB collections and tools to their required collections.

This design provides:
- Single source of truth for access control
- Easy maintenance and updates
- Clear separation of concerns
- Audit trail capabilities

Collections mapped:
- patients: Core patient demographics and identification
- conditions: Medical diagnoses and health conditions
- medications: Prescription and medication records
- observations: Clinical measurements and test results
- encounters: Healthcare visit and interaction records
- procedures: Medical procedures and interventions
- immunizations: Vaccination and immunization records
- care_plans: Structured care management plans
- claims: Insurance claims (billing records)
- explanationofbenefits: Insurance payment explanations
- diagnosticreports: Clinical summary reports
- drugs: Comprehensive drug reference database
"""

from .authentication import UserRole

# Role-to-collection access mapping
# Defines which collections each role can access
ROLE_COLLECTION_ACCESS = {
    UserRole.ADMIN: ["*"],  # Full access to all collections
    UserRole.CLINICIAN: [
        "patients",
        "conditions",
        "medications",
        "observations",
        "encounters",
        "procedures",
        "immunizations",
        "care_plans",
        "diagnosticreports",
    ],
    UserRole.BILLING: ["patients", "claims", "explanationofbenefits"],
    UserRole.RESEARCHER: ["conditions", "medications", "observations", "procedures", "drugs"],
    UserRole.READ_ONLY: ["patients"],
}


# Tool-to-collection mapping
# Defines which collections each tool requires access to
TOOL_COLLECTION_MAP = {
    "search_patients": ["patients"],
    "get_patient_clinical_timeline": [
        "patients",
        "encounters",
        "conditions",
        "medications",
        "procedures",
        "immunizations",
        "observations",
    ],
    "analyze_conditions": ["conditions"],
    "get_financial_summary": ["claims", "explanationofbenefits"],
    "get_medication_history": ["medications", "drugs"],
    "search_drugs": ["drugs"],
    "analyze_drug_classes": ["drugs"],
}


def get_required_collections_for_tool(tool_name: str) -> list:
    """Get the collections required for a specific tool

    Args:
        tool_name: Name of the tool function

    Returns:
        List of collection names required by the tool

    Raises:
        ValueError: If tool is not found in the mapping
    """
    if tool_name not in TOOL_COLLECTION_MAP:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOL_COLLECTION_MAP[tool_name]


def check_role_access_to_collections(role: UserRole, collections: list) -> bool:
    """Check if a role has access to all specified collections

    Args:
        role: User role to check
        collections: List of collection names

    Returns:
        True if role has access to all collections, False otherwise
    """
    if role not in ROLE_COLLECTION_ACCESS:
        return False

    allowed_collections = ROLE_COLLECTION_ACCESS[role]

    # Admin has access to everything
    if "*" in allowed_collections:
        return True

    # Check if all required collections are allowed
    return all(collection in allowed_collections for collection in collections)


def get_allowed_tools_for_role(role: UserRole) -> list:
    """Get all tools that a role is allowed to use

    Args:
        role: User role

    Returns:
        List of tool names the role can access
    """
    allowed_tools = []
    for tool_name, required_collections in TOOL_COLLECTION_MAP.items():
        if check_role_access_to_collections(role, required_collections):
            allowed_tools.append(tool_name)
    return allowed_tools
