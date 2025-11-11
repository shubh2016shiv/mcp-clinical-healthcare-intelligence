"""Test script for MCP healthcare tools.

This script tests each MCP tool to verify they work correctly.
Run this while the MCP server is running in another terminal.
"""

import asyncio
import json
import logging
from src.mcp_server.database.connection import get_connection_manager
from src.mcp_server.tools.patient_tools import PatientTools
from src.mcp_server.tools.analytics_tools import AnalyticsTools
from src.mcp_server.tools._healthcare.medications import MedicationTools
from src.mcp_server.tools._healthcare.medications import DrugAnalysisTools as DrugTools
from src.mcp_server.tools.models import (
    SearchPatientsRequest,
    ClinicalTimelineRequest,
    ConditionAnalysisRequest,
    MedicationHistoryRequest,
    SearchDrugsRequest,
    DrugClassAnalysisRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_patient_tools():
    """Test patient-focused tools."""
    print("\n" + "="*80)
    print("TESTING PATIENT TOOLS")
    print("="*80)
    
    tools = PatientTools()
    
    # Test 1: Search patients
    print("\n[TEST 1] search_patients - Finding first 5 patients")
    print("-" * 80)
    try:
        request = SearchPatientsRequest(limit=5)
        result = await tools.search_patients(request)
        print(f"✓ Found {len(result)} patients")
        for patient in result[:2]:
            print(f"  - {patient.first_name} {patient.last_name} (ID: {patient.patient_id})")
        if result:
            first_patient_id = result[0].patient_id
        else:
            print("  No patients found in database!")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    
    # Test 2: Get patient clinical timeline
    print(f"\n[TEST 2] get_patient_clinical_timeline - Timeline for patient {first_patient_id}")
    print("-" * 80)
    try:
        request = ClinicalTimelineRequest(patient_id=first_patient_id, limit=10)
        result = await tools.get_patient_clinical_timeline(request)
        print(f"✓ Retrieved {result.total_events} clinical events")
        for event in result.events[:3]:
            print(f"  - {event.event_type}: {event.event_name} ({event.event_date})")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    return first_patient_id


async def test_analytics_tools():
    """Test analytics tools."""
    print("\n" + "="*80)
    print("TESTING ANALYTICS TOOLS")
    print("="*80)
    
    tools = AnalyticsTools()
    
    # Test 1: Analyze conditions
    print("\n[TEST 1] analyze_conditions - Finding top 5 conditions")
    print("-" * 80)
    try:
        request = ConditionAnalysisRequest(
            group_by="condition",
            limit=5
        )
        result = await tools.analyze_conditions(request)
        print(f"✓ Found {result.total_count} condition types")
        if result.groups:
            for group in result.groups[:3]:
                print(f"  - {group.condition_name}: {group.condition_count} cases ({group.patient_count} patients)")
    except Exception as e:
        print(f"✗ Error: {e}")


async def test_medication_tools():
    """Test medication tools."""
    print("\n" + "="*80)
    print("TESTING MEDICATION TOOLS")
    print("="*80)
    
    tools = MedicationTools()
    
    # Test 1: Get medications with drug details
    print("\n[TEST 1] get_medication_history - Finding medications with drug classification")
    print("-" * 80)
    try:
        request = MedicationHistoryRequest(
            include_drug_details=True,
            limit=5
        )
        result = await tools.get_medication_history(request)
        print(f"✓ Found {result.total_medications} medication records")
        print(f"  Enriched with drug data: {result.enriched_with_drug_data}")
        for med in result.medications[:2]:
            print(f"  - {med.medication_name} (Status: {med.status})")
            if med.drug_classification:
                print(f"    Class: {med.drug_classification.get('drug_class_l3')}")
    except Exception as e:
        print(f"✗ Error: {e}")


async def test_drug_tools():
    """Test drug tools."""
    print("\n" + "="*80)
    print("TESTING DRUG TOOLS")
    print("="*80)
    
    tools = DrugTools()
    
    # Test 1: Search drugs
    print("\n[TEST 1] search_drugs - Searching for 'aspirin'")
    print("-" * 80)
    try:
        request = SearchDrugsRequest(drug_name="aspirin", limit=3)
        result = await tools.search_drugs(request)
        print(f"✓ Found {result.total_drugs} drugs matching 'aspirin'")
        for drug in result.drugs[:2]:
            print(f"  - {drug.primary_drug_name} (RxCUI: {drug.ingredient_rxcui})")
            print(f"    Class: {drug.drug_class_l3}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Analyze drug classes
    print("\n[TEST 2] analyze_drug_classes - Top therapeutic classes")
    print("-" * 80)
    try:
        request = DrugClassAnalysisRequest(
            group_by="therapeutic_class",
            limit=5
        )
        result = await tools.analyze_drug_classes(request)
        print(f"✓ Found {result.total_classes} therapeutic classes")
        for drug_class in result.classes[:3]:
            print(f"  - {drug_class.class_name}: {drug_class.drug_count} drugs")
            print(f"    Examples: {', '.join(drug_class.example_drugs[:2])}")
    except Exception as e:
        print(f"✗ Error: {e}")


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Healthcare MCP Tools - Integration Tests" + " "*19 + "║")
    print("╚" + "="*78 + "╝")
    
    # Ensure connection is established
    manager = get_connection_manager()
    if not manager.is_connected():
        print("Connecting to MongoDB...")
        manager.connect()
    
    try:
        # Run tests
        patient_id = await test_patient_tools()
        await test_analytics_tools()
        await test_medication_tools()
        await test_drug_tools()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())


