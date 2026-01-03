"""Automated Test Matrix Generator for MCP Tools.

This module discovers all MCP tools and MongoDB collections, then runs
comprehensive integration tests against real data. No mocking, no unit tests -
only real tool calls against the actual database.

Usage:
    python -m pytest tests/integration/test_matrix_generator.py -v
    OR
    python tests/integration/test_matrix_generator.py
"""

import asyncio
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.mcp_server.database import database
from src.mcp_server.security import initialize_security
from src.mcp_server.security.authentication import SecurityContext, UserRole
from src.mcp_server.tools._healthcare.analytics_tools import AnalyticsTools
from src.mcp_server.tools._healthcare.medications import DrugAnalysisTools, MedicationTools
from src.mcp_server.tools._healthcare.patient_tools import PatientTools
from src.mcp_server.tools.models import (
    ClinicalTimelineRequest,
    ConditionAnalysisRequest,
    DrugClassAnalysisRequest,
    FinancialSummaryRequest,
    MedicationHistoryRequest,
    SearchDrugsRequest,
    SearchPatientsRequest,
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ToolTestResult:
    """Result of a single test execution."""

    tool_name: str
    collection: str | None
    test_case: str
    success: bool
    duration_ms: float
    error: str | None = None
    result_data: dict[str, Any] | None = None
    result_count: int | None = None
    fields_extracted: list[str] = field(default_factory=list)


@dataclass
class ToolTestMatrix:
    """Complete test matrix with all results."""

    patient_id: str
    patient_name: str
    collections: list[str]
    tools: list[str]
    results: list[ToolTestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total_tests - passed
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0

        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
            "duration_seconds": duration,
            "collections_tested": len({r.collection for r in self.results if r.collection}),
            "tools_tested": len({r.tool_name for r in self.results}),
        }


class MCPToolTestMatrixGenerator:
    """Automated test matrix generator for MCP tools."""

    def __init__(
        self, test_patient_first_name: str = "Alton320", test_patient_last_name: str = "Roob72"
    ):
        """Initialize test matrix generator.

        Args:
            test_patient_first_name: First name of test patient
            test_patient_last_name: Last name of test patient
        """
        self.test_patient_first_name = test_patient_first_name
        self.test_patient_last_name = test_patient_last_name
        self.patient_id: str | None = None
        self.matrix = ToolTestMatrix(
            patient_id="",
            patient_name=f"{test_patient_first_name} {test_patient_last_name}",
            collections=[],
            tools=[],
        )

        # Initialize database
        database.initialize(
            connection_uri=settings.mongodb_connection_string,
            database_name=settings.mongodb_database,
            min_pool_size=5,
            max_pool_size=20,
        )

        # Initialize security (required for tools that use security manager)
        if settings.security_enabled:
            initialize_security()

        # Create security context (CLINICIAN role for full access)
        self.security_context = SecurityContext(
            user_id="test_user",
            role=UserRole.CLINICIAN,
            session_id="test_session",
            ip_address="127.0.0.1",
        )

    async def discover_collections(self) -> list[str]:
        """Discover all MongoDB collections."""
        try:
            db = database.get_database()
            collections = await db.list_collection_names()
            # Filter out system collections
            collections = [c for c in collections if not c.startswith("system.")]
            self.matrix.collections = sorted(collections)
            logger.info(f"Discovered {len(collections)} collections: {collections}")
            return collections
        except Exception as e:
            logger.error(f"Failed to discover collections: {e}", exc_info=True)
            return []

    async def discover_patient_id(self) -> str | None:
        """Find the test patient ID."""
        try:
            patient_tools = PatientTools()
            request = SearchPatientsRequest(
                first_name=self.test_patient_first_name,
                last_name=self.test_patient_last_name,
                limit=1,
            )
            results = await patient_tools.search_patients(request, self.security_context)
            if results and len(results) > 0:
                self.patient_id = results[0].patient_id
                self.matrix.patient_id = self.patient_id
                logger.info(f"Found patient ID: {self.patient_id}")
                return self.patient_id
            else:
                logger.warning(
                    f"Patient {self.test_patient_first_name} {self.test_patient_last_name} not found"
                )
                return None
        except Exception as e:
            logger.error(f"Failed to find patient: {e}", exc_info=True)
            return None

    def discover_tools(self) -> list[tuple[str, Callable]]:
        """Discover all MCP tools from the server."""
        tools = []

        # Patient Tools
        patient_tools = PatientTools()
        tools.append(("search_patients", patient_tools.search_patients))
        tools.append(("get_patient_clinical_timeline", patient_tools.get_patient_clinical_timeline))

        # Analytics Tools
        analytics_tools = AnalyticsTools()
        tools.append(("analyze_conditions", analytics_tools.analyze_conditions))
        tools.append(("get_financial_summary", analytics_tools.get_financial_summary))

        # Medication Tools
        medication_tools = MedicationTools()
        tools.append(("get_medication_history", medication_tools.get_medication_history))

        # Drug Tools
        drug_tools = DrugAnalysisTools()
        tools.append(("search_drugs", drug_tools.search_drugs))
        tools.append(("analyze_drug_classes", drug_tools.analyze_drug_classes))

        self.matrix.tools = [name for name, _ in tools]
        logger.info(f"Discovered {len(tools)} tools: {self.matrix.tools}")
        return tools

    def get_tool_collection_mapping(self) -> dict[str, list[str]]:
        """Map tools to their relevant collections."""
        return {
            "search_patients": ["patients"],
            "get_patient_clinical_timeline": [
                "patients",
                "encounters",
                "conditions",
                "observations",
                "medications",
                "procedures",
                "immunizations",
            ],
            "analyze_conditions": ["conditions"],
            "get_financial_summary": ["claims", "explanationofbenefits"],
            "get_medication_history": ["medications", "drugs"],
            "search_drugs": ["drugs"],
            "analyze_drug_classes": ["drugs"],
        }

    async def run_test(
        self, tool_name: str, tool_func: Callable, test_case: str, **kwargs
    ) -> ToolTestResult:
        """Run a single test case."""
        start_time = datetime.now()
        collection = kwargs.pop("collection", None)

        try:
            # Execute tool
            result = await tool_func(**kwargs)

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Extract result metadata
            result_data = None
            result_count = None
            fields_extracted = []

            if isinstance(result, dict):
                result_data = result
                # Try to extract count
                if "count" in result:
                    result_count = result["count"]
                elif "total_count" in result:
                    result_count = result["total_count"]
                elif "total_medications" in result:
                    result_count = result["total_medications"]
                elif "total_drugs" in result:
                    result_count = result["total_drugs"]
                elif "total_events" in result:
                    result_count = result["total_events"]
                elif "total_classes" in result:
                    result_count = result["total_classes"]
                elif "patients" in result and isinstance(result["patients"], list):
                    result_count = len(result["patients"])
                elif "medications" in result and isinstance(result["medications"], list):
                    result_count = len(result["medications"])
                elif "drugs" in result and isinstance(result["drugs"], list):
                    result_count = len(result["drugs"])
                elif "events" in result and isinstance(result["events"], list):
                    result_count = len(result["events"])

                # Extract field names from first item if list exists
                for key in ["patients", "medications", "drugs", "events", "groups", "classes"]:
                    if key in result and isinstance(result[key], list) and len(result[key]) > 0:
                        first_item = result[key][0]
                        if isinstance(first_item, dict):
                            fields_extracted = list(first_item.keys())
                            break

            elif hasattr(result, "model_dump"):
                result_data = result.model_dump()
                # Try to get count from model
                if hasattr(result, "count"):
                    result_count = result.count
                elif hasattr(result, "total_count"):
                    result_count = result.total_count
                elif hasattr(result, "total_medications"):
                    result_count = result.total_medications
                elif hasattr(result, "total_events"):
                    result_count = result.total_events

            return ToolTestResult(
                tool_name=tool_name,
                collection=collection,
                test_case=test_case,
                success=True,
                duration_ms=duration_ms,
                result_data=result_data,
                result_count=result_count,
                fields_extracted=fields_extracted,
            )

        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Test failed: {tool_name} - {error_msg}", exc_info=True)
            return ToolTestResult(
                tool_name=tool_name,
                collection=collection,
                test_case=test_case,
                success=False,
                duration_ms=duration_ms,
                error=error_msg,
            )

    async def test_search_patients(self, tool_func: Callable) -> list[ToolTestResult]:
        """Test search_patients tool."""
        results = []

        # Test 1: Search by name
        result = await self.run_test(
            "search_patients",
            tool_func,
            "Search by first and last name",
            request=SearchPatientsRequest(
                first_name=self.test_patient_first_name,
                last_name=self.test_patient_last_name,
                limit=10,
            ),
            security_context=self.security_context,
            collection="patients",
        )
        results.append(result)

        # Test 2: Search by first name only
        result = await self.run_test(
            "search_patients",
            tool_func,
            "Search by first name only",
            request=SearchPatientsRequest(first_name=self.test_patient_first_name, limit=10),
            security_context=self.security_context,
            collection="patients",
        )
        results.append(result)

        # Test 3: Search with limit
        result = await self.run_test(
            "search_patients",
            tool_func,
            "Search with limit=5",
            request=SearchPatientsRequest(limit=5),
            security_context=self.security_context,
            collection="patients",
        )
        results.append(result)

        return results

    async def test_get_patient_clinical_timeline(self, tool_func: Callable) -> list[ToolTestResult]:
        """Test get_patient_clinical_timeline tool."""
        if not self.patient_id:
            logger.warning("No patient ID available, skipping timeline tests")
            return []

        results = []

        # Test 1: Full timeline
        result = await self.run_test(
            "get_patient_clinical_timeline",
            tool_func,
            "Get full clinical timeline",
            request=ClinicalTimelineRequest(patient_id=self.patient_id, limit=100),
            security_context=self.security_context,
            collection="patients",
        )
        results.append(result)

        # Test 2: Timeline with event types
        result = await self.run_test(
            "get_patient_clinical_timeline",
            tool_func,
            "Get timeline with event types (encounters, conditions)",
            request=ClinicalTimelineRequest(
                patient_id=self.patient_id,
                event_types=["encounter", "condition"],
                limit=50,
            ),
            security_context=self.security_context,
            collection="patients",
        )
        results.append(result)

        return results

    async def test_analyze_conditions(self, tool_func: Callable) -> list[ToolTestResult]:
        """Test analyze_conditions tool."""
        results = []

        # Test 1: Group by condition
        result = await self.run_test(
            "analyze_conditions",
            tool_func,
            "Analyze conditions grouped by condition name",
            request=ConditionAnalysisRequest(group_by="condition", limit=20),
            security_context=self.security_context,
            collection="conditions",
        )
        results.append(result)

        # Test 2: Patient-specific conditions
        if self.patient_id:
            result = await self.run_test(
                "analyze_conditions",
                tool_func,
                "Analyze conditions for test patient",
                request=ConditionAnalysisRequest(patient_id=self.patient_id, limit=20),
                security_context=self.security_context,
                collection="conditions",
            )
            results.append(result)

        # Test 3: Active conditions only
        result = await self.run_test(
            "analyze_conditions",
            tool_func,
            "Analyze active conditions",
            request=ConditionAnalysisRequest(status="active", limit=20),
            security_context=self.security_context,
            collection="conditions",
        )
        results.append(result)

        return results

    async def test_get_financial_summary(self, tool_func: Callable) -> list[ToolTestResult]:
        """Test get_financial_summary tool."""
        results = []

        # Test 1: Overall summary
        result = await self.run_test(
            "get_financial_summary",
            tool_func,
            "Get overall financial summary",
            request=FinancialSummaryRequest(),
            security_context=self.security_context,
            collection="claims",
        )
        results.append(result)

        # Test 2: Patient-specific summary
        if self.patient_id:
            result = await self.run_test(
                "get_financial_summary",
                tool_func,
                "Get financial summary for test patient",
                request=FinancialSummaryRequest(patient_id=self.patient_id),
                security_context=self.security_context,
                collection="claims",
            )
            results.append(result)

        return results

    async def test_get_medication_history(self, tool_func: Callable) -> list[ToolTestResult]:
        """Test get_medication_history tool."""
        results = []

        # Test 1: All medications
        result = await self.run_test(
            "get_medication_history",
            tool_func,
            "Get all medications",
            request=MedicationHistoryRequest(include_drug_details=True, limit=50),
            security_context=self.security_context,
            collection="medications",
        )
        results.append(result)

        # Test 2: Patient-specific medications
        if self.patient_id:
            result = await self.run_test(
                "get_medication_history",
                tool_func,
                "Get medication history for test patient",
                request=MedicationHistoryRequest(
                    patient_id=self.patient_id, include_drug_details=True, limit=50
                ),
                security_context=self.security_context,
                collection="medications",
            )
            results.append(result)

        # Test 3: Active medications only
        result = await self.run_test(
            "get_medication_history",
            tool_func,
            "Get active medications only",
            request=MedicationHistoryRequest(status="active", include_drug_details=True, limit=50),
            security_context=self.security_context,
            collection="medications",
        )
        results.append(result)

        return results

    async def test_search_drugs(self, tool_func: Callable) -> list[ToolTestResult]:
        """Test search_drugs tool."""
        results = []

        # Test 1: Search by name
        result = await self.run_test(
            "search_drugs",
            tool_func,
            "Search drugs by name (aspirin)",
            request=SearchDrugsRequest(drug_name="aspirin", limit=10),
            collection="drugs",
        )
        results.append(result)

        # Test 2: Search by therapeutic class
        result = await self.run_test(
            "search_drugs",
            tool_func,
            "Search drugs by therapeutic class",
            request=SearchDrugsRequest(therapeutic_class="ANTIDIABETIC AGENTS", limit=10),
            collection="drugs",
        )
        results.append(result)

        return results

    async def test_analyze_drug_classes(self, tool_func: Callable) -> list[ToolTestResult]:
        """Test analyze_drug_classes tool."""
        results = []

        # Test 1: Group by therapeutic class
        result = await self.run_test(
            "analyze_drug_classes",
            tool_func,
            "Analyze drug classes by therapeutic class",
            request=DrugClassAnalysisRequest(group_by="therapeutic_class", limit=20),
            collection="drugs",
        )
        results.append(result)

        # Test 2: Group by drug class
        result = await self.run_test(
            "analyze_drug_classes",
            tool_func,
            "Analyze drug classes by drug class",
            request=DrugClassAnalysisRequest(group_by="drug_class", limit=20),
            collection="drugs",
        )
        results.append(result)

        return results

    async def run_all_tests(self) -> ToolTestMatrix:
        """Run complete test matrix."""
        logger.info("=" * 80)
        logger.info("MCP TOOL TEST MATRIX GENERATOR")
        logger.info("=" * 80)

        # Step 1: Discover collections
        logger.info("\n[STEP 1] Discovering MongoDB collections...")
        await self.discover_collections()

        # Step 2: Discover patient ID
        logger.info(
            f"\n[STEP 2] Finding test patient: {self.test_patient_first_name} {self.test_patient_last_name}..."
        )
        await self.discover_patient_id()

        if not self.patient_id:
            logger.warning("WARNING: Test patient not found. Some tests will be skipped.")

        # Step 3: Discover tools
        logger.info("\n[STEP 3] Discovering MCP tools...")
        tools = self.discover_tools()

        # Step 4: Run tests for each tool
        logger.info("\n[STEP 4] Running test matrix...")
        logger.info("-" * 80)

        test_methods = {
            "search_patients": self.test_search_patients,
            "get_patient_clinical_timeline": self.test_get_patient_clinical_timeline,
            "analyze_conditions": self.test_analyze_conditions,
            "get_financial_summary": self.test_get_financial_summary,
            "get_medication_history": self.test_get_medication_history,
            "search_drugs": self.test_search_drugs,
            "analyze_drug_classes": self.test_analyze_drug_classes,
        }

        for tool_name, tool_func in tools:
            if tool_name in test_methods:
                logger.info(f"\nTesting tool: {tool_name}")
                try:
                    test_results = await test_methods[tool_name](tool_func)
                    self.matrix.results.extend(test_results)
                    passed = sum(1 for r in test_results if r.success)
                    logger.info(f"  ✓ {passed}/{len(test_results)} tests passed")
                except Exception as e:
                    logger.error(f"  ✗ Failed to run tests for {tool_name}: {e}", exc_info=True)

        # Finalize
        self.matrix.end_time = datetime.now()
        return self.matrix

    def generate_report(self, output_file: str | None = None) -> str:
        """Generate comprehensive test report."""
        summary = self.matrix.get_summary()

        report_lines = [
            "=" * 80,
            "MCP TOOL TEST MATRIX REPORT",
            "=" * 80,
            "",
            f"Test Patient: {self.matrix.patient_name} (ID: {self.matrix.patient_id})",
            f"Test Duration: {summary['duration_seconds']:.2f} seconds",
            f"Start Time: {self.matrix.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time: {self.matrix.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.matrix.end_time else 'N/A'}",
            "",
            "SUMMARY",
            "-" * 80,
            f"Total Tests: {summary['total_tests']}",
            f"Passed: {summary['passed']}",
            f"Failed: {summary['failed']}",
            f"Success Rate: {summary['success_rate']:.1f}%",
            f"Collections Tested: {summary['collections_tested']}",
            f"Tools Tested: {summary['tools_tested']}",
            "",
            "COLLECTIONS DISCOVERED",
            "-" * 80,
            ", ".join(self.matrix.collections),
            "",
            "TOOLS DISCOVERED",
            "-" * 80,
            ", ".join(self.matrix.tools),
            "",
            "DETAILED RESULTS",
            "-" * 80,
        ]

        # Group results by tool
        by_tool: dict[str, list[ToolTestResult]] = {}
        for result in self.matrix.results:
            if result.tool_name not in by_tool:
                by_tool[result.tool_name] = []
            by_tool[result.tool_name].append(result)

        for tool_name, results in sorted(by_tool.items()):
            report_lines.append(f"\n{tool_name.upper()}")
            report_lines.append("-" * 80)
            for result in results:
                status = "[PASS]" if result.success else "[FAIL]"
                duration = f"{result.duration_ms:.1f}ms"
                collection_info = f" [{result.collection}]" if result.collection else ""
                report_lines.append(
                    f"  {status} | {duration} | {result.test_case}{collection_info}"
                )
                if result.result_count is not None:
                    report_lines.append(f"    → Results: {result.result_count}")
                if result.fields_extracted:
                    report_lines.append(f"    → Fields: {', '.join(result.fields_extracted[:5])}")
                if result.error:
                    report_lines.append(f"    → Error: {result.error}")

        # Failed tests summary
        failed_tests = [r for r in self.matrix.results if not r.success]
        if failed_tests:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("FAILED TESTS SUMMARY")
            report_lines.append("=" * 80)
            for result in failed_tests:
                report_lines.append(f"\n{result.tool_name} - {result.test_case}")
                report_lines.append(f"  Collection: {result.collection or 'N/A'}")
                report_lines.append(f"  Error: {result.error}")

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"\nReport saved to: {output_file}")

        return report


async def main():
    """Main entry point."""
    generator = MCPToolTestMatrixGenerator()
    matrix = await generator.run_all_tests()
    report = generator.generate_report("test_matrix_report.txt")
    print("\n" + report)

    # Exit with error code if any tests failed
    summary = matrix.get_summary()
    if summary["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
