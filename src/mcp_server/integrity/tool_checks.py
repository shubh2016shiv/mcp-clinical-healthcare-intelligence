"""Tool-specific integrity checks for MCP server.

This module provides specialized checks for MCP tools including
parameter validation, return type checking, error handling, and
tool-specific functionality validation.
"""

import inspect
from typing import Any, get_type_hints

from .checker import CheckCategory, CheckSeverity


class ToolIntegrityChecker:
    """Specialized checker for tool-related integrity."""

    def __init__(self):
        self.tools = {}
        self._load_tools()

    def _load_tools(self):
        """Load available tools safely."""
        try:
            from src.mcp_server.tools._healthcare.analytics_tools import AnalyticsTools
            from src.mcp_server.tools._healthcare.medications import (
                DrugAnalysisTools,
                MedicationTools,
            )
            from src.mcp_server.tools._healthcare.patient_tools import PatientTools

            self.tools = {
                "PatientTools": PatientTools,
                "AnalyticsTools": AnalyticsTools,
                "MedicationTools": MedicationTools,
                "DrugAnalysisTools": DrugAnalysisTools,
            }
        except Exception:
            self.tools = {}

    def check_tool_signatures(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check that all tools have proper signatures."""
        if not self.tools:
            return (
                False,
                "No tools available for signature checking",
                {"tools_loaded": 0},
                ["Load tools before running signature checks"],
            )

        try:
            signature_issues = []
            valid_signatures = 0

            for tool_name, tool_class in self.tools.items():
                try:
                    # Check if it's a class
                    if not inspect.isclass(tool_class):
                        signature_issues.append(f"{tool_name}: Not a class")
                        continue

                    # Check for required methods
                    required_methods = []
                    if tool_name == "PatientTools":
                        required_methods = ["search_patients", "get_patient_clinical_timeline"]
                    elif tool_name == "AnalyticsTools":
                        required_methods = ["analyze_conditions", "get_financial_summary"]
                    elif tool_name == "MedicationTools":
                        required_methods = ["get_medication_history"]
                    elif tool_name == "DrugAnalysisTools":
                        required_methods = ["search_drugs", "analyze_drug_classes"]

                    missing_methods = []
                    for method in required_methods:
                        if not hasattr(tool_class, method):
                            missing_methods.append(method)

                    if missing_methods:
                        signature_issues.append(f"{tool_name}: Missing methods: {missing_methods}")
                    else:
                        valid_signatures += 1

                    # Check method signatures
                    for method_name in required_methods:
                        if hasattr(tool_class, method_name):
                            method = getattr(tool_class, method_name)
                            if callable(method):
                                sig = inspect.signature(method)
                                # Check if method accepts expected parameters
                                params = list(sig.parameters.keys())
                                if method_name == "search_patients" and "request" not in params:
                                    signature_issues.append(
                                        f"{tool_name}.{method_name}: Missing 'request' parameter"
                                    )
                                elif (
                                    method_name == "get_patient_clinical_timeline"
                                    and "request" not in params
                                ):
                                    signature_issues.append(
                                        f"{tool_name}.{method_name}: Missing 'request' parameter"
                                    )

                except Exception as e:
                    signature_issues.append(f"{tool_name}: Signature check failed - {e}")

            passed = len(signature_issues) == 0 and valid_signatures > 0

            details = {
                "tools_checked": len(self.tools),
                "valid_signatures": valid_signatures,
                "signature_issues": len(signature_issues),
                "issues": signature_issues[:5],  # Limit to first 5 issues
            }

            recommendations = []
            if signature_issues:
                recommendations.append(f"Fix {len(signature_issues)} tool signature issues")
                recommendations.extend(signature_issues[:3])  # First 3 issues as recommendations

            return (
                passed,
                f"Tool signatures {'valid' if passed else 'have issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Tool signature check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check tool implementations",
                    "Verify method signatures",
                    "Review tool class definitions",
                ],
            )

    def check_tool_return_types(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check that tools return expected types."""
        if not self.tools:
            return (
                False,
                "No tools available for return type checking",
                {"tools_loaded": 0},
                ["Load tools before running return type checks"],
            )

        try:
            return_type_issues = []
            valid_return_types = 0

            for tool_name, tool_class in self.tools.items():
                try:
                    tool_instance = tool_class()

                    # Test methods that can be called safely
                    test_methods = []
                    if tool_name == "PatientTools":
                        # Can't test without proper request objects, so just check method existence
                        test_methods = ["search_patients", "get_patient_clinical_timeline"]
                    elif tool_name == "AnalyticsTools":
                        test_methods = ["analyze_conditions", "get_financial_summary"]
                    elif tool_name == "MedicationTools":
                        test_methods = ["get_medication_history"]
                    elif tool_name == "DrugAnalysisTools":
                        test_methods = ["search_drugs", "analyze_drug_classes"]

                    for method_name in test_methods:
                        if hasattr(tool_instance, method_name):
                            method = getattr(tool_instance, method_name)

                            # Check if method has type hints
                            try:
                                hints = get_type_hints(method)
                                if "return" in hints:
                                    valid_return_types += 1
                                else:
                                    return_type_issues.append(
                                        f"{tool_name}.{method_name}: No return type hint"
                                    )
                            except Exception:
                                return_type_issues.append(
                                    f"{tool_name}.{method_name}: Type hint parsing failed"
                                )

                except Exception as e:
                    return_type_issues.append(f"{tool_name}: Return type check failed - {e}")

            passed = len(return_type_issues) == 0 and valid_return_types > 0

            details = {
                "tools_checked": len(self.tools),
                "valid_return_types": valid_return_types,
                "return_type_issues": len(return_type_issues),
                "issues": return_type_issues[:5],  # Limit to first 5 issues
            }

            recommendations = []
            if return_type_issues:
                recommendations.append(f"Fix {len(return_type_issues)} return type issues")
                recommendations.append("Add proper type hints to tool methods")
                recommendations.extend(return_type_issues[:2])  # First 2 issues

            return (
                passed,
                f"Tool return types {'valid' if passed else 'have issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Tool return type check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check tool method implementations",
                    "Add proper type hints",
                    "Verify return type annotations",
                ],
            )

    def check_tool_error_handling(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check that tools handle errors properly."""
        if not self.tools:
            return (
                False,
                "No tools available for error handling checking",
                {"tools_loaded": 0},
                ["Load tools before running error handling checks"],
            )

        try:
            error_handling_issues = []
            error_handling_good = 0

            for tool_name, tool_class in self.tools.items():
                try:
                    # Check if tools use proper error handling decorators

                    # Check class methods for error handling
                    methods_with_error_handling = 0
                    methods_checked = 0

                    for attr_name in dir(tool_class):
                        if not attr_name.startswith("_"):
                            attr = getattr(tool_class, attr_name)
                            if callable(attr) and hasattr(attr, "__wrapped__"):
                                # Method has been decorated (assuming decorators wrap functions)
                                methods_with_error_handling += 1
                            methods_checked += 1

                    if methods_checked > 0:
                        error_handling_ratio = methods_with_error_handling / methods_checked
                        if (
                            error_handling_ratio < 0.5
                        ):  # Less than 50% of methods have error handling
                            error_handling_issues.append(
                                f"{tool_name}: Only {methods_with_error_handling}/{methods_checked} methods have error handling"
                            )
                        else:
                            error_handling_good += 1

                except Exception as e:
                    error_handling_issues.append(f"{tool_name}: Error handling check failed - {e}")

            passed = len(error_handling_issues) == 0 and error_handling_good > 0

            details = {
                "tools_checked": len(self.tools),
                "error_handling_good": error_handling_good,
                "error_handling_issues": len(error_handling_issues),
                "issues": error_handling_issues[:5],  # Limit to first 5 issues
            }

            recommendations = []
            if error_handling_issues:
                recommendations.append(f"Fix error handling in {len(error_handling_issues)} tools")
                recommendations.append("Use @handle_mongo_errors decorator on database operations")
                recommendations.extend(error_handling_issues[:2])  # First 2 issues

            return (
                passed,
                f"Tool error handling {'good' if passed else 'needs improvement'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Tool error handling check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check tool implementations for error handling",
                    "Use proper error handling decorators",
                    "Review exception handling patterns",
                ],
            )

    def check_tool_imports(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check that tools can import all required dependencies."""
        if not self.tools:
            return (
                False,
                "No tools available for import checking",
                {"tools_loaded": 0},
                ["Load tools before running import checks"],
            )

        try:
            import_issues = []
            imports_working = 0

            for tool_name, tool_class in self.tools.items():
                try:
                    # Try to instantiate the tool (this will catch import issues)
                    tool_class()  # Instantiate to verify imports work
                    imports_working += 1

                    # Check for common dependencies
                    missing_deps = []
                    if tool_name == "PatientTools":
                        try:
                            # Import to verify dependencies are available
                            from src.mcp_server.tools._healthcare.clinical_timeline import (
                                PatientTimelineTools,  # noqa: F401
                            )
                            from src.mcp_server.tools._healthcare.demographics import (
                                search_patients,  # noqa: F401
                            )
                        except ImportError as e:
                            missing_deps.append(f"Patient tool dependencies: {e}")

                    elif tool_name == "AnalyticsTools":
                        try:
                            # Import to verify dependencies are available
                            from src.mcp_server.tools._healthcare.clinical_data.condition_analytics import (
                                ConditionAnalytics,  # noqa: F401
                            )
                            from src.mcp_server.tools._healthcare.financial import (
                                FinancialAnalytics,  # noqa: F401
                            )
                        except ImportError as e:
                            missing_deps.append(f"Analytics tool dependencies: {e}")

                    if missing_deps:
                        import_issues.extend(missing_deps)

                except Exception as e:
                    import_issues.append(f"{tool_name}: Import/instantiation failed - {e}")

            passed = len(import_issues) == 0 and imports_working > 0

            details = {
                "tools_checked": len(self.tools),
                "imports_working": imports_working,
                "import_issues": len(import_issues),
                "issues": import_issues[:5],  # Limit to first 5 issues
            }

            recommendations = []
            if import_issues:
                recommendations.append(f"Fix {len(import_issues)} import issues")
                recommendations.append("Check module dependencies and paths")
                recommendations.extend(import_issues[:2])  # First 2 issues

            return (
                passed,
                f"Tool imports {'working' if passed else 'have issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Tool import check failed: {type(e).__name__}",
                {"error": str(e)},
                ["Check module paths", "Verify import statements", "Review dependency management"],
            )

    def check_tool_async_support(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check that tools properly support async operations."""
        if not self.tools:
            return (
                False,
                "No tools available for async checking",
                {"tools_loaded": 0},
                ["Load tools before running async checks"],
            )

        try:
            async_issues = []
            async_good = 0

            for tool_name, tool_class in self.tools.items():
                try:
                    # Check if key methods are async
                    async_methods = []
                    sync_methods = []

                    if tool_name == "PatientTools":
                        methods_to_check = ["search_patients", "get_patient_clinical_timeline"]
                    elif tool_name == "AnalyticsTools":
                        methods_to_check = ["analyze_conditions", "get_financial_summary"]
                    elif tool_name == "MedicationTools":
                        methods_to_check = ["get_medication_history"]
                    elif tool_name == "DrugAnalysisTools":
                        methods_to_check = ["search_drugs", "analyze_drug_classes"]
                    else:
                        methods_to_check = []

                    for method_name in methods_to_check:
                        if hasattr(tool_class, method_name):
                            method = getattr(tool_class, method_name)
                            if inspect.iscoroutinefunction(method):
                                async_methods.append(method_name)
                            else:
                                sync_methods.append(method_name)

                    # All main methods should be async
                    if sync_methods:
                        async_issues.append(f"{tool_name}: Non-async methods: {sync_methods}")
                    elif async_methods:
                        async_good += 1

                except Exception as e:
                    async_issues.append(f"{tool_name}: Async check failed - {e}")

            passed = len(async_issues) == 0 and async_good > 0

            details = {
                "tools_checked": len(self.tools),
                "async_good": async_good,
                "async_issues": len(async_issues),
                "issues": async_issues[:5],  # Limit to first 5 issues
            }

            recommendations = []
            if async_issues:
                recommendations.append(f"Fix async support in {len(async_issues)} tools")
                recommendations.append("Use async def for database operations")
                recommendations.extend(async_issues[:2])  # First 2 issues

            return (
                passed,
                f"Tool async support {'good' if passed else 'needs fixing'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Tool async check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check async method implementations",
                    "Use proper async patterns",
                    "Review concurrent operation handling",
                ],
            )

    def run_tool_checks(self) -> list[tuple[str, CheckCategory, CheckSeverity, callable]]:
        """Return all tool checks to be executed."""
        return [
            (
                "Tool Signatures",
                CheckCategory.TOOLS,
                CheckSeverity.HIGH,
                self.check_tool_signatures,
            ),
            (
                "Tool Return Types",
                CheckCategory.TOOLS,
                CheckSeverity.MEDIUM,
                self.check_tool_return_types,
            ),
            (
                "Tool Error Handling",
                CheckCategory.TOOLS,
                CheckSeverity.MEDIUM,
                self.check_tool_error_handling,
            ),
            ("Tool Imports", CheckCategory.TOOLS, CheckSeverity.HIGH, self.check_tool_imports),
            (
                "Tool Async Support",
                CheckCategory.TOOLS,
                CheckSeverity.MEDIUM,
                self.check_tool_async_support,
            ),
        ]
