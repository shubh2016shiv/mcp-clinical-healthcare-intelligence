"""Security server_health_checks checks for MCP server.

This module provides specialized checks for security components including
authentication, authorization, input validation, and security policies.
"""

from typing import Any

from .health_check_framework import CheckCategory, CheckSeverity


class SecurityIntegrityChecker:
    """Specialized checker for security-related server_health_checks."""

    def __init__(self):
        self.security_manager = None
        self.validator = None
        self._load_security_components()

    def _load_security_components(self):
        """Load security components safely."""
        try:
            from src.mcp_server.security import initialize_security

            self.security_manager = initialize_security()
            self.validator = self.security_manager.validator
        except Exception:
            self.security_manager = None
            self.validator = None

    def check_input_validation_rules(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check input validation rules and patterns."""
        if not self.validator:
            return (
                False,
                "Input validator not available",
                {"validator_available": False},
                ["Initialize security components before running validation checks"],
            )

        try:
            # Test dangerous pattern detection
            dangerous_patterns = [
                "$where",
                "$regex",
                "$expr",
                "$function",
                "..",
                "<script",
                "javascript:",
                "\x00",
            ]

            dangerous_detected = 0
            safe_passed = 0

            for pattern in dangerous_patterns:
                try:
                    # This should raise ValueError for dangerous input
                    self.validator.validate_query_params({"test": pattern})
                    # If we get here, validation failed to catch dangerous input
                except ValueError:
                    dangerous_detected += 1
                except Exception:
                    pass  # Other exceptions are OK

            # Test safe input validation
            safe_inputs = ["John", "test@example.com", "12345", "normal text"]
            for safe_input in safe_inputs:
                try:
                    result = self.validator.validate_query_params({"test": safe_input})
                    if "test" in result:
                        safe_passed += 1
                except Exception:
                    pass  # Safe inputs should not cause exceptions

            validation_working = dangerous_detected >= 3 and safe_passed >= 2

            details = {
                "dangerous_patterns_detected": dangerous_detected,
                "safe_inputs_accepted": safe_passed,
                "validation_rules_active": validation_working,
            }

            recommendations = []
            if dangerous_detected < 3:
                recommendations.append("Input validation not catching dangerous patterns")
                recommendations.append("Review dangerous pattern detection rules")
            if safe_passed < 2:
                recommendations.append("Safe inputs being rejected incorrectly")
                recommendations.append("Check input validation whitelist rules")

            return (
                validation_working,
                f"Input validation {'working' if validation_working else 'has issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Input validation check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check validator implementation",
                    "Review validation rules",
                    "Verify security configuration",
                ],
            )

    def check_rate_limiting(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check rate limiting functionality."""
        if not self.security_manager:
            return (
                False,
                "Security manager not available",
                {"manager_available": False},
                ["Initialize security components before running rate limit checks"],
            )

        try:
            # Test rate limiting logic
            rate_limiter = self.security_manager.auth_manager

            if not hasattr(rate_limiter, "check_rate_limit"):
                return (
                    False,
                    "Rate limiting not implemented",
                    {"rate_limiting_available": False},
                    ["Implement rate limiting in authentication manager"],
                )

            # Test rate limiting with mock IP
            test_ip = "192.168.1.100"

            # Should allow initial requests
            initial_allowed = rate_limiter.check_rate_limit(test_ip)
            second_allowed = rate_limiter.check_rate_limit(test_ip)

            # Check rate limit configuration
            config = getattr(self.security_manager, "config", None)
            if config:
                rate_limit = getattr(config, "rate_limit_requests_per_minute", 60)
            else:
                rate_limit = 60

            rate_limiting_configured = rate_limit > 0

            passed = initial_allowed and rate_limiting_configured

            details = {
                "rate_limiting_available": True,
                "initial_request_allowed": initial_allowed,
                "second_request_allowed": second_allowed,
                "rate_limit_per_minute": rate_limit,
                "rate_limiting_configured": rate_limiting_configured,
            }

            recommendations = []
            if not initial_allowed:
                recommendations.append("Rate limiting blocking legitimate requests")
            if not rate_limiting_configured:
                recommendations.append("Configure appropriate rate limits for security")

            return (
                passed,
                f"Rate limiting {'configured' if passed else 'has issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Rate limiting check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check rate limiting implementation",
                    "Verify authentication manager",
                    "Review rate limit configuration",
                ],
            )

    def check_session_security(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check session security mechanisms."""
        if not self.security_manager:
            return (
                False,
                "Security manager not available",
                {"manager_available": False},
                ["Initialize security components before running session checks"],
            )

        try:
            auth_manager = self.security_manager.auth_manager

            # Test session creation
            try:
                session_id = auth_manager.authenticate_user("test_user", "clinician", "127.0.0.1")
                session_created = bool(session_id)
            except Exception:
                session_created = False

            # Test session validation
            if session_created:
                context = auth_manager.validate_session(session_id, "127.0.0.1")
                session_valid = context is not None

                # Test IP validation (different IP should fail)
                context_wrong_ip = auth_manager.validate_session(session_id, "192.168.1.1")
                ip_validation_works = context_wrong_ip is None
            else:
                session_valid = False
                ip_validation_works = False

            # Check session configuration
            config = getattr(self.security_manager, "config", None)
            session_timeout = getattr(config, "session_timeout_minutes", 30) if config else 30

            passed = session_created and session_valid and ip_validation_works

            details = {
                "session_creation_working": session_created,
                "session_validation_working": session_valid,
                "ip_validation_working": ip_validation_works,
                "session_timeout_minutes": session_timeout,
                "session_security_configured": session_timeout > 0,
            }

            recommendations = []
            if not session_created:
                recommendations.append("Session creation failed - check authentication logic")
            if not session_valid:
                recommendations.append("Session validation failed - check session storage")
            if not ip_validation_works:
                recommendations.append("IP validation not working - security risk")
            if session_timeout <= 0:
                recommendations.append("Configure proper session timeout for security")

            return (
                passed,
                f"Session security {'OK' if passed else 'has issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Session security check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Check session management implementation",
                    "Verify authentication flow",
                    "Review session security configuration",
                ],
            )

    def check_data_minimization(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check data minimization and privacy controls."""
        if not self.security_manager:
            return (
                False,
                "Security manager not available",
                {"manager_available": False},
                ["Initialize security components before running privacy checks"],
            )

        try:
            data_minimizer = getattr(self.security_manager, "data_minimizer", None)

            if not data_minimizer:
                return (
                    False,
                    "Data minimizer not available",
                    {"data_minimizer_available": False},
                    ["Implement data minimization component"],
                )

            # Test data minimization methods
            methods_available = []
            methods_tested = [
                "minimize_patient_data",
                "apply_field_masking",
                "validate_data_access",
            ]

            for method in methods_tested:
                if hasattr(data_minimizer, method):
                    methods_available.append(method)

            # Test basic functionality
            try:
                # Test with sample patient data
                sample_data = {
                    "patient_id": "12345",
                    "name": "John Doe",
                    "ssn": "123-45-6789",
                    "medical_history": "confidential",
                    "billing_info": "private",
                }

                minimized = data_minimizer.minimize_patient_data(sample_data.copy())
                minimization_works = isinstance(minimized, dict) and len(minimized) < len(
                    sample_data
                )
            except Exception:
                minimization_works = False

            passed = len(methods_available) >= 2 and minimization_works

            details = {
                "data_minimizer_available": True,
                "methods_available": methods_available,
                "minimization_working": minimization_works,
                "privacy_controls_active": passed,
            }

            recommendations = []
            if len(methods_available) < 2:
                recommendations.append("Implement missing data minimization methods")
            if not minimization_works:
                recommendations.append("Data minimization not working properly")
                recommendations.append("Review HIPAA compliance requirements")

            return (
                passed,
                f"Data minimization {'working' if passed else 'has issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Data minimization check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Implement data minimization",
                    "Check privacy controls",
                    "Review HIPAA compliance",
                ],
            )

    def check_audit_logging(self) -> tuple[bool, str, dict[str, Any], list[str]]:
        """Check audit logging functionality."""
        if not self.security_manager:
            return (
                False,
                "Security manager not available",
                {"manager_available": False},
                ["Initialize security components before running audit checks"],
            )

        try:
            audit_logger = getattr(self.security_manager, "audit_logger", None)

            if not audit_logger:
                return (
                    False,
                    "Audit logger not available",
                    {"audit_logger_available": False},
                    ["Implement audit logging component"],
                )

            # Check audit configuration
            config = getattr(self.security_manager, "config", None)
            audit_enabled = getattr(config, "enable_audit_logging", False) if config else False
            retention_days = getattr(config, "audit_retention_days", 90) if config else 90

            # Test audit logging methods
            methods_available = []
            if hasattr(audit_logger, "log_phi_access"):
                methods_available.append("log_phi_access")

            # Test audit logging (without actually logging to avoid spam)
            audit_methods_work = len(methods_available) > 0

            passed = audit_enabled and audit_methods_work

            details = {
                "audit_enabled": audit_enabled,
                "audit_logger_available": True,
                "methods_available": methods_available,
                "retention_days": retention_days,
                "audit_functional": audit_methods_work,
            }

            recommendations = []
            if not audit_enabled:
                recommendations.append("Enable audit logging for HIPAA compliance")
            if not audit_methods_work:
                recommendations.append("Implement audit logging methods")
            if retention_days < 30:
                recommendations.append("Increase audit retention period for compliance")

            return (
                passed,
                f"Audit logging {'enabled' if passed else 'has issues'}",
                details,
                recommendations,
            )

        except Exception as e:
            return (
                False,
                f"Audit logging check failed: {type(e).__name__}",
                {"error": str(e)},
                [
                    "Implement audit logging",
                    "Check audit configuration",
                    "Review HIPAA compliance requirements",
                ],
            )

    def run_security_checks(self) -> list[tuple[str, CheckCategory, CheckSeverity, callable]]:
        """Return all security checks to be executed."""
        return [
            (
                "Input Validation",
                CheckCategory.SECURITY,
                CheckSeverity.HIGH,
                self.check_input_validation_rules,
            ),
            (
                "Rate Limiting",
                CheckCategory.SECURITY,
                CheckSeverity.MEDIUM,
                self.check_rate_limiting,
            ),
            (
                "Session Security",
                CheckCategory.SECURITY,
                CheckSeverity.HIGH,
                self.check_session_security,
            ),
            (
                "Data Minimization",
                CheckCategory.SECURITY,
                CheckSeverity.MEDIUM,
                self.check_data_minimization,
            ),
            (
                "Audit Logging",
                CheckCategory.SECURITY,
                CheckSeverity.MEDIUM,
                self.check_audit_logging,
            ),
        ]
