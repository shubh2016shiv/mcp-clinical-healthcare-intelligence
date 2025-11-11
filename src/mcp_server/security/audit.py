"""Audit Logging for HIPAA Compliance

This module implements comprehensive audit logging for all PHI (Protected Health Information)
access in the healthcare MCP server. Audit logging is a critical HIPAA Security Rule requirement.

Rationale for Audit Logging in Healthcare:
- HIPAA Security Rule § 164.312(b): Implement mechanisms to record and examine activity
- Breach Investigation: Audit logs enable investigation of security incidents
- Compliance Evidence: Demonstrates adherence to access controls
- Deterrence: Knowledge of logging reduces unauthorized access attempts

Audit Requirements:
- Who accessed data (user_id, role, IP)
- What was accessed (resource type, specific IDs)
- When accessed (timestamp)
- Why accessed (operation/purpose)
- Result (success/failure, records returned)
"""

import json
import logging
from datetime import datetime
from typing import Any

from .authentication import SecurityContext
from .config import SecurityConfig


class AuditLogger:
    """Comprehensive audit logging for PHI access

    Rationale: HIPAA Security Rule § 164.312(b) requires:
    'Implement hardware, software, and/or procedural mechanisms that
    record and examine activity in information systems that contain or
    use electronic protected health information.'

    Every PHI access must be logged with:
    - Who accessed it (user_id, role, IP)
    - What was accessed (resource type, IDs)
    - When it was accessed (timestamp)
    - Why it was accessed (purpose/operation)
    - Result (success/failure, count)

    Audit logs are stored separately from application logs for security and compliance.
    """

    def __init__(self, config: SecurityConfig):
        """Initialize audit logger with security configuration

        Args:
            config: Security configuration with audit settings
        """
        self.config = config

        # Create separate audit logger
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)

        # Configure audit file handler
        audit_log_path = getattr(config, "audit_log_path", "audit.log")  # Default if not set
        audit_handler = logging.FileHandler(audit_log_path)
        audit_handler.setLevel(logging.INFO)

        # Create JSON formatter for structured audit logs
        formatter = logging.Formatter(
            "%(asctime)s - AUDIT - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        audit_handler.setFormatter(formatter)

        # Add handler if not already present
        if not self.audit_logger.handlers:
            self.audit_logger.addHandler(audit_handler)

        # Also keep reference to regular logger for non-audit events
        self.logger = logging.getLogger(__name__)

    def log_phi_access(
        self,
        context: SecurityContext,
        operation: str,
        resource_type: str,
        resource_ids: list[str],
        query_params: dict[str, Any],
        result_count: int,
        success: bool,
        error: str | None = None,
    ):
        """Log PHI access for audit trail

        All parameters are required for HIPAA compliance and
        breach investigation if security incident occurs.

        Rationale: Comprehensive logging enables:
        - Security incident investigation
        - Compliance auditing
        - Access pattern analysis
        - Breach notification compliance

        Args:
            context: Security context of the accessing user
            operation: Operation being performed (e.g., "search_patients")
            resource_type: Type of resource accessed (e.g., "patient", "condition")
            resource_ids: Specific resource IDs accessed
            query_params: Query parameters used (redacted for security)
            result_count: Number of records returned
            success: Whether operation succeeded
            error: Error message if operation failed
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": context.user_id,
            "role": context.role.value,
            "session_id": self._redact_session_id(context.session_id),
            "ip_address": context.ip_address,
            "operation": operation,
            "resource_type": resource_type,
            "resource_ids": resource_ids[:10],  # Limit for log size
            "resource_count": len(resource_ids),
            "query_params": self._redact_sensitive_params(query_params),
            "result_count": result_count,
            "success": success,
            "error": error,
            "compliance": {
                "hipaa_required": True,
                "audit_retention_years": 7,
                "data_minimization_applied": self._check_data_minimization(context, resource_type),
            },
        }

        # Log to audit file
        self.audit_logger.info(f"PHI_ACCESS: {json.dumps(audit_entry, default=str)}")

        # Also log security-relevant events to regular log
        if not success or result_count > 100:  # High-volume or failed access
            log_level = logging.WARNING if not success else logging.INFO
            self.logger.log(
                log_level,
                f"PHI Access: {operation} by {context.user_id} ({context.role.value}) - "
                f"{result_count} records, success={success}",
                extra={
                    "user_id": context.user_id,
                    "operation": operation,
                    "resource_count": result_count,
                    "success": success,
                },
            )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: dict[str, Any],
        context: SecurityContext | None = None,
    ):
        """Log security-related events (non-PHI access)

        Rationale: Security events like failed authentications, rate limit hits,
        and system security issues need separate tracking from PHI access logs.

        Args:
            event_type: Type of security event
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
            details: Event details
            context: Security context if available
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "compliance": {"hipaa_required": False, "security_monitoring": True},
        }

        if context:
            audit_entry.update(
                {
                    "user_id": context.user_id,
                    "role": context.role.value,
                    "ip_address": context.ip_address,
                }
            )

        self.audit_logger.info(f"SECURITY_EVENT: {json.dumps(audit_entry, default=str)}")

        # Log critical security events to regular logger
        if severity in ["ERROR", "CRITICAL"]:
            self.logger.error(
                f"Security Event: {event_type} - {details}",
                extra={"event_type": event_type, "severity": severity},
            )

    def log_authentication_event(
        self,
        user_id: str,
        success: bool,
        ip_address: str,
        method: str = "unknown",
        error: str | None = None,
    ):
        """Log authentication attempts

        Rationale: Authentication failures are critical security events
        that must be tracked for HIPAA compliance and security monitoring.

        Args:
            user_id: User attempting authentication
            success: Whether authentication succeeded
            ip_address: Client IP address
            method: Authentication method used
            error: Error message if authentication failed
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "authentication",
            "user_id": user_id,
            "success": success,
            "ip_address": ip_address,
            "method": method,
            "error": error,
            "compliance": {"hipaa_required": True, "audit_retention_years": 7},
        }

        self.audit_logger.info(f"AUTH_EVENT: {json.dumps(audit_entry, default=str)}")

        if not success:
            self.logger.warning(
                f"Authentication failed for user {user_id} from {ip_address}: {error}",
                extra={"user_id": user_id, "ip_address": ip_address},
            )

    def log_rate_limit_event(self, identifier: str, limit: int, window_seconds: int = 60):
        """Log rate limit violations

        Rationale: Rate limiting prevents abuse and DoS attacks.
        Violations should be logged for security monitoring.

        Args:
            identifier: Client identifier that hit the limit
            limit: Rate limit that was exceeded
            window_seconds: Time window for the limit
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "rate_limit_exceeded",
            "identifier": identifier,
            "limit": limit,
            "window_seconds": window_seconds,
            "compliance": {"hipaa_required": False, "security_monitoring": True},
        }

        self.audit_logger.info(f"RATE_LIMIT: {json.dumps(audit_entry, default=str)}")

        self.logger.warning(
            f"Rate limit exceeded for {identifier}: {limit} requests per {window_seconds}s",
            extra={"identifier": identifier, "limit": limit},
        )

    def _redact_session_id(self, session_id: str) -> str:
        """Redact session ID for audit logs

        Rationale: Session IDs are sensitive and should not appear
        in audit logs in plain text, even though they're temporary.

        Args:
            session_id: Session ID to redact

        Returns:
            str: Redacted session ID
        """
        if not session_id or len(session_id) < 8:
            return "[INVALID]"

        # Show first 4 and last 4 characters
        return f"{session_id[:4]}...{session_id[-4:]}"

    def _redact_sensitive_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Redact PII from logged query params

        Rationale: Audit logs themselves should not contain excessive PII.
        Log structure and access patterns, not full PHI content.
        This balances security monitoring with privacy protection.

        Args:
            params: Query parameters to redact

        Returns:
            Dict with sensitive fields redacted
        """
        if not self.config.enable_pii_redaction:
            return params

        sensitive_fields = {
            "ssn",
            "social_security",
            "phone",
            "email",
            "address",
            "birth_date",
            "first_name",
            "last_name",
            "full_name",
            "patient_id",
            "medical_record_number",
            "account_number",
        }

        redacted = {}
        for key, value in params.items():
            if key.lower() in sensitive_fields:
                redacted[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 50:
                # Truncate long strings in logs
                redacted[key] = f"{value[:47]}..."
            else:
                redacted[key] = value

        return redacted

    def _check_data_minimization(self, context: SecurityContext, resource_type: str) -> bool:
        """Check if data minimization was applied based on role

        Rationale: Audit logs should indicate whether HIPAA's "minimum necessary"
        principle was followed based on the user's role.

        Args:
            context: Security context
            resource_type: Type of resource accessed

        Returns:
            bool: True if data minimization should be applied
        """
        from .authentication import UserRole

        # Researchers and billing users should have data minimization
        roles_with_minimization = {UserRole.RESEARCHER, UserRole.BILLING, UserRole.READ_ONLY}

        # PHI resources should have minimization applied
        phi_resources = {"patient", "condition", "medication", "observation", "encounter"}

        return context.role in roles_with_minimization and resource_type in phi_resources

    def get_audit_summary(self, days: int = 7) -> dict[str, Any]:
        """Get audit log summary for compliance reporting

        Rationale: Enables periodic review of access patterns for HIPAA compliance.
        Provides summary statistics without exposing individual PHI access details.

        Args:
            days: Number of days to summarize

        Returns:
            Dict with audit summary statistics
        """
        # In a real implementation, this would read from audit logs
        # For now, return a placeholder structure
        return {
            "period_days": days,
            "total_access_events": 0,
            "failed_access_attempts": 0,
            "high_volume_access": 0,
            "unique_users": 0,
            "compliance_status": "audit_logging_enabled",
            "next_review_date": (datetime.utcnow().date()).isoformat(),
        }
