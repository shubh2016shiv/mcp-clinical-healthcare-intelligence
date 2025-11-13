"""Security Configuration with HIPAA-Compliant Defaults

This module provides centralized security configuration for the healthcare MCP server.
All security settings are consolidated here to prevent scattered configuration that
could lead to misconfigurations and security gaps.

Rationale: Centralized security config ensures:
- Consistent security policies across all components
- Prevention of misconfiguration through safe defaults
- Easy auditing and compliance verification
- Environment-variable-based configuration for deployment flexibility

HIPAA Compliance Notes:
- Audit retention set to 7 years (HIPAA requirement)
- Rate limiting prevents bulk PHI extraction
- Query limits enforce "minimum necessary" principle
- Field whitelisting prevents unauthorized data access
"""

from pydantic import BaseModel, Field, field_validator


class SecurityConfig(BaseModel):
    """Centralized security configuration with safe defaults

    Rationale: Centralized config prevents scattered security settings
    that might be accidentally misconfigured across the codebase.
    All settings have HIPAA-compliant defaults that prioritize security.
    """

    # Authentication settings
    api_key_min_length: int = Field(
        default=32, description="Minimum API key length for secure authentication"
    )
    session_timeout_minutes: int = Field(
        default=30, description="Session timeout to prevent stale authenticated sessions"
    )
    max_failed_auth_attempts: int = Field(
        default=5, description="Lockout threshold to prevent brute force attacks"
    )
    lockout_duration_minutes: int = Field(
        default=15, description="Account lockout time after failed attempts"
    )

    # Rate limiting (prevents abuse and DoS)
    rate_limit_requests_per_minute: int = Field(
        default=60, description="Max requests per minute per client to prevent DoS attacks"
    )
    rate_limit_burst: int = Field(
        default=10, description="Burst allowance for legitimate traffic spikes"
    )

    # Query limits (prevents excessive PHI exposure)
    max_query_results: int = Field(
        default=100, description="Max records per query to prevent bulk PHI extraction"
    )
    max_query_depth: int = Field(
        default=3, description="Max aggregation pipeline depth to prevent complex attacks"
    )

    # Audit settings
    enable_audit_logging: bool = Field(
        default=True, description="Enable PHI access audit trail (required by HIPAA)"
    )
    audit_log_path: str = Field(
        default="audit.log", description="Path for HIPAA-compliant audit logging"
    )
    audit_retention_days: int = Field(
        default=2555,  # 7 years
        description="Audit log retention period (7 years per HIPAA)",
    )

    # Data sanitization
    enable_pii_redaction: bool = Field(
        default=False, description="Redact PII in logs (balance between security and debugging)"
    )
    allowed_query_fields: list[str] = Field(
        default_factory=lambda: [
            # Explicitly whitelist safe query fields to prevent unauthorized access
            # Rationale: Only allow querying fields that are safe and necessary
            "patient_id",
            "first_name",
            "last_name",
            "city",
            "state",
            "address.city",  # Nested address fields for patient search
            "address.state",  # Nested address fields for patient search
            "birth_date",
            "gender",
            "medication_name",
            "condition_name",
            "insurance_provider",
            "claim_amount",
            "procedure_name",
            "observation_type",
            "test_name",
            "value",
            "unit",
            "prescribed_date",
            "performed_date",
            "administration_date",
            "onset_date",
            "status",
            "verification_status",
            "severity",
            "encounter_type",
            "visit_reason",
            "location",
            "provider",
            "drug_name",
            "therapeutic_class",
            "drug_class",
            "drug_subclass",
            "rxcui",
            "ingredient_rxcui",
            "group_by",
            "min_count",
            "limit",
        ],
        description="Whitelisted fields allowed in queries to prevent data exposure",
    )

    # Security monitoring
    enable_security_monitoring: bool = Field(
        default=True, description="Enable real-time security monitoring and alerting"
    )
    suspicious_activity_threshold: int = Field(
        default=10, description="Threshold for triggering security alerts"
    )

    @field_validator("api_key_min_length")
    @classmethod
    def validate_api_key_length(cls, value: int) -> int:
        """Validate API key minimum length.

        Rationale: Short API keys are vulnerable to brute force attacks.
        Minimum 32 characters provides adequate security for healthcare systems.
        """
        if value < 16:
            raise ValueError("API key minimum length must be at least 16 characters")
        return value

    @field_validator("max_query_results")
    @classmethod
    def validate_max_results(cls, value: int) -> int:
        """Validate maximum query results limit.

        Rationale: Unbounded queries can expose excessive PHI or cause
        performance issues. Limit enforces HIPAA's "minimum necessary" principle.
        """
        if value < 1:
            raise ValueError("max_query_results must be at least 1")
        if value > 1000:
            raise ValueError("max_query_results cannot exceed 1000 for security")
        return value

    @field_validator("audit_retention_days")
    @classmethod
    def validate_audit_retention(cls, value: int) -> int:
        """Validate audit retention period.

        Rationale: HIPAA requires 7 years of audit logs for breach investigation.
        This ensures compliance while allowing reasonable retention periods.
        """
        if value < 365:  # At least 1 year
            raise ValueError("audit_retention_days must be at least 365")
        if value > 3650:  # Max 10 years
            raise ValueError("audit_retention_days cannot exceed 3650 (10 years)")
        return value

    @field_validator("allowed_query_fields")
    @classmethod
    def validate_allowed_fields(cls, value: list[str]) -> list[str]:
        """Validate allowed query fields.

        Rationale: Field whitelisting prevents unauthorized access to sensitive
        fields. This list should be carefully reviewed for each deployment.
        """
        if not value:
            raise ValueError("allowed_query_fields cannot be empty")

        # Check for dangerous field names that could indicate security issues
        dangerous_patterns = ["password", "secret", "key", "token", "ssn", "social"]
        for field in value:
            if any(pattern in field.lower() for pattern in dangerous_patterns):
                raise ValueError(f"Potentially sensitive field not allowed: {field}")

        return value

    def get_safe_config_dict(self) -> dict:
        """Get configuration as dict with sensitive data redacted.

        Rationale: Prevents accidental exposure of sensitive configuration
        in logs or error messages while still providing useful debugging info.
        """
        config_dict = self.model_dump()
        # Note: In a real implementation, you might want to redact more fields
        # For now, this config doesn't contain sensitive data like passwords
        return config_dict

    def is_hipaa_compliant(self) -> bool:
        """Check if current configuration meets HIPAA requirements.

        Rationale: Automated compliance checking prevents deployment
        with insecure configurations that could lead to HIPAA violations.
        """
        checks = [
            self.enable_audit_logging,  # HIPAA requires audit trails
            self.audit_retention_days >= 2555,  # 7 years minimum
            self.max_query_results <= 500,  # Reasonable limit for PHI access
            len(self.allowed_query_fields) > 0,  # Must have field restrictions
            self.session_timeout_minutes <= 480,  # 8 hours max (reasonable)
        ]
        return all(checks)
