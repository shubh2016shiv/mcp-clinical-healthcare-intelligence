"""Data Minimization for HIPAA Minimum Necessary Principle

This module implements HIPAA's "minimum necessary" principle by filtering data
fields based on user roles and access requirements.

Rationale for Data Minimization in Healthcare:
- HIPAA Privacy Rule § 164.502(b): Use or disclose the minimum necessary information
- Reduce Breach Risk: Less data exposed means smaller breach impact
- Access Control: Role-based field filtering enforces least privilege
- Privacy Protection: Prevents unnecessary PHI exposure

Field Access Rules by Role (All Read-Only):
- ADMIN: Read access to all fields (system administration)
- CLINICIAN: Read access to clinical data + patient identifiers (patient care)
- RESEARCHER: Read access to aggregated/de-identified only (research, no individual identifiers)
- BILLING: Read access to financial + patient identifiers only (billing operations)
- READ_ONLY: Read access to limited view fields (minimal access)

Security Note: MCP servers are read-only by design. All data access is
read-only to ensure data server_health_checks and reduce attack surface. No write
operations are permitted through MCP tools.
"""

from typing import Any

from .authentication import UserRole


class DataMinimizer:
    """Implements HIPAA's 'minimum necessary' principle

    Rationale: HIPAA Privacy Rule § 164.502(b) requires covered entities
    to make reasonable efforts to limit PHI to the minimum necessary
    to accomplish the intended purpose.

    This class filters data fields based on user roles to ensure
    users only receive the data they need for their job functions.
    """

    def __init__(self):
        """Initialize data minimizer with role-based field permissions"""

        # Initialize Redis field schema cache if available
        self._field_schema_cache = None
        try:
            from src.mcp_server.cache import get_cache_manager

            cache_manager = get_cache_manager()
            if cache_manager.is_available():
                self._field_schema_cache = cache_manager.field_schema_cache
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Field schema cache not available: {e}")

        # Define field permissions for each role
        # These mappings implement the principle of least privilege
        self.role_field_permissions: dict[UserRole, set[str] | None] = {
            # ADMIN: Full read access to all fields (system administration)
            # Note: Read-only access - MCP servers never modify data
            UserRole.ADMIN: None,  # None = all fields allowed (read-only)
            # CLINICIAN: Clinical care requires comprehensive read access to patient data
            UserRole.CLINICIAN: {
                # Patient identification
                "patient_id",
                "first_name",
                "last_name",
                "birth_date",
                "gender",
                # Contact (for care coordination)
                "phone",
                "email",
                "address",
                "city",
                "state",
                # Clinical data (essential for patient care)
                "conditions",
                "medications",
                "allergies",
                "observations",
                "encounters",
                "procedures",
                "immunizations",
                # Clinical details
                "condition_name",
                "status",
                "onset_date",
                "severity",
                "medication_name",
                "dosage_instruction",
                "prescribed_date",
                "test_name",
                "value",
                "unit",
                "effective_date_time",
                "visit_reason",
                "location",
                "provider",
                "performed_date",
                # Metadata
                "resource_type",
                "source_id",
                "event_date",
                "event_type",
            },
            # RESEARCHER: Read access to aggregated/de-identified data only (HIPAA research exception)
            UserRole.RESEARCHER: {
                # Demographic aggregates only (no individual identifiers) - read-only
                "gender",
                "birth_year",
                "city",
                "state",
                # De-identified clinical data
                "condition_name",
                "medication_name",
                "test_name",
                "observation_type",
                "procedure_name",
                # Temporal data (for trend analysis)
                "year",
                "month",
                "age_group",
                # Aggregated counts and statistics
                "count",
                "average",
                "total",
                "percentage",
                "prevalence",
                # Research-safe metadata
                "resource_type",
                "event_type",
            },
            # BILLING: Read access to financial data for billing operations
            UserRole.BILLING: {
                # Patient identification (for billing) - read-only
                "patient_id",
                "first_name",
                "last_name",
                # Financial data
                "insurance_provider",
                "claim_amount",
                "charges",
                "billing_code",
                "diagnosis_codes",
                "procedure_codes",
                # Limited clinical (for billing codes)
                "condition_name",
                "procedure_name",
                # Temporal (for billing periods)
                "service_date",
                "billing_date",
                # Metadata
                "resource_type",
                "claim_status",
            },
            # READ_ONLY: Minimal read access for basic viewing
            UserRole.READ_ONLY: {
                # Basic identification only - read-only
                "patient_id",
                "first_name",
                "last_name",
                "birth_date",
                "gender",
                # Limited demographics
                "city",
                "state",
                # Condition fields - REQUIRED for condition queries to function
                # Rationale: The analyze_conditions tool must return meaningful data.
                # Without these fields, condition queries return null values even when
                # data exists in the database, breaking tool functionality. These are
                # basic condition attributes (name, status, dates) that are necessary
                # for any useful condition query. This maintains HIPAA's "minimum
                # necessary" principle while ensuring tools remain functional.
                # Enterprise consideration: In production, field permissions should be
                # validated through integration tests to catch missing fields early.
                "condition_name",  # Required: Condition name for identification
                "status",  # Required: Condition status (active/resolved/inactive)
                "onset_date",  # Required: When condition started (temporal context)
                "verification_status",  # Required: Verification status (confirmed/provisional)
                # Metadata only
                "resource_type",
            },
        }

    def filter_fields(self, data: dict[str, Any], role: UserRole) -> dict[str, Any]:
        """Filter data fields based on role

        Rationale: Different roles need different data. Researchers don't
        need patient names. Billing doesn't need clinical notes.
        This enforces HIPAA's minimum necessary principle.

        Args:
            data: Data dictionary to filter
            role: User role for access control

        Returns:
            Dict with only allowed fields for the role
        """
        if not isinstance(data, dict):
            return data

        # Admin gets all fields
        allowed_fields = self.role_field_permissions.get(role)
        if allowed_fields is None:
            return data

        # Filter fields based on role permissions
        filtered = {}
        for key, value in data.items():
            if key in allowed_fields:
                # Recursively filter nested objects/arrays
                filtered[key] = self._filter_nested_data(value, role)
            elif self._is_metadata_field(key):
                # Allow metadata fields for all roles
                filtered[key] = value

        return filtered

    def filter_record_list(
        self, records: list[dict[str, Any]], role: UserRole
    ) -> list[dict[str, Any]]:
        """Filter a list of records based on role permissions

        Rationale: Batch filtering for query results. Ensures consistent
        application of minimum necessary principle across all returned records.

        Args:
            records: List of data records
            role: User role for filtering

        Returns:
            List of filtered records
        """
        if not isinstance(records, list):
            # Return empty list instead of invalid type to maintain contract
            return []

        filtered = []
        for record in records:
            # Ensure each record is a dict before filtering
            if not isinstance(record, dict):
                continue
            filtered_record = self.filter_fields(record, role)
            # Ensure filter_fields returned a dict
            if isinstance(filtered_record, dict):
                filtered.append(filtered_record)

        return filtered

    def de_identify(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove direct identifiers for research use

        Rationale: De-identified data is not PHI under HIPAA and can be
        used more freely for research and analytics. This implements
        HIPAA's Safe Harbor method for de-identification.

        Args:
            data: Data to de-identify

        Returns:
            De-identified data dictionary
        """
        if not isinstance(data, dict):
            return data

        # Direct identifiers to remove (HIPAA Safe Harbor)
        identifiers_to_remove = {
            # Names
            "name",
            "first_name",
            "last_name",
            "full_name",
            "middle_name",
            # Geographic identifiers (more specific than zip code first 3 digits)
            "address",
            "street_address",
            "city",
            "state",
            "zip_code",
            "postal_code",
            # Contact information
            "phone",
            "phone_number",
            "email",
            "email_address",
            # IDs
            "patient_id",
            "medical_record_number",
            "mrn",
            "account_number",
            "certificate_number",
            "vehicle_identifier",
            "device_identifier",
            "url",
            "ip_address",
            # Biometric identifiers
            "biometric_identifier",
            "photo",
            "photograph",
            "fingerprint",
            "voiceprint",
            "iris_scan",
            "retina_scan",
            # Other unique identifiers
            "ssn",
            "social_security",
            "social_security_number",
        }

        de_identified = data.copy()

        # Remove direct identifiers
        for identifier in identifiers_to_remove:
            de_identified.pop(identifier, None)

        # Generalize dates to year only (not month/day)
        if "birth_date" in de_identified:
            birth_date = de_identified["birth_date"]
            if isinstance(birth_date, str) and len(birth_date) >= 4:
                de_identified["birth_year"] = birth_date[:4]
            del de_identified["birth_date"]

        # Generalize ages to age groups
        if "age" in de_identified:
            age = de_identified["age"]
            if isinstance(age, int | float):
                de_identified["age_group"] = self._categorize_age(age)
            del de_identified["age"]

        return de_identified

    def apply_minimum_necessary(self, data: Any, role: UserRole, operation: str = "read") -> Any:
        """Apply minimum necessary principle based on role and operation

        Rationale: Comprehensive data minimization that considers both
        role permissions and the specific operation being performed.

        Security Note: MCP servers are read-only. The operation parameter
        is maintained for API compatibility but only "read" operations
        are supported. All data access is read-only.

        Args:
            data: Data to minimize
            role: User role
            operation: Operation type (always "read" for MCP servers)

        Returns:
            Minimized data appropriate for role (read-only)
        """
        if role == UserRole.RESEARCHER:
            # Researchers get de-identified data
            if isinstance(data, dict):
                return self.de_identify(self.filter_fields(data, role))
            elif isinstance(data, list):
                return [self.de_identify(self.filter_fields(item, role)) for item in data]
            else:
                return data
        else:
            # Other roles get role-filtered data
            if isinstance(data, dict):
                return self.filter_fields(data, role)
            elif isinstance(data, list):
                return self.filter_record_list(data, role)
            else:
                return data

    def get_allowed_fields(self, role: UserRole) -> set[str] | None:
        """Get the set of allowed fields for a role

        Rationale: Expose field permissions for validation and documentation.
        Useful for API documentation and client applications.

        Uses Redis cache for performance optimization on repeated lookups.

        Args:
            role: User role

        Returns:
            Set of allowed field names, or None for all fields
        """
        # Try to get from Redis cache first
        if self._field_schema_cache:
            try:
                cached_schema = self._field_schema_cache.get_role_schema(role.value)
                if cached_schema is not None:
                    return cached_schema
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Failed to get schema from cache: {e}")

        # Get from memory and cache in Redis
        allowed_fields = self.role_field_permissions.get(role)

        # Cache in Redis for future lookups
        if allowed_fields is not None and self._field_schema_cache:
            try:
                self._field_schema_cache.cache_role_schema(role.value, allowed_fields)
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Failed to cache schema in Redis: {e}")

        return allowed_fields

    def validate_field_access(self, field: str, role: UserRole, operation: str = "read") -> bool:
        """Validate if a field can be accessed by a role (read-only)

        Rationale: Permission checking for individual fields.
        Useful for fine-grained access control decisions.

        Security Note: MCP servers are read-only. Only read operations
        are permitted. The operation parameter is maintained for API
        compatibility but only "read" is supported.

        Args:
            field: Field name to check
            role: User role
            operation: Operation type (always "read" for MCP servers)

        Returns:
            bool: True if field read access is allowed
        """
        allowed_fields = self.role_field_permissions.get(role)

        # Admin can access all fields
        if allowed_fields is None:
            return True

        # Check if field is in allowed set
        if field in allowed_fields:
            return True

        # Check if it's a metadata field (allowed for all roles)
        if self._is_metadata_field(field):
            return True

        return False

    def _filter_nested_data(self, data: Any, role: UserRole) -> Any:
        """Recursively filter nested data structures

        Rationale: Data often contains nested objects and arrays.
        Ensure filtering applies to all levels of the data structure.

        Args:
            data: Nested data to filter
            role: User role for filtering

        Returns:
            Filtered nested data
        """
        if isinstance(data, dict):
            return self.filter_fields(data, role)
        elif isinstance(data, list):
            return [self._filter_nested_data(item, role) for item in data]
        else:
            # Primitive values pass through unchanged
            return data

    def _is_metadata_field(self, field: str) -> bool:
        """Check if a field is metadata that should be accessible to all roles

        Rationale: Certain fields like IDs, timestamps, and resource types
        are necessary for basic functionality and don't contain sensitive PHI.

        Args:
            field: Field name to check

        Returns:
            bool: True if field is metadata
        """
        metadata_fields = {
            "id",
            "_id",
            "resource_id",
            "source_id",
            "resource_type",
            "created_at",
            "updated_at",
            "timestamp",
            "version",
            "event_type",
            "event_date",
            "count",
            "total",
            "index",
        }

        return field.lower() in metadata_fields

    def _categorize_age(self, age: float) -> str:
        """Categorize age into groups for de-identification

        Rationale: Age groups provide useful demographic information
        for research while protecting individual privacy.

        Args:
            age: Numeric age

        Returns:
            str: Age group category
        """
        if age < 18:
            return "0-17"
        elif age < 30:
            return "18-29"
        elif age < 50:
            return "30-49"
        elif age < 70:
            return "50-69"
        else:
            return "70+"
