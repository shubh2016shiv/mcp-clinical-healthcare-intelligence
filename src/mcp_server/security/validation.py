"""Input Validation & Sanitization for Healthcare MCP Server

This module implements comprehensive input validation and sanitization to prevent
NoSQL injection attacks and other security vulnerabilities in healthcare data queries.

Rationale for Input Validation in Healthcare:
- NoSQL Injection Prevention: MongoDB operators can bypass access controls
- Data Integrity: Ensures queries match expected patterns
- HIPAA Compliance: Prevents unauthorized data access
- Defense in Depth: Multiple validation layers

Key Security Threats Addressed:
- NoSQL injection via $ operators
- Command injection in query parameters
- Path traversal attacks
- Cross-site scripting (XSS) in stored data
- Buffer overflow via oversized inputs
"""

import logging
import re
from typing import Any

from .config import SecurityConfig


class InputValidator:
    """Validates and sanitizes user inputs to prevent injection attacks

    Rationale: Healthcare data often contains special characters that could
    be exploited in NoSQL injection attacks. Validation prevents:
    - NoSQL injection (MongoDB operator injection)
    - Command injection in query parameters
    - Path traversal attacks
    - Cross-site scripting (if data is displayed in web UI)

    Validation Strategy:
    - Whitelist allowed query fields
    - Detect and block dangerous operators
    - Sanitize string inputs
    - Enforce length limits
    - Validate data types and formats
    """

    def __init__(self, config: SecurityConfig):
        """Initialize input validator with security configuration

        Args:
            config: Security configuration with validation rules
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Dangerous MongoDB operators that should never come from user input
        # Rationale: These operators can bypass access controls or execute arbitrary queries
        self.dangerous_operators = {
            "$where",
            "$regex",
            "$expr",
            "$function",
            "$accumulator",
            "$let",
            "$reduce",
            "$map",
            "$filter",
            "$lookup",
            "$unionWith",
            "$out",
            "$merge",
            "$replaceRoot",
            "$addFields",
            "$project",
            "$match",
            "$group",
            "$sort",
            "$limit",
            "$skip",
            "$count",
            "$facet",
            "$bucket",
            "$bucketAuto",
            "$sortByCount",
        }

        # Additional dangerous patterns
        self.dangerous_patterns = [
            r"\.\.",  # Path traversal (..)
            r"[\x00-\x1F\x7F-\x9F]",  # Control characters
            r"<script",  # XSS attempts (case insensitive)
            r"javascript:",  # JavaScript injection
        ]

        # Safe alphanumeric pattern for IDs (allows hyphens and underscores)
        self.id_pattern = re.compile(r"^[a-zA-Z0-9\-_]+$")

        # Safe text pattern for names (letters, numbers, spaces, hyphens, apostrophes)
        # Note: Numbers are allowed for synthetic test data (e.g., Synthea generates names like "Dewitt635")
        self.name_pattern = re.compile(r"^[a-zA-Z0-9\s\-']+$")

        # Email pattern for validation
        self.email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    def validate_query_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize query parameters

        Rationale: Prevents NoSQL injection by ensuring only safe,
        expected fields are queried with safe operators. This is the
        primary defense against injection attacks.

        Args:
            params: Query parameters to validate

        Returns:
            Dict of sanitized parameters

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(params, dict):
            raise ValueError("Query parameters must be a dictionary")

        sanitized = {}

        for key, value in params.items():
            # Skip None values
            if value is None:
                continue

            # 1. Whitelist check: Only allow expected query fields
            if key not in self.config.allowed_query_fields:
                self.logger.warning(f"Rejected non-whitelisted field: {key}")
                continue

            # 2. Check for operator injection in keys
            if self._contains_dangerous_operators(str(key)):
                self.logger.error(f"Dangerous operator detected in key: {key}")
                raise ValueError(f"Invalid query parameter: {key}")

            # 3. Sanitize value based on expected type
            try:
                sanitized[key] = self._sanitize_value(key, value)
            except ValueError as e:
                self.logger.error(f"Value sanitization failed for {key}: {e}")
                raise

        return sanitized

    def validate_patient_id(self, patient_id: str) -> str:
        """Validate patient ID format

        Rationale: Patient IDs are high-risk inputs that appear in many queries.
        Strict validation prevents injection and ensures data integrity.
        Only allows safe alphanumeric characters.

        Args:
            patient_id: Patient ID to validate

        Returns:
            str: Validated patient ID

        Raises:
            ValueError: If patient ID is invalid
        """
        if not patient_id or not isinstance(patient_id, str):
            raise ValueError("Patient ID must be a non-empty string")

        if len(patient_id) > 100:
            raise ValueError("Patient ID too long (max 100 characters)")

        if not self.id_pattern.match(patient_id):
            raise ValueError("Patient ID contains invalid characters")

        return patient_id

    def validate_limit(self, limit: int | None) -> int:
        """Validate result limit to prevent excessive PHI exposure

        Rationale: Limits prevent bulk PHI extraction. HIPAA requires
        "minimum necessary" - users should only receive data they need.
        Prevents DoS attacks via huge result sets.

        Args:
            limit: Requested limit or None

        Returns:
            int: Validated limit within bounds
        """
        if limit is None:
            return self.config.max_query_results

        if not isinstance(limit, int) or limit < 1:
            return self.config.max_query_results

        if limit > self.config.max_query_results:
            self.logger.warning(
                f"Limit {limit} exceeds max {self.config.max_query_results}, capping to maximum"
            )
            return self.config.max_query_results

        return limit

    def validate_date_range(
        self, start_date: str | None, end_date: str | None
    ) -> tuple[str | None, str | None]:
        """Validate date range parameters

        Rationale: Date ranges are common in healthcare queries but can be
        manipulated for data exfiltration. Validates format and logical consistency.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            Tuple of validated dates

        Raises:
            ValueError: If date validation fails
        """
        from datetime import datetime

        validated_start = None
        validated_end = None

        date_format = "%Y-%m-%d"

        if start_date:
            try:
                validated_start = datetime.strptime(start_date, date_format).date().isoformat()
            except ValueError as e:
                raise ValueError(
                    f"Invalid start_date format: {start_date} (expected YYYY-MM-DD)"
                ) from e

        if end_date:
            try:
                validated_end = datetime.strptime(end_date, date_format).date().isoformat()
            except ValueError as e:
                raise ValueError(
                    f"Invalid end_date format: {end_date} (expected YYYY-MM-DD)"
                ) from e

        # Validate logical consistency
        if validated_start and validated_end and validated_start > validated_end:
            raise ValueError("start_date cannot be after end_date")

        return validated_start, validated_end

    def validate_name_field(self, name: str, field_name: str) -> str:
        """Validate name fields (first_name, last_name, etc.)

        Rationale: Names are common query parameters but can contain
        dangerous characters. Validates against safe patterns.

        Args:
            name: Name string to validate
            field_name: Field name for error messages

        Returns:
            str: Validated name

        Raises:
            ValueError: If name validation fails
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"{field_name} must be a non-empty string")

        if len(name) > 100:
            raise ValueError(f"{field_name} too long (max 100 characters)")

        if not self.name_pattern.match(name.strip()):
            raise ValueError(f"{field_name} contains invalid characters")

        return name.strip()

    def _sanitize_value(self, key: str, value: Any) -> Any:
        """Sanitize a parameter value based on its expected type

        Rationale: Different parameter types need different sanitization.
        Ensures type safety and prevents injection attacks.

        Args:
            key: Parameter key
            value: Value to sanitize

        Returns:
            Sanitized value

        Raises:
            ValueError: If sanitization fails
        """
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, int | float):
            return value
        elif isinstance(value, bool):
            return value
        elif isinstance(value, dict):
            # Recursively validate nested objects
            return self.validate_query_params(value)
        elif isinstance(value, list):
            # Validate list contents
            return [self._sanitize_value(key, item) for item in value]
        else:
            raise ValueError(f"Unsupported type for {key}: {type(value)}")

    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input

        Rationale: Remove potentially dangerous characters while
        preserving legitimate medical terminology and names.
        Prevents injection attacks while allowing valid data.

        Args:
            value: String to sanitize

        Returns:
            str: Sanitized string

        Raises:
            ValueError: If dangerous content detected
        """
        if not isinstance(value, str):
            raise ValueError("Value must be a string")

        # Check for dangerous operators
        if self._contains_dangerous_operators(value):
            raise ValueError(f"Dangerous operator detected in value: {value}")

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(f"Dangerous pattern detected in value: {value}")

        # Remove null bytes (can cause parsing issues)
        value = value.replace("\x00", "")

        # Limit length to prevent DoS via huge strings
        max_length = 500
        if len(value) > max_length:
            self.logger.warning(f"Truncating oversized input: {len(value)} chars")
            value = value[:max_length]

        return value

    def _contains_dangerous_operators(self, text: str) -> bool:
        """Check if text contains dangerous MongoDB operators

        Rationale: MongoDB operators like $where, $regex can execute
        arbitrary JavaScript or bypass query restrictions.

        Args:
            text: Text to check

        Returns:
            bool: True if dangerous operators found
        """
        if not isinstance(text, str):
            return False

        # Check for exact operator matches
        for operator in self.dangerous_operators:
            if operator in text:
                return True

        return False

    def validate_query_depth(self, query: dict[str, Any]) -> bool:
        """Validate aggregation pipeline depth

        Rationale: Deep aggregation pipelines can be used for DoS attacks
        or to bypass security controls. Limits prevent abuse.

        Args:
            query: Query to validate

        Returns:
            bool: True if within depth limits
        """

        def get_depth(obj: Any, current_depth: int = 0) -> int:
            """Recursively calculate object depth"""
            if current_depth > self.config.max_query_depth:
                return current_depth

            if isinstance(obj, dict):
                return max(get_depth(value, current_depth + 1) for value in obj.values())
            elif isinstance(obj, list):
                return max(get_depth(item, current_depth + 1) for item in obj)
            return current_depth

        depth = get_depth(query)
        if depth > self.config.max_query_depth:
            raise ValueError(f"Query depth {depth} exceeds maximum {self.config.max_query_depth}")

        return True
