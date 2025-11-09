#!/usr/bin/env python3
"""
Deduplication Module - Enterprise Edition

Content-based deduplication with hash-based duplicate detection for ETL pipelines.

Features:
- Hash-based duplicate detection
- Configurable deduplication rules per collection
- Content normalization for consistent hashing
- Performance-optimized duplicate checking
- Integration with data quality framework
"""

import hashlib
import threading
from collections import defaultdict
from typing import Any

from healthcare_data_pipeline.connection_manager import get_database
from healthcare_data_pipeline.metrics import record_failure
from healthcare_data_pipeline.structured_logging import get_logger

logger = get_logger(__name__)


class DeduplicationRule:
    """Deduplication rule for a collection."""

    def __init__(
        self,
        collection_name: str,
        fields: list[str],
        hash_algorithm: str = "sha256",
        normalize_content: bool = True,
        case_sensitive: bool = False,
    ):
        """Initialize deduplication rule.

        Args:
            collection_name: Collection name this rule applies to
            fields: List of fields to include in deduplication hash
            hash_algorithm: Hash algorithm to use
            normalize_content: Whether to normalize content before hashing
            case_sensitive: Whether field comparison is case sensitive
        """
        self.collection_name = collection_name
        self.fields = fields
        self.hash_algorithm = hash_algorithm
        self.normalize_content = normalize_content
        self.case_sensitive = case_sensitive

    def generate_hash(self, record: dict[str, Any]) -> str:
        """Generate deduplication hash for a record.

        Args:
            record: Record to generate hash for

        Returns:
            Hash string
        """
        # Extract relevant fields
        content_parts = []

        for field in self.fields:
            value = record.get(field)
            if value is not None:
                # Normalize content if requested
                if self.normalize_content:
                    value = self._normalize_value(value)

                if not self.case_sensitive and isinstance(value, str):
                    value = value.lower()

                content_parts.append(f"{field}:{value}")

        # Sort for consistent ordering
        content_parts.sort()

        # Create content string
        content_string = "|".join(content_parts)

        # Generate hash
        hash_obj = hashlib.new(self.hash_algorithm)
        hash_obj.update(content_string.encode("utf-8"))

        return hash_obj.hexdigest()

    def _normalize_value(self, value: Any) -> Any:
        """Normalize value for consistent hashing.

        Args:
            value: Value to normalize

        Returns:
            Normalized value
        """
        if isinstance(value, str):
            # Normalize whitespace and remove extra spaces
            return " ".join(value.split())
        elif isinstance(value, int | float):
            # Convert numbers to string representation
            return str(value)
        elif isinstance(value, list):
            # Sort lists for consistent ordering
            return sorted([self._normalize_value(item) for item in value])
        elif isinstance(value, dict):
            # Sort dict keys and normalize values
            return {k: self._normalize_value(v) for k, v in sorted(value.items())}
        else:
            # Convert to string for other types
            return str(value)


class DeduplicationEngine:
    """Engine for content-based deduplication."""

    def __init__(self):
        """Initialize deduplication engine."""
        self.rules: dict[str, DeduplicationRule] = {}
        self._hash_cache: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

        # Set up default rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up default deduplication rules for collections."""

        # Patients - deduplicate by name and birth date
        self.add_rule(
            DeduplicationRule(
                collection_name="clean_patients",
                fields=["first_name", "last_name", "birth_date"],
                case_sensitive=False,
            )
        )

        # Conditions - deduplicate by patient and condition name
        self.add_rule(
            DeduplicationRule(
                collection_name="clean_conditions",
                fields=["patient_id", "condition_name"],
                case_sensitive=False,
            )
        )

        # Observations - deduplicate by patient, test name, and date
        self.add_rule(
            DeduplicationRule(
                collection_name="clean_observations",
                fields=["patient_id", "test_name", "test_date"],
                case_sensitive=False,
            )
        )

        # Medications - deduplicate by patient, medication name, and date
        self.add_rule(
            DeduplicationRule(
                collection_name="clean_medications",
                fields=["patient_id", "medication_name", "prescribed_date"],
                case_sensitive=False,
            )
        )

        # Encounters - deduplicate by patient and start date
        self.add_rule(
            DeduplicationRule(
                collection_name="clean_encounters",
                fields=["patient_id", "start_date"],
                case_sensitive=False,
            )
        )

    def add_rule(self, rule: DeduplicationRule) -> None:
        """Add a deduplication rule.

        Args:
            rule: Deduplication rule to add
        """
        with self._lock:
            self.rules[rule.collection_name] = rule
            logger.info(f"Added deduplication rule for {rule.collection_name}")

    def get_rule(self, collection_name: str) -> DeduplicationRule | None:
        """Get deduplication rule for a collection.

        Args:
            collection_name: Collection name

        Returns:
            Deduplication rule or None if not found
        """
        return self.rules.get(collection_name)

    def check_duplicate(
        self, collection_name: str, record: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Check if a record is a duplicate.

        Args:
            collection_name: Collection name
            record: Record to check

        Returns:
            Tuple of (is_duplicate, duplicate_id)
        """
        rule = self.get_rule(collection_name)
        if not rule:
            return False, None

        try:
            # Generate hash for the record
            record_hash = rule.generate_hash(record)

            # Check cache first
            if record_hash in self._hash_cache[collection_name]:
                return True, None

            # Check database
            db = get_database()
            collection = db[collection_name]

            # Look for existing record with same hash
            existing = collection.find_one({"_deduplication_hash": record_hash})

            if existing:
                return True, str(existing["_id"])

            # Not a duplicate, add to cache
            self._hash_cache[collection_name].add(record_hash)
            return False, None

        except Exception as e:
            logger.warning(f"Error checking for duplicates in {collection_name}: {e}")
            return False, None

    def mark_record_hash(
        self, collection_name: str, record: dict[str, Any], record_id: str
    ) -> None:
        """Mark a record with its deduplication hash.

        Args:
            collection_name: Collection name
            record: Record data
            record_id: Record ID
        """
        rule = self.get_rule(collection_name)
        if not rule:
            return

        try:
            record_hash = rule.generate_hash(record)

            # Update record with hash
            db = get_database()
            collection = db[collection_name]

            collection.update_one(
                {"_id": record_id}, {"$set": {"_deduplication_hash": record_hash}}
            )

            # Add to cache
            self._hash_cache[collection_name].add(record_hash)

        except Exception as e:
            logger.warning(f"Error marking record hash for {record_id}: {e}")

    def find_duplicates(self, collection_name: str, batch_size: int = 1000) -> list[dict[str, Any]]:
        """Find all duplicate records in a collection.

        Args:
            collection_name: Collection name
            batch_size: Batch size for processing

        Returns:
            List of duplicate groups (each group has original and duplicates)
        """
        rule = self.get_rule(collection_name)
        if not rule:
            return []

        try:
            db = get_database()
            collection = db[collection_name]

            # Group by hash to find duplicates
            pipeline = [
                {"$match": {"_deduplication_hash": {"$exists": True}}},
                {
                    "$group": {
                        "_id": "$_deduplication_hash",
                        "records": {"$push": {"_id": "$_id", "data": "$$ROOT"}},
                        "count": {"$sum": 1},
                    }
                },
                {"$match": {"count": {"$gt": 1}}},
                {"$limit": batch_size},
            ]

            duplicates = []
            for group in collection.aggregate(pipeline):
                hash_value = group["_id"]
                records = group["records"]

                # Sort by creation time or ID to determine "original"
                records.sort(key=lambda r: str(r["_id"]))

                duplicate_group = {
                    "hash": hash_value,
                    "original": records[0],
                    "duplicates": records[1:],
                    "count": len(records),
                }

                duplicates.append(duplicate_group)

            return duplicates

        except Exception as e:
            logger.error(f"Error finding duplicates in {collection_name}: {e}")
            return []

    def remove_duplicates(
        self,
        collection_name: str,
        duplicate_groups: list[dict[str, Any]],
        keep_strategy: str = "first",
    ) -> int:
        """Remove duplicate records from a collection.

        Args:
            collection_name: Collection name
            duplicate_groups: List of duplicate groups
            keep_strategy: Strategy for which record to keep ("first", "last", "newest", "oldest")

        Returns:
            Number of records removed
        """
        try:
            db = get_database()
            collection = db[collection_name]

            removed_count = 0

            for group in duplicate_groups:
                records = [group["original"]] + group["duplicates"]

                if keep_strategy == "first":
                    # Keep first record (already sorted)
                    to_remove = records[1:]
                elif keep_strategy == "last":
                    # Keep last record
                    to_remove = records[:-1]
                elif keep_strategy == "newest":
                    # Sort by some timestamp field (assuming it exists)
                    records.sort(key=lambda r: r["data"].get("transformed_at", ""), reverse=True)
                    to_remove = records[1:]
                elif keep_strategy == "oldest":
                    # Sort by some timestamp field
                    records.sort(key=lambda r: r["data"].get("transformed_at", ""))
                    to_remove = records[1:]
                else:
                    continue

                # Remove duplicate records
                for record in to_remove:
                    try:
                        collection.delete_one({"_id": record["_id"]})
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Error removing duplicate record {record['_id']}: {e}")

            logger.info(f"Removed {removed_count} duplicate records from {collection_name}")
            return removed_count

        except Exception as e:
            logger.error(f"Error removing duplicates from {collection_name}: {e}")
            return 0

    def deduplicate_collection(
        self, collection_name: str, keep_strategy: str = "first", batch_size: int = 1000
    ) -> dict[str, Any]:
        """Complete deduplication process for a collection.

        Args:
            collection_name: Collection name
            keep_strategy: Strategy for which record to keep
            batch_size: Batch size for processing

        Returns:
            Deduplication results
        """
        logger.info(f"Starting deduplication for {collection_name}")

        # Find duplicates
        duplicate_groups = self.find_duplicates(collection_name, batch_size)

        if not duplicate_groups:
            logger.info(f"No duplicates found in {collection_name}")
            return {
                "collection_name": collection_name,
                "duplicate_groups_found": 0,
                "records_removed": 0,
                "status": "no_duplicates",
            }

        # Remove duplicates
        removed_count = self.remove_duplicates(collection_name, duplicate_groups, keep_strategy)

        result = {
            "collection_name": collection_name,
            "duplicate_groups_found": len(duplicate_groups),
            "records_removed": removed_count,
            "status": "completed",
        }

        logger.info(
            f"Deduplication completed for {collection_name}: "
            f"{len(duplicate_groups)} groups, {removed_count} records removed"
        )

        return result

    def clear_cache(self, collection_name: str | None = None) -> None:
        """Clear hash cache.

        Args:
            collection_name: Specific collection to clear (None for all)
        """
        with self._lock:
            if collection_name:
                self._hash_cache.pop(collection_name, None)
            else:
                self._hash_cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {collection: len(hashes) for collection, hashes in self._hash_cache.items()}


class EnhancedUpsertManager:
    """Enhanced upsert manager with deduplication support."""

    def __init__(self, deduplication_engine: DeduplicationEngine):
        """Initialize enhanced upsert manager.

        Args:
            deduplication_engine: Deduplication engine instance
        """
        self.deduplication_engine = deduplication_engine

    def upsert_with_deduplication(
        self,
        collection_name: str,
        record: dict[str, Any],
        filter_query: dict[str, Any],
        deduplication_enabled: bool = True,
    ) -> tuple[bool, str | None, str | None]:
        """Upsert record with deduplication check.

        Args:
            collection_name: Collection name
            record: Record data
            filter_query: MongoDB filter query for upsert
            deduplication_enabled: Whether to perform deduplication

        Returns:
            Tuple of (success, record_id, duplicate_id)
            - success: True if operation succeeded
            - record_id: ID of inserted/updated record (None if duplicate)
            - duplicate_id: ID of existing duplicate record (None if not duplicate)
        """
        try:
            db = get_database()
            collection = db[collection_name]

            # Check for duplicates if enabled
            if deduplication_enabled:
                is_duplicate, duplicate_id = self.deduplication_engine.check_duplicate(
                    collection_name, record
                )
                if is_duplicate:
                    logger.debug(f"Duplicate record detected in {collection_name}")
                    return True, None, duplicate_id

            # Perform upsert
            result = collection.update_one(filter_query, {"$set": record}, upsert=True)

            # Get the record ID
            if result.upserted_id:
                record_id = str(result.upserted_id)
            else:
                # Find existing record
                existing = collection.find_one(filter_query, {"_id": 1})
                record_id = str(existing["_id"]) if existing else None

            # Mark record with deduplication hash
            if record_id and deduplication_enabled:
                self.deduplication_engine.mark_record_hash(collection_name, record, record_id)

            return True, record_id, None

        except Exception as e:
            logger.error(f"Error in enhanced upsert for {collection_name}: {e}")
            record_failure(collection_name, "upsert", type(e).__name__)
            return False, None, None


# Global instances
_deduplication_engine = DeduplicationEngine()
_enhanced_upsert_manager = EnhancedUpsertManager(_deduplication_engine)


def get_deduplication_engine() -> DeduplicationEngine:
    """Get the global deduplication engine instance.

    Returns:
        Deduplication engine instance
    """
    return _deduplication_engine


def get_enhanced_upsert_manager() -> EnhancedUpsertManager:
    """Get the global enhanced upsert manager instance.

    Returns:
        Enhanced upsert manager instance
    """
    return _enhanced_upsert_manager


def check_duplicate(collection_name: str, record: dict[str, Any]) -> tuple[bool, str | None]:
    """Check if a record is a duplicate.

    Args:
        collection_name: Collection name
        record: Record to check

    Returns:
        Tuple of (is_duplicate, duplicate_id)
    """
    return _deduplication_engine.check_duplicate(collection_name, record)


def upsert_with_deduplication(
    collection_name: str,
    record: dict[str, Any],
    filter_query: dict[str, Any],
    deduplication_enabled: bool = True,
) -> tuple[bool, str | None, str | None]:
    """Upsert record with deduplication.

    Args:
        collection_name: Collection name
        record: Record data
        filter_query: MongoDB filter query
        deduplication_enabled: Whether to perform deduplication

    Returns:
        Tuple of (success, record_id, duplicate_id)
    """
    return _enhanced_upsert_manager.upsert_with_deduplication(
        collection_name, record, filter_query, deduplication_enabled
    )


def deduplicate_collection(collection_name: str, keep_strategy: str = "first") -> dict[str, Any]:
    """Deduplicate a collection.

    Args:
        collection_name: Collection name
        keep_strategy: Strategy for which record to keep

    Returns:
        Deduplication results
    """
    return _deduplication_engine.deduplicate_collection(collection_name, keep_strategy)
