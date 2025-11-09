#!/usr/bin/env python3
"""
Dead Letter Queue Manager - Enterprise Edition

Manages failed records with comprehensive error context for manual review,
reprocessing, and troubleshooting in ETL pipelines.

Features:
- Failed record storage with full error context
- Automatic retry scheduling with backoff
- Manual review and reprocessing workflows
- Error pattern analysis and reporting
- Integration with checkpoint system
"""

import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from healthcare_data_pipeline.connection_manager import get_database
from healthcare_data_pipeline.metrics import record_failure
from healthcare_data_pipeline.structured_logging import get_logger

logger = get_logger(__name__)


class DLQStatus(Enum):
    """Dead letter queue entry status."""

    PENDING = "pending"
    RETRY_SCHEDULED = "retry_scheduled"
    RETRIED = "retried"
    REVIEWED = "reviewed"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


class DLQEntry:
    """Dead letter queue entry."""

    def __init__(
        self,
        entry_id: str,
        collection_name: str,
        operation: str,
        record_data: dict[str, Any],
        error_message: str,
        error_type: str,
        error_context: dict[str, Any],
        retry_count: int = 0,
        max_retries: int = 3,
        status: DLQStatus = DLQStatus.PENDING,
        created_at: str | None = None,
        updated_at: str | None = None,
        next_retry_at: str | None = None,
    ):
        """Initialize DLQ entry.

        Args:
            entry_id: Unique entry identifier
            collection_name: Target collection name
            operation: Operation that failed (insert, update, transform, etc.)
            record_data: The record data that failed
            error_message: Error message
            error_type: Error type/class name
            error_context: Additional error context
            retry_count: Number of retries attempted
            max_retries: Maximum retry attempts
            status: Current status
            created_at: Creation timestamp
            updated_at: Last update timestamp
            next_retry_at: Next retry timestamp
        """
        self.entry_id = entry_id
        self.collection_name = collection_name
        self.operation = operation
        self.record_data = record_data
        self.error_message = error_message
        self.error_type = error_type
        self.error_context = error_context
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.status = status
        self.created_at = created_at or datetime.utcnow().isoformat() + "Z"
        self.updated_at = updated_at or self.created_at
        self.next_retry_at = next_retry_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entry_id": self.entry_id,
            "collection_name": self.collection_name,
            "operation": self.operation,
            "record_data": self.record_data,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "error_context": self.error_context,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "next_retry_at": self.next_retry_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DLQEntry":
        """Create from dictionary."""
        data["status"] = DLQStatus(data["status"])
        return cls(**data)

    def can_retry(self) -> bool:
        """Check if entry can be retried."""
        return self.retry_count < self.max_retries and self.status in [
            DLQStatus.PENDING,
            DLQStatus.RETRY_SCHEDULED,
        ]

    def schedule_retry(self, delay_seconds: int = 300) -> None:
        """Schedule next retry attempt."""
        if not self.can_retry():
            return

        self.retry_count += 1
        self.next_retry_at = (
            datetime.utcnow() + timedelta(seconds=delay_seconds)
        ).isoformat() + "Z"
        self.status = DLQStatus.RETRY_SCHEDULED
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def mark_retried(self) -> None:
        """Mark as retried."""
        self.status = DLQStatus.RETRIED
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def mark_resolved(self) -> None:
        """Mark as resolved."""
        self.status = DLQStatus.RESOLVED
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def mark_reviewed(self) -> None:
        """Mark as reviewed."""
        self.status = DLQStatus.REVIEWED
        self.updated_at = datetime.utcnow().isoformat() + "Z"


class DLQManager:
    """Dead letter queue manager for failed records."""

    def __init__(self, dlq_collection: str = "pipeline_dlq"):
        """Initialize DLQ manager.

        Args:
            dlq_collection: MongoDB collection name for DLQ
        """
        self.dlq_collection = dlq_collection
        self._lock = threading.RLock()
        self._retry_scheduler_active = False
        self._scheduler_thread: threading.Thread | None = None

    def add_failed_record(
        self,
        collection_name: str,
        operation: str,
        record_data: dict[str, Any],
        error: Exception,
        error_context: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> str:
        """Add a failed record to the DLQ.

        Args:
            collection_name: Target collection name
            operation: Operation that failed
            record_data: The record data that failed
            error: Exception that occurred
            error_context: Additional error context
            max_retries: Maximum retry attempts

        Returns:
            DLQ entry ID
        """
        with self._lock:
            entry_id = f"dlq_{collection_name}_{operation}_{int(time.time() * 1000000)}"

            error_context = error_context or {}
            error_context.update(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "operation": operation,
                    "collection": collection_name,
                }
            )

            entry = DLQEntry(
                entry_id=entry_id,
                collection_name=collection_name,
                operation=operation,
                record_data=record_data,
                error_message=str(error),
                error_type=type(error).__name__,
                error_context=error_context,
                max_retries=max_retries,
            )

            # Save to database
            self._save_entry(entry)

            # Record metrics
            record_failure(collection_name, operation, type(error).__name__)

            logger.warning(
                f"Added failed record to DLQ: {entry_id}",
                entry_id=entry_id,
                collection_name=collection_name,
                operation=operation,
                error_type=type(error).__name__,
                error_message=str(error)[:200],
            )

            return entry_id

    def get_entry(self, entry_id: str) -> DLQEntry | None:
        """Get DLQ entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            DLQ entry or None if not found
        """
        try:
            db = get_database()
            collection = db[self.dlq_collection]

            doc = collection.find_one({"entry_id": entry_id})
            if doc:
                return DLQEntry.from_dict(doc)

        except Exception as e:
            logger.error(f"Failed to get DLQ entry {entry_id}: {e}")

        return None

    def list_entries(
        self,
        collection_name: str | None = None,
        operation: str | None = None,
        status: DLQStatus | None = None,
        limit: int = 100,
    ) -> list[DLQEntry]:
        """List DLQ entries with optional filtering.

        Args:
            collection_name: Filter by collection name
            operation: Filter by operation
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of DLQ entries
        """
        try:
            db = get_database()
            collection = db[self.dlq_collection]

            # Build query
            query = {}
            if collection_name:
                query["collection_name"] = collection_name
            if operation:
                query["operation"] = operation
            if status:
                query["status"] = status.value

            # Get results
            cursor = collection.find(query).sort("created_at", -1).limit(limit)
            entries = []

            for doc in cursor:
                try:
                    entry = DLQEntry.from_dict(doc)
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to load DLQ entry {doc.get('entry_id')}: {e}")

            return entries

        except Exception as e:
            logger.error(f"Failed to list DLQ entries: {e}")
            return []

    def get_retry_candidates(self) -> list[DLQEntry]:
        """Get entries that are ready for retry.

        Returns:
            List of entries ready for retry
        """
        try:
            db = get_database()
            collection = db[self.dlq_collection]

            # Find entries that can be retried and are due
            now = datetime.utcnow().isoformat() + "Z"

            query = {
                "status": {"$in": [DLQStatus.PENDING.value, DLQStatus.RETRY_SCHEDULED.value]},
                "retry_count": {"$lt": "$max_retries"},  # This is a field comparison
                "$or": [{"next_retry_at": {"$exists": False}}, {"next_retry_at": {"$lte": now}}],
            }

            cursor = collection.find(query).sort("created_at", 1)
            entries = []

            for doc in cursor:
                try:
                    entry = DLQEntry.from_dict(doc)
                    if entry.can_retry():
                        entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to load retry candidate {doc.get('entry_id')}: {e}")

            return entries

        except Exception as e:
            logger.error(f"Failed to get retry candidates: {e}")
            return []

    def retry_entry(self, entry_id: str, processor_func: callable) -> bool:
        """Retry processing a DLQ entry.

        Args:
            entry_id: Entry ID to retry
            processor_func: Function to process the record (should accept record_data)

        Returns:
            True if retry successful, False otherwise
        """
        with self._lock:
            entry = self.get_entry(entry_id)
            if not entry or not entry.can_retry():
                return False

            try:
                # Attempt to process the record
                processor_func(entry.record_data)

                # Mark as retried successfully
                entry.mark_retried()
                self._save_entry(entry)

                logger.info(
                    f"Successfully retried DLQ entry: {entry_id}",
                    entry_id=entry_id,
                    collection_name=entry.collection_name,
                    operation=entry.operation,
                    retry_count=entry.retry_count,
                )

                return True

            except Exception as e:
                # Retry failed, schedule next attempt if possible
                if entry.can_retry():
                    entry.schedule_retry()
                    logger.warning(
                        f"Retry failed for DLQ entry {entry_id}, scheduled next attempt: {e}",
                        entry_id=entry_id,
                        retry_count=entry.retry_count,
                        next_retry_at=entry.next_retry_at,
                    )
                else:
                    logger.error(
                        f"Max retries exceeded for DLQ entry {entry_id}: {e}",
                        entry_id=entry_id,
                        max_retries=entry.max_retries,
                    )

                self._save_entry(entry)
                return False

    def resolve_entry(self, entry_id: str, resolution_notes: str | None = None) -> bool:
        """Mark DLQ entry as resolved.

        Args:
            entry_id: Entry ID
            resolution_notes: Optional resolution notes

        Returns:
            True if successfully resolved, False otherwise
        """
        with self._lock:
            entry = self.get_entry(entry_id)
            if not entry:
                return False

            entry.mark_resolved()
            if resolution_notes:
                entry.error_context["resolution_notes"] = resolution_notes

            self._save_entry(entry)

            logger.info(
                f"Resolved DLQ entry: {entry_id}",
                entry_id=entry_id,
                collection_name=entry.collection_name,
                resolution_notes=resolution_notes,
            )

            return True

    def archive_entry(self, entry_id: str) -> bool:
        """Archive DLQ entry.

        Args:
            entry_id: Entry ID

        Returns:
            True if successfully archived, False otherwise
        """
        with self._lock:
            entry = self.get_entry(entry_id)
            if not entry:
                return False

            entry.status = DLQStatus.ARCHIVED
            entry.updated_at = datetime.utcnow().isoformat() + "Z"

            self._save_entry(entry)

            logger.info(
                f"Archived DLQ entry: {entry_id}",
                entry_id=entry_id,
                collection_name=entry.collection_name,
            )

            return True

    def cleanup_old_entries(self, days_to_keep: int = 90) -> int:
        """Clean up old resolved/archived entries.

        Args:
            days_to_keep: Number of days to keep old entries

        Returns:
            Number of entries cleaned up
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat() + "Z"

            db = get_database()
            collection = db[self.dlq_collection]

            # Only delete resolved or archived entries older than cutoff
            result = collection.delete_many(
                {
                    "status": {"$in": [DLQStatus.RESOLVED.value, DLQStatus.ARCHIVED.value]},
                    "updated_at": {"$lt": cutoff_date},
                }
            )

            cleaned_count = result.deleted_count
            logger.info(f"Cleaned up {cleaned_count} old DLQ entries")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old DLQ entries: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get DLQ statistics.

        Returns:
            Dictionary with DLQ statistics
        """
        try:
            db = get_database()
            collection = db[self.dlq_collection]

            pipeline = [
                {
                    "$group": {
                        "_id": {
                            "collection": "$collection_name",
                            "operation": "$operation",
                            "status": "$status",
                        },
                        "count": {"$sum": 1},
                        "total_retries": {"$sum": "$retry_count"},
                    }
                }
            ]

            results = list(collection.aggregate(pipeline))

            stats = {
                "total_entries": 0,
                "by_collection": {},
                "by_operation": {},
                "by_status": {},
                "total_retries": 0,
            }

            for result in results:
                count = result["count"]
                retries = result["total_retries"]
                collection_name = result["_id"]["collection"]
                operation = result["_id"]["operation"]
                status = result["_id"]["status"]

                stats["total_entries"] += count
                stats["total_retries"] += retries

                # By collection
                if collection_name not in stats["by_collection"]:
                    stats["by_collection"][collection_name] = {}
                stats["by_collection"][collection_name][status] = count

                # By operation
                if operation not in stats["by_operation"]:
                    stats["by_operation"][operation] = {}
                stats["by_operation"][operation][status] = count

                # By status
                if status not in stats["by_status"]:
                    stats["by_status"][status] = 0
                stats["by_status"][status] += count

            return stats

        except Exception as e:
            logger.error(f"Failed to get DLQ stats: {e}")
            return {}

    def _save_entry(self, entry: DLQEntry) -> None:
        """Save DLQ entry to database.

        Args:
            entry: DLQ entry to save
        """
        try:
            db = get_database()
            collection = db[self.dlq_collection]

            collection.replace_one({"entry_id": entry.entry_id}, entry.to_dict(), upsert=True)

        except Exception as e:
            logger.error(f"Failed to save DLQ entry {entry.entry_id}: {e}")

    def start_retry_scheduler(self, check_interval: int = 60) -> None:
        """Start the retry scheduler thread.

        Args:
            check_interval: Interval in seconds between retry checks
        """
        if self._retry_scheduler_active:
            return

        self._retry_scheduler_active = True
        self._scheduler_thread = threading.Thread(
            target=self._retry_scheduler_loop,
            args=(check_interval,),
            daemon=True,
            name="DLQRetryScheduler",
        )
        self._scheduler_thread.start()

        logger.info("DLQ retry scheduler started")

    def stop_retry_scheduler(self) -> None:
        """Stop the retry scheduler thread."""
        self._retry_scheduler_active = False
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5)

        logger.info("DLQ retry scheduler stopped")

    def _retry_scheduler_loop(self, check_interval: int) -> None:
        """Retry scheduler loop.

        Args:
            check_interval: Check interval in seconds
        """
        while self._retry_scheduler_active:
            try:
                # Get retry candidates
                candidates = self.get_retry_candidates()

                if candidates:
                    logger.info(f"Found {len(candidates)} DLQ entries ready for retry")

                    for entry in candidates:
                        # For now, just mark as needing manual review
                        # In a real implementation, this would call a retry processor
                        logger.info(
                            f"DLQ entry {entry.entry_id} ready for retry "
                            f"(attempt {entry.retry_count + 1}/{entry.max_retries})"
                        )

                        # Schedule next retry with exponential backoff
                        delay = 300 * (2**entry.retry_count)  # 5min, 10min, 20min...
                        entry.schedule_retry(delay)
                        self._save_entry(entry)

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in DLQ retry scheduler: {e}")
                time.sleep(check_interval)


# Global instance
_dlq_manager = DLQManager()


def get_dlq_manager() -> DLQManager:
    """Get the global DLQ manager instance.

    Returns:
        DLQ manager instance
    """
    return _dlq_manager


def add_failed_record(
    collection_name: str,
    operation: str,
    record_data: dict[str, Any],
    error: Exception,
    error_context: dict[str, Any] | None = None,
) -> str:
    """Add a failed record to the DLQ.

    Args:
        collection_name: Target collection name
        operation: Operation that failed
        record_data: The record data that failed
        error: Exception that occurred
        error_context: Additional error context

    Returns:
        DLQ entry ID
    """
    return _dlq_manager.add_failed_record(
        collection_name, operation, record_data, error, error_context
    )


def get_retry_candidates() -> list[DLQEntry]:
    """Get entries ready for retry.

    Returns:
        List of entries ready for retry
    """
    return _dlq_manager.get_retry_candidates()


def resolve_dlq_entry(entry_id: str, resolution_notes: str | None = None) -> bool:
    """Resolve a DLQ entry.

    Args:
        entry_id: Entry ID
        resolution_notes: Optional resolution notes

    Returns:
        True if resolved, False otherwise
    """
    return _dlq_manager.resolve_entry(entry_id, resolution_notes)


def archive_dlq_entry(entry_id: str) -> bool:
    """Archive a DLQ entry.

    Args:
        entry_id: Entry ID

    Returns:
        True if archived, False otherwise
    """
    return _dlq_manager.archive_entry(entry_id)
