#!/usr/bin/env python3
"""
Checkpoint Manager - Enterprise Edition

Advanced checkpointing system for ETL pipeline resume capability and fault tolerance.

Features:
- File-level and document-level checkpoint tracking
- Automatic resume from last successful checkpoint
- Metadata storage with processing statistics
- Checkpoint validation and cleanup
- Thread-safe operations
"""

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from healthcare_data_pipeline.connection_manager import get_database
from healthcare_data_pipeline.structured_logging import get_logger

logger = get_logger(__name__)


class CheckpointStatus(Enum):
    """Checkpoint status types."""

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointType(Enum):
    """Checkpoint types."""

    FILE_LEVEL = "file_level"
    BATCH_LEVEL = "batch_level"
    COLLECTION_LEVEL = "collection_level"


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    checkpoint_id: str
    checkpoint_type: CheckpointType
    pipeline_stage: str
    collection_name: str | None = None
    file_path: str | None = None
    batch_id: str | None = None
    status: CheckpointStatus = CheckpointStatus.STARTED
    start_time: str = ""
    end_time: str | None = None
    records_processed: int = 0
    records_failed: int = 0
    records_skipped: int = 0
    error_message: str | None = None
    retry_count: int = 0
    checksum: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Initialize timestamps and metadata."""
        if not self.start_time:
            self.start_time = datetime.utcnow().isoformat() + "Z"
        if self.metadata is None:
            self.metadata = {}

    def mark_completed(
        self, records_processed: int = 0, records_failed: int = 0, records_skipped: int = 0
    ):
        """Mark checkpoint as completed."""
        self.status = CheckpointStatus.COMPLETED
        self.end_time = datetime.utcnow().isoformat() + "Z"
        self.records_processed = records_processed
        self.records_failed = records_failed
        self.records_skipped = records_skipped

    def mark_failed(self, error_message: str, records_processed: int = 0, records_failed: int = 0):
        """Mark checkpoint as failed."""
        self.status = CheckpointStatus.FAILED
        self.end_time = datetime.utcnow().isoformat() + "Z"
        self.error_message = error_message
        self.records_processed = records_processed
        self.records_failed = records_failed

    def increment_retry(self):
        """Increment retry count."""
        self.retry_count += 1

    def calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert enums to values
        data["checkpoint_type"] = self.checkpoint_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        # Convert string values back to enums
        data["checkpoint_type"] = CheckpointType(data["checkpoint_type"])
        data["status"] = CheckpointStatus(data["status"])
        return cls(**data)


class CheckpointManager:
    """Manager for pipeline checkpoints with MongoDB persistence."""

    def __init__(self, checkpoint_collection: str = "pipeline_checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_collection: MongoDB collection name for checkpoints
        """
        self.checkpoint_collection = checkpoint_collection
        self._lock = threading.RLock()
        self._active_checkpoints: dict[str, CheckpointMetadata] = {}

    def _get_checkpoint_key(self, checkpoint: CheckpointMetadata) -> str:
        """Generate unique key for checkpoint.

        Args:
            checkpoint: Checkpoint metadata

        Returns:
            Unique checkpoint key
        """
        key_parts = [
            checkpoint.pipeline_stage,
            checkpoint.collection_name or "none",
            checkpoint.file_path or "none",
            checkpoint.batch_id or "none",
        ]
        return "|".join(key_parts)

    def start_checkpoint(
        self,
        checkpoint_type: CheckpointType,
        pipeline_stage: str,
        collection_name: str | None = None,
        file_path: str | None = None,
        batch_id: str | None = None,
        **metadata,
    ) -> str:
        """Start a new checkpoint.

        Args:
            checkpoint_type: Type of checkpoint
            pipeline_stage: Pipeline stage (ingestion, transformation)
            collection_name: MongoDB collection name
            file_path: File path being processed
            batch_id: Batch identifier
            **metadata: Additional metadata

        Returns:
            Checkpoint ID
        """
        with self._lock:
            checkpoint_id = f"{pipeline_stage}_{int(time.time() * 1000000)}"

            checkpoint = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type,
                pipeline_stage=pipeline_stage,
                collection_name=collection_name,
                file_path=file_path,
                batch_id=batch_id,
                metadata=metadata,
            )

            key = self._get_checkpoint_key(checkpoint)
            self._active_checkpoints[key] = checkpoint

            # Persist checkpoint
            self._save_checkpoint(checkpoint)

            logger.info(
                f"Started checkpoint {checkpoint_id}",
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type.value,
                pipeline_stage=pipeline_stage,
                collection_name=collection_name,
            )

            return checkpoint_id

    def update_checkpoint(
        self,
        checkpoint_id: str,
        records_processed: int = 0,
        records_failed: int = 0,
        records_skipped: int = 0,
        **metadata,
    ) -> None:
        """Update checkpoint progress.

        Args:
            checkpoint_id: Checkpoint ID
            records_processed: Number of records processed
            records_failed: Number of records failed
            records_skipped: Number of records skipped
            **metadata: Additional metadata to update
        """
        with self._lock:
            checkpoint = self._find_checkpoint_by_id(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint {checkpoint_id} not found for update")
                return

            checkpoint.records_processed = records_processed
            checkpoint.records_failed = records_failed
            checkpoint.records_skipped = records_skipped

            if metadata:
                checkpoint.metadata.update(metadata)

            # Update status if not already completed/failed
            if checkpoint.status == CheckpointStatus.STARTED:
                checkpoint.status = CheckpointStatus.IN_PROGRESS

            self._save_checkpoint(checkpoint)

    def complete_checkpoint(
        self,
        checkpoint_id: str,
        records_processed: int = 0,
        records_failed: int = 0,
        records_skipped: int = 0,
    ) -> None:
        """Mark checkpoint as completed.

        Args:
            checkpoint_id: Checkpoint ID
            records_processed: Final count of records processed
            records_failed: Final count of records failed
            records_skipped: Final count of records skipped
        """
        with self._lock:
            checkpoint = self._find_checkpoint_by_id(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint {checkpoint_id} not found for completion")
                return

            checkpoint.mark_completed(records_processed, records_failed, records_skipped)

            key = self._get_checkpoint_key(checkpoint)
            if key in self._active_checkpoints:
                del self._active_checkpoints[key]

            self._save_checkpoint(checkpoint)

            logger.info(
                f"Completed checkpoint {checkpoint_id}",
                checkpoint_id=checkpoint_id,
                records_processed=records_processed,
                records_failed=records_failed,
                records_skipped=records_skipped,
            )

    def fail_checkpoint(self, checkpoint_id: str, error_message: str) -> None:
        """Mark checkpoint as failed.

        Args:
            checkpoint_id: Checkpoint ID
            error_message: Error message
        """
        with self._lock:
            checkpoint = self._find_checkpoint_by_id(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint {checkpoint_id} not found for failure")
                return

            checkpoint.mark_failed(error_message)

            key = self._get_checkpoint_key(checkpoint)
            if key in self._active_checkpoints:
                del self._active_checkpoints[key]

            self._save_checkpoint(checkpoint)

            logger.error(
                f"Failed checkpoint {checkpoint_id}: {error_message}",
                checkpoint_id=checkpoint_id,
                error_message=error_message,
            )

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointMetadata | None:
        """Get checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint metadata or None if not found
        """
        with self._lock:
            return self._find_checkpoint_by_id(checkpoint_id)

    def list_checkpoints(
        self,
        pipeline_stage: str | None = None,
        collection_name: str | None = None,
        status: CheckpointStatus | None = None,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints with optional filtering.

        Args:
            pipeline_stage: Filter by pipeline stage
            collection_name: Filter by collection name
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of checkpoint metadata
        """
        try:
            db = get_database()
            collection = db[self.checkpoint_collection]

            # Build query
            query = {}
            if pipeline_stage:
                query["pipeline_stage"] = pipeline_stage
            if collection_name:
                query["collection_name"] = collection_name
            if status:
                query["status"] = status.value

            # Get results
            cursor = collection.find(query).sort("start_time", -1).limit(limit)
            checkpoints = []

            for doc in cursor:
                try:
                    checkpoint = CheckpointMetadata.from_dict(doc)
                    checkpoints.append(checkpoint)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {doc.get('_id')}: {e}")

            return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def get_resume_points(
        self, pipeline_stage: str, collection_name: str | None = None
    ) -> list[CheckpointMetadata]:
        """Get checkpoints that can be resumed from.

        Args:
            pipeline_stage: Pipeline stage
            collection_name: Collection name (optional)

        Returns:
            List of checkpoints that can be resumed
        """
        checkpoints = self.list_checkpoints(
            pipeline_stage=pipeline_stage,
            collection_name=collection_name,
            status=CheckpointStatus.IN_PROGRESS,
            limit=50,
        )

        # Also include failed checkpoints that can be retried
        failed_checkpoints = self.list_checkpoints(
            pipeline_stage=pipeline_stage,
            collection_name=collection_name,
            status=CheckpointStatus.FAILED,
            limit=50,
        )

        # Filter failed checkpoints with low retry counts
        retryable_failed = [
            cp
            for cp in failed_checkpoints
            if cp.retry_count < 3  # Max 3 retries
        ]

        return checkpoints + retryable_failed

    def cleanup_old_checkpoints(self, days_to_keep: int = 30) -> int:
        """Clean up old completed checkpoints.

        Args:
            days_to_keep: Number of days to keep completed checkpoints

        Returns:
            Number of checkpoints cleaned up
        """
        try:
            from datetime import timedelta

            cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat() + "Z"

            db = get_database()
            collection = db[self.checkpoint_collection]

            # Only delete completed checkpoints older than cutoff
            result = collection.delete_many(
                {"status": CheckpointStatus.COMPLETED.value, "end_time": {"$lt": cutoff_date}}
            )

            cleaned_count = result.deleted_count
            logger.info(f"Cleaned up {cleaned_count} old checkpoints")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0

    def _find_checkpoint_by_id(self, checkpoint_id: str) -> CheckpointMetadata | None:
        """Find checkpoint by ID in active checkpoints or database.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint metadata or None
        """
        # Check active checkpoints first
        for checkpoint in self._active_checkpoints.values():
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint

        # Check database
        try:
            db = get_database()
            collection = db[self.checkpoint_collection]

            doc = collection.find_one({"checkpoint_id": checkpoint_id})
            if doc:
                return CheckpointMetadata.from_dict(doc)

        except Exception as e:
            logger.warning(f"Failed to find checkpoint {checkpoint_id} in database: {e}")

        return None

    def _save_checkpoint(self, checkpoint: CheckpointMetadata) -> None:
        """Save checkpoint to database.

        Args:
            checkpoint: Checkpoint metadata to save
        """
        try:
            db = get_database()
            collection = db[self.checkpoint_collection]

            # Update the checkpoint in database
            collection.replace_one(
                {"checkpoint_id": checkpoint.checkpoint_id}, checkpoint.to_dict(), upsert=True
            )

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")

    def get_checkpoint_stats(self) -> dict[str, Any]:
        """Get checkpoint statistics.

        Returns:
            Dictionary with checkpoint statistics
        """
        try:
            db = get_database()
            collection = db[self.checkpoint_collection]

            pipeline = [
                {
                    "$group": {
                        "_id": {"stage": "$pipeline_stage", "status": "$status"},
                        "count": {"$sum": 1},
                        "total_records": {"$sum": "$records_processed"},
                        "total_failed": {"$sum": "$records_failed"},
                    }
                }
            ]

            results = list(collection.aggregate(pipeline))

            stats = {
                "total_checkpoints": 0,
                "by_stage": {},
                "by_status": {},
                "total_records_processed": 0,
                "total_records_failed": 0,
            }

            for result in results:
                stage = result["_id"]["stage"]
                status = result["_id"]["status"]
                count = result["count"]
                records = result["total_records"]
                failed = result["total_failed"]

                stats["total_checkpoints"] += count
                stats["total_records_processed"] += records
                stats["total_records_failed"] += failed

                if stage not in stats["by_stage"]:
                    stats["by_stage"][stage] = {}
                stats["by_stage"][stage][status] = {
                    "count": count,
                    "records_processed": records,
                    "records_failed": failed,
                }

                if status not in stats["by_status"]:
                    stats["by_status"][status] = 0
                stats["by_status"][status] += count

            return stats

        except Exception as e:
            logger.error(f"Failed to get checkpoint stats: {e}")
            return {}


# Global instance
_checkpoint_manager = CheckpointManager()


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance.

    Returns:
        Checkpoint manager instance
    """
    return _checkpoint_manager


def start_checkpoint(
    checkpoint_type: CheckpointType,
    pipeline_stage: str,
    collection_name: str | None = None,
    file_path: str | None = None,
    batch_id: str | None = None,
    **metadata,
) -> str:
    """Start a new checkpoint.

    Args:
        checkpoint_type: Type of checkpoint
        pipeline_stage: Pipeline stage
        collection_name: Collection name
        file_path: File path
        batch_id: Batch ID
        **metadata: Additional metadata

    Returns:
        Checkpoint ID
    """
    return _checkpoint_manager.start_checkpoint(
        checkpoint_type, pipeline_stage, collection_name, file_path, batch_id, **metadata
    )


def complete_checkpoint(
    checkpoint_id: str,
    records_processed: int = 0,
    records_failed: int = 0,
    records_skipped: int = 0,
) -> None:
    """Complete a checkpoint."""
    _checkpoint_manager.complete_checkpoint(
        checkpoint_id, records_processed, records_failed, records_skipped
    )


def fail_checkpoint(checkpoint_id: str, error_message: str) -> None:
    """Fail a checkpoint."""
    _checkpoint_manager.fail_checkpoint(checkpoint_id, error_message)


def get_resume_points(
    pipeline_stage: str, collection_name: str | None = None
) -> list[CheckpointMetadata]:
    """Get checkpoints that can be resumed from."""
    return _checkpoint_manager.get_resume_points(pipeline_stage, collection_name)


def cleanup_checkpoints(days_to_keep: int = 30) -> int:
    """Clean up old checkpoints."""
    return _checkpoint_manager.cleanup_old_checkpoints(days_to_keep)
