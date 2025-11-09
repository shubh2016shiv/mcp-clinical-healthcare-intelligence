#!/usr/bin/env python3
"""
Metrics Collection - Enterprise Edition

Comprehensive metrics collection for ETL pipeline monitoring and observability.

Features:
- Throughput, latency, and error rate tracking
- Data quality score metrics
- Prometheus-compatible metrics export
- Real-time dashboard support
- Thread-safe operations
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import psutil


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricLabels:
    """Standard metric labels."""

    def __init__(
        self,
        pipeline_stage: str | None = None,
        collection_name: str | None = None,
        operation_type: str | None = None,
        status: str | None = None,
        **extra_labels,
    ):
        """Initialize metric labels.

        Args:
            pipeline_stage: Pipeline stage (ingestion, transformation, etc.)
            collection_name: MongoDB collection name
            operation_type: Operation type (insert, update, etc.)
            status: Operation status (success, error, etc.)
            **extra_labels: Additional custom labels
        """
        self.pipeline_stage = pipeline_stage
        self.collection_name = collection_name
        self.operation_type = operation_type
        self.status = status
        self.extra_labels = extra_labels

    def to_dict(self) -> dict[str, str]:
        """Convert labels to dictionary."""
        labels = {}
        if self.pipeline_stage:
            labels["pipeline_stage"] = self.pipeline_stage
        if self.collection_name:
            labels["collection"] = self.collection_name
        if self.operation_type:
            labels["operation"] = self.operation_type
        if self.status:
            labels["status"] = self.status
        labels.update(self.extra_labels)
        return labels

    def get_label_string(self) -> str:
        """Get labels as Prometheus label string."""
        labels = self.to_dict()
        if not labels:
            return ""

        label_parts = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_parts) + "}"


@dataclass
class CounterMetric:
    """Counter metric (monotonically increasing value)."""

    name: str
    description: str
    value: int = 0
    labels: MetricLabels = field(default_factory=MetricLabels)

    def increment(self, amount: int = 1) -> None:
        """Increment counter by amount."""
        self.value += amount

    def reset(self) -> None:
        """Reset counter to zero."""
        self.value = 0


@dataclass
class GaugeMetric:
    """Gauge metric (can increase or decrease)."""

    name: str
    description: str
    value: float = 0.0
    labels: MetricLabels = field(default_factory=MetricLabels)

    def set(self, value: float) -> None:
        """Set gauge to specific value."""
        self.value = value

    def increment(self, amount: float = 1.0) -> None:
        """Increment gauge by amount."""
        self.value += amount

    def decrement(self, amount: float = 1.0) -> None:
        """Decrement gauge by amount."""
        self.value -= amount


@dataclass
class HistogramMetric:
    """Histogram metric for measuring distributions."""

    name: str
    description: str
    buckets: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
    observations: list[float] = field(default_factory=list)
    labels: MetricLabels = field(default_factory=MetricLabels)

    def observe(self, value: float) -> None:
        """Add an observation."""
        self.observations.append(value)

    def get_count(self) -> int:
        """Get total count of observations."""
        return len(self.observations)

    def get_sum(self) -> float:
        """Get sum of all observations."""
        return sum(self.observations) if self.observations else 0.0

    def get_bucket_counts(self) -> dict[float, int]:
        """Get counts for each bucket."""
        bucket_counts = defaultdict(int)
        for obs in self.observations:
            for bucket in self.buckets:
                if obs <= bucket:
                    bucket_counts[bucket] += 1
            # Also count in +Inf bucket
            bucket_counts[float("inf")] += 1
        return dict(bucket_counts)

    def reset(self) -> None:
        """Reset histogram observations."""
        self.observations.clear()


class SystemMetrics:
    """System resource metrics collector."""

    def __init__(self, collection_interval: float = 5.0):
        """Initialize system metrics collector.

        Args:
            collection_interval: Interval between collections (seconds)
        """
        self.collection_interval = collection_interval
        self.last_collection = 0
        self._lock = threading.RLock()

        # Initialize metrics
        self.cpu_percent = GaugeMetric("system_cpu_percent", "CPU usage percentage")
        self.memory_percent = GaugeMetric("system_memory_percent", "Memory usage percentage")
        self.memory_used_mb = GaugeMetric("system_memory_used_mb", "Memory used in MB")
        self.disk_usage_percent = GaugeMetric("system_disk_usage_percent", "Disk usage percentage")

    def collect(self) -> None:
        """Collect current system metrics."""
        with self._lock:
            current_time = time.time()
            if current_time - self.last_collection < self.collection_interval:
                return

            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_percent.set(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_percent.set(memory.percent)
                self.memory_used_mb.set(memory.used / 1024 / 1024)

                # Disk usage (root filesystem)
                disk = psutil.disk_usage("/")
                self.disk_usage_percent.set(disk.percent)

                self.last_collection = current_time

            except Exception:
                # Silently fail system metrics collection
                pass

    def get_metrics(self) -> list[Any]:
        """Get all system metrics."""
        self.collect()
        return [self.cpu_percent, self.memory_percent, self.memory_used_mb, self.disk_usage_percent]


class PipelineMetrics:
    """Pipeline-specific metrics collector."""

    def __init__(self):
        """Initialize pipeline metrics."""
        self._lock = threading.RLock()

        # Throughput metrics
        self.records_processed = CounterMetric(
            "pipeline_records_processed", "Total records processed"
        )
        self.records_transformed = CounterMetric(
            "pipeline_records_transformed", "Total records transformed"
        )
        self.records_failed = CounterMetric(
            "pipeline_records_failed", "Total records that failed processing"
        )

        # Performance metrics
        self.ingestion_duration = HistogramMetric(
            "pipeline_ingestion_duration_seconds", "Ingestion operation duration"
        )
        self.transformation_duration = HistogramMetric(
            "pipeline_transformation_duration_seconds", "Transformation operation duration"
        )
        self.batch_processing_duration = HistogramMetric(
            "pipeline_batch_duration_seconds", "Batch processing duration"
        )

        # Data quality metrics
        self.data_quality_score = GaugeMetric(
            "pipeline_data_quality_score", "Overall data quality score (0-100)"
        )
        self.schema_validation_errors = CounterMetric(
            "pipeline_schema_validation_errors", "Schema validation errors"
        )
        self.duplicate_records = CounterMetric(
            "pipeline_duplicate_records", "Duplicate records detected"
        )

        # Collection-specific metrics
        self.collection_metrics: dict[str, dict[str, Any]] = defaultdict(dict)

    def record_record_processed(self, collection_name: str, count: int = 1) -> None:
        """Record records processed."""
        with self._lock:
            self.records_processed.increment(count)
            if collection_name not in self.collection_metrics:
                self.collection_metrics[collection_name] = {}
            if "processed" not in self.collection_metrics[collection_name]:
                self.collection_metrics[collection_name]["processed"] = CounterMetric(
                    "pipeline_collection_records_processed",
                    "Records processed per collection",
                    labels=MetricLabels(collection_name=collection_name),
                )
            self.collection_metrics[collection_name]["processed"].increment(count)

    def record_transformation_success(
        self, collection_name: str, duration: float, count: int = 1
    ) -> None:
        """Record successful transformation."""
        with self._lock:
            self.records_transformed.increment(count)
            self.transformation_duration.observe(duration)

            if collection_name not in self.collection_metrics:
                self.collection_metrics[collection_name] = {}
            if "transformed" not in self.collection_metrics[collection_name]:
                self.collection_metrics[collection_name]["transformed"] = CounterMetric(
                    "pipeline_collection_records_transformed",
                    "Records transformed per collection",
                    labels=MetricLabels(collection_name=collection_name),
                )
            self.collection_metrics[collection_name]["transformed"].increment(count)

    def record_ingestion_success(
        self, collection_name: str, duration: float, count: int = 1
    ) -> None:
        """Record successful ingestion."""
        with self._lock:
            self.records_processed.increment(count)
            self.ingestion_duration.observe(duration)

            if collection_name not in self.collection_metrics:
                self.collection_metrics[collection_name] = {}
            if "ingested" not in self.collection_metrics[collection_name]:
                self.collection_metrics[collection_name]["ingested"] = CounterMetric(
                    "pipeline_collection_records_ingested",
                    "Records ingested per collection",
                    labels=MetricLabels(collection_name=collection_name),
                )
            self.collection_metrics[collection_name]["ingested"].increment(count)

    def record_failure(
        self, collection_name: str, operation: str, error_type: str, count: int = 1
    ) -> None:
        """Record processing failure."""
        with self._lock:
            self.records_failed.increment(count)

            if collection_name not in self.collection_metrics:
                self.collection_metrics[collection_name] = {}
            failure_key = f"failures_{operation}"
            if failure_key not in self.collection_metrics[collection_name]:
                self.collection_metrics[collection_name][failure_key] = CounterMetric(
                    f"pipeline_collection_{operation}_failures",
                    f"{operation.title()} failures per collection",
                    labels=MetricLabels(collection_name=collection_name, status="error"),
                )
            self.collection_metrics[collection_name][failure_key].increment(count)

    def update_data_quality_score(self, score: float) -> None:
        """Update overall data quality score."""
        with self._lock:
            self.data_quality_score.set(max(0.0, min(100.0, score)))

    def record_batch_duration(self, duration: float) -> None:
        """Record batch processing duration."""
        with self._lock:
            self.batch_processing_duration.observe(duration)

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            total_processed = self.records_processed.value
            total_transformed = self.records_transformed.value
            total_failed = self.records_failed.value

            success_rate = 0.0
            if total_processed > 0:
                success_rate = (
                    (total_transformed + total_processed - total_failed) / total_processed
                ) * 100

            # Calculate average durations
            ingestion_avg = self.ingestion_duration.get_sum() / max(
                1, self.ingestion_duration.get_count()
            )
            transformation_avg = self.transformation_duration.get_sum() / max(
                1, self.transformation_duration.get_count()
            )
            batch_avg = self.batch_processing_duration.get_sum() / max(
                1, self.batch_processing_duration.get_count()
            )

            return {
                "total_records_processed": total_processed,
                "total_records_transformed": total_transformed,
                "total_records_failed": total_failed,
                "success_rate_percent": success_rate,
                "avg_ingestion_duration_sec": ingestion_avg,
                "avg_transformation_duration_sec": transformation_avg,
                "avg_batch_duration_sec": batch_avg,
                "data_quality_score": self.data_quality_score.value,
                "collections_processed": len(self.collection_metrics),
            }

    def get_all_metrics(self) -> list[Any]:
        """Get all metrics for export."""
        with self._lock:
            metrics = [
                self.records_processed,
                self.records_transformed,
                self.records_failed,
                self.ingestion_duration,
                self.transformation_duration,
                self.batch_processing_duration,
                self.data_quality_score,
                self.schema_validation_errors,
                self.duplicate_records,
            ]

            # Add collection-specific metrics
            for collection_data in self.collection_metrics.values():
                metrics.extend(collection_data.values())

            return metrics


class MetricsCollector:
    """Main metrics collector with system and pipeline metrics."""

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize metrics collector."""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.system_metrics = SystemMetrics()
        self.pipeline_metrics = PipelineMetrics()
        self._start_time = time.time()

    def get_uptime_seconds(self) -> float:
        """Get collector uptime in seconds."""
        return time.time() - self._start_time

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary."""
        summary = {
            "uptime_seconds": self.get_uptime_seconds(),
            "pipeline": self.pipeline_metrics.get_summary_stats(),
        }

        # Add system metrics
        system_stats = {}
        for metric in self.system_metrics.get_metrics():
            system_stats[metric.name] = metric.value
        summary["system"] = system_stats

        return summary

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Add pipeline metrics
        for metric in self.pipeline_metrics.get_all_metrics():
            if isinstance(metric, CounterMetric):
                label_str = metric.labels.get_label_string()
                lines.append(f"# HELP {metric.name} {metric.description}")
                lines.append(f"# TYPE {metric.name} counter")
                lines.append(f"{metric.name}{label_str} {metric.value}")
            elif isinstance(metric, GaugeMetric):
                label_str = metric.labels.get_label_string()
                lines.append(f"# HELP {metric.name} {metric.description}")
                lines.append(f"# TYPE {metric.name} gauge")
                lines.append(f"{metric.name}{label_str} {metric.value}")
            elif isinstance(metric, HistogramMetric):
                label_str = metric.labels.get_label_string()
                lines.append(f"# HELP {metric.name} {metric.description}")
                lines.append(f"# TYPE {metric.name} histogram")
                lines.append(f"{metric.name}_count{label_str} {metric.get_count()}")
                lines.append(f"{metric.name}_sum{label_str} {metric.get_sum()}")
                for bucket, count in metric.get_bucket_counts().items():
                    bucket_label = (
                        f'{label_str[:-1]},le="{bucket}"}}' if label_str else f'{{le="{bucket}"}}'
                    )
                    lines.append(f"{metric.name}_bucket{bucket_label} {count}")

        # Add system metrics
        for metric in self.system_metrics.get_metrics():
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} gauge")
            lines.append(f"{metric.name} {metric.value}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all metrics."""
        self.pipeline_metrics = PipelineMetrics()
        self._start_time = time.time()


# Global instance and convenience functions
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Metrics collector instance
    """
    return _metrics_collector


def record_ingestion_success(collection_name: str, duration: float, count: int = 1) -> None:
    """Record successful ingestion operation."""
    _metrics_collector.pipeline_metrics.record_ingestion_success(collection_name, duration, count)


def record_transformation_success(collection_name: str, duration: float, count: int = 1) -> None:
    """Record successful transformation operation."""
    _metrics_collector.pipeline_metrics.record_transformation_success(
        collection_name, duration, count
    )


def record_failure(collection_name: str, operation: str, error_type: str, count: int = 1) -> None:
    """Record processing failure."""
    _metrics_collector.pipeline_metrics.record_failure(
        collection_name, operation, error_type, count
    )


def record_batch_duration(duration: float) -> None:
    """Record batch processing duration."""
    _metrics_collector.pipeline_metrics.record_batch_duration(duration)


def update_data_quality_score(score: float) -> None:
    """Update data quality score."""
    _metrics_collector.pipeline_metrics.update_data_quality_score(score)


def get_metrics_summary() -> dict[str, Any]:
    """Get metrics summary."""
    return _metrics_collector.get_summary()


def export_prometheus_metrics() -> str:
    """Export metrics in Prometheus format."""
    return _metrics_collector.export_prometheus()


def print_metrics_dashboard() -> None:
    """Print a real-time metrics dashboard."""
    summary = get_metrics_summary()

    print("\n" + "=" * 80)
    print("PIPELINE METRICS DASHBOARD")
    print("=" * 80)
    print(".2f")
    print()

    pipeline = summary["pipeline"]
    print("PIPELINE STATISTICS:")
    print(f"  Records Processed: {pipeline['total_records_processed']:,}")
    print(f"  Records Transformed: {pipeline['total_records_transformed']:,}")
    print(f"  Records Failed: {pipeline['total_records_failed']:,}")
    print(".1f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".1f")
    print()

    print("SYSTEM RESOURCES:")
    print(".1f")
    print(".1f")
    print(".0f")
    print(".1f")
    print("=" * 80 + "\n")
