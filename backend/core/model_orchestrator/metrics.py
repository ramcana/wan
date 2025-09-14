"""
Prometheus metrics for Model Orchestrator observability.

This module provides comprehensive metrics collection for downloads, errors,
storage usage, and performance tracking with limited cardinality.
"""

import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, Counter

try:
    from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for type hints when prometheus is not available
    class CollectorRegistry:
        pass


class MetricType(Enum):
    """Types of metrics collected by the orchestrator."""
    DOWNLOAD_STARTED = "download_started"
    DOWNLOAD_COMPLETED = "download_completed"
    DOWNLOAD_FAILED = "download_failed"
    ERROR_OCCURRED = "error_occurred"
    STORAGE_USAGE = "storage_usage"
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_TIMEOUT = "lock_timeout"
    INTEGRITY_CHECK = "integrity_check"
    GC_RUN = "gc_run"


@dataclass
class MetricEvent:
    """Represents a metric event with labels and value."""
    metric_type: MetricType
    labels: Dict[str, str]
    value: float = 1.0
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MetricsCollector:
    """
    Collects and manages Prometheus metrics for the Model Orchestrator.
    
    Provides both Prometheus integration when available and in-memory
    fallback for environments without prometheus_client.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self._lock = threading.Lock()
        self._enabled = PROMETHEUS_AVAILABLE
        self._registry = registry
        
        # In-memory fallback metrics
        self._counters: Dict[str, Counter] = defaultdict(Counter)
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._histograms: Dict[str, list] = defaultdict(list)
        
        if self._enabled:
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics with limited cardinality labels."""
        if not self._enabled:
            return

        # Download metrics
        self.download_total = PrometheusCounter(
            'model_downloads_total',
            'Total number of model downloads',
            ['model_id', 'variant', 'source', 'status'],
            registry=self._registry
        )
        
        self.download_duration = Histogram(
            'model_download_duration_seconds',
            'Time spent downloading models',
            ['model_id', 'variant', 'source'],
            registry=self._registry
        )
        
        self.download_bytes = PrometheusCounter(
            'model_download_bytes_total',
            'Total bytes downloaded',
            ['model_id', 'variant', 'source'],
            registry=self._registry
        )

        # Storage metrics
        self.storage_bytes_used = Gauge(
            'model_storage_bytes_used',
            'Storage space used by models',
            ['model_family'],
            registry=self._registry
        )
        
        self.storage_files_total = Gauge(
            'model_storage_files_total',
            'Total number of model files',
            ['model_family'],
            registry=self._registry
        )

        # Error metrics
        self.errors_total = PrometheusCounter(
            'model_errors_total',
            'Total number of errors',
            ['error_code', 'model_id', 'source'],
            registry=self._registry
        )

        # Lock metrics
        self.lock_timeouts_total = PrometheusCounter(
            'model_lock_timeouts_total',
            'Total number of lock timeouts',
            ['model_id'],
            registry=self._registry
        )

        # Integrity metrics
        self.integrity_failures_total = PrometheusCounter(
            'model_integrity_failures_total',
            'Total number of integrity check failures',
            ['model_id', 'file_path'],
            registry=self._registry
        )

        # Garbage collection metrics
        self.gc_runs_total = PrometheusCounter(
            'model_gc_runs_total',
            'Total number of garbage collection runs',
            ['trigger_reason'],
            registry=self._registry
        )
        
        self.gc_bytes_reclaimed = PrometheusCounter(
            'model_gc_bytes_reclaimed_total',
            'Total bytes reclaimed by garbage collection',
            ['trigger_reason'],
            registry=self._registry
        )

        # Deduplication metrics
        self.deduplication_runs_total = PrometheusCounter(
            'model_deduplication_runs_total',
            'Total number of deduplication runs',
            ['model_id', 'type'],
            registry=self._registry
        )
        
        self.deduplication_bytes_saved = PrometheusCounter(
            'model_deduplication_bytes_saved_total',
            'Total bytes saved through deduplication',
            ['model_id', 'type'],
            registry=self._registry
        )
        
        self.deduplication_links_created = PrometheusCounter(
            'model_deduplication_links_created_total',
            'Total number of links created during deduplication',
            ['model_id', 'type'],
            registry=self._registry
        )

    def record_download_started(self, model_id: str, variant: str, source: str):
        """Record that a download has started."""
        labels = {
            'model_id': self._sanitize_label(model_id),
            'variant': variant or 'default',
            'source': self._extract_source_type(source),
            'status': 'started'
        }
        
        if self._enabled:
            self.download_total.labels(**labels).inc()
        else:
            with self._lock:
                self._counters['downloads_total'][str(labels)] += 1

    def record_download_completed(self, model_id: str, variant: str, source: str, 
                                duration: float, bytes_downloaded: int):
        """Record a successful download completion."""
        labels = {
            'model_id': self._sanitize_label(model_id),
            'variant': variant or 'default',
            'source': self._extract_source_type(source)
        }
        
        if self._enabled:
            self.download_total.labels(**labels, status='completed').inc()
            self.download_duration.labels(**labels).observe(duration)
            self.download_bytes.labels(**labels).inc(bytes_downloaded)
        else:
            with self._lock:
                completed_labels = {**labels, 'status': 'completed'}
                self._counters['downloads_total'][str(completed_labels)] += 1
                self._histograms['download_duration'].append((labels, duration))
                self._counters['download_bytes'][str(labels)] += bytes_downloaded

    def record_download_failed(self, model_id: str, variant: str, source: str, error_code: str):
        """Record a failed download."""
        download_labels = {
            'model_id': self._sanitize_label(model_id),
            'variant': variant or 'default',
            'source': self._extract_source_type(source),
            'status': 'failed'
        }
        
        error_labels = {
            'error_code': error_code,
            'model_id': self._sanitize_label(model_id),
            'source': self._extract_source_type(source)
        }
        
        if self._enabled:
            self.download_total.labels(**download_labels).inc()
            self.errors_total.labels(**error_labels).inc()
        else:
            with self._lock:
                self._counters['downloads_total'][str(download_labels)] += 1
                self._counters['errors_total'][str(error_labels)] += 1

    def record_storage_usage(self, model_family: str, bytes_used: int, files_count: int):
        """Record current storage usage for a model family."""
        if self._enabled:
            self.storage_bytes_used.labels(model_family=model_family).set(bytes_used)
            self.storage_files_total.labels(model_family=model_family).set(files_count)
        else:
            with self._lock:
                self._gauges['storage_bytes_used'][model_family] = bytes_used
                self._gauges['storage_files_total'][model_family] = files_count

    def record_lock_timeout(self, model_id: str):
        """Record a lock timeout event."""
        labels = {'model_id': self._sanitize_label(model_id)}
        
        if self._enabled:
            self.lock_timeouts_total.labels(**labels).inc()
        else:
            with self._lock:
                self._counters['lock_timeouts_total'][str(labels)] += 1

    def record_integrity_failure(self, model_id: str, file_path: str):
        """Record an integrity check failure."""
        labels = {
            'model_id': self._sanitize_label(model_id),
            'file_path': self._sanitize_label(file_path)
        }
        
        if self._enabled:
            self.integrity_failures_total.labels(**labels).inc()
        else:
            with self._lock:
                self._counters['integrity_failures_total'][str(labels)] += 1

    def record_gc_run(self, trigger_reason: str, bytes_reclaimed: int):
        """Record a garbage collection run."""
        labels = {'trigger_reason': trigger_reason}
        
        if self._enabled:
            self.gc_runs_total.labels(**labels).inc()
            self.gc_bytes_reclaimed.labels(**labels).inc(bytes_reclaimed)
        else:
            with self._lock:
                self._counters['gc_runs_total'][str(labels)] += 1
                self._counters['gc_bytes_reclaimed'][str(labels)] += bytes_reclaimed

    def record_deduplication_completed(self, model_id: str, bytes_saved: int, links_created: int, dedup_type: str = "single_model"):
        """Record a completed deduplication operation."""
        labels = {
            'model_id': self._sanitize_label(model_id),
            'type': dedup_type
        }
        
        if self._enabled:
            self.deduplication_runs_total.labels(**labels).inc()
            self.deduplication_bytes_saved.labels(**labels).inc(bytes_saved)
            self.deduplication_links_created.labels(**labels).inc(links_created)
        else:
            with self._lock:
                self._counters['deduplication_runs_total'][str(labels)] += 1
                self._counters['deduplication_bytes_saved'][str(labels)] += bytes_saved
                self._counters['deduplication_links_created'][str(labels)] += links_created

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if self._enabled and self._registry:
            return generate_latest(self._registry).decode('utf-8')
        else:
            return self._generate_fallback_metrics()

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary for JSON serialization."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {k: len(v) for k, v in self._histograms.items()},
                'prometheus_available': self._enabled
            }

    def _sanitize_label(self, value: str) -> str:
        """Sanitize label values to prevent cardinality explosion."""
        if not value:
            return 'unknown'
        
        # Remove version suffixes and variants to limit cardinality
        if '@' in value:
            value = value.split('@')[0]
        
        # Limit length and sanitize characters
        value = value[:50]  # Limit length
        return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in value)

    def _extract_source_type(self, source: str) -> str:
        """Extract source type from URL to limit cardinality."""
        if not source:
            return 'unknown'
        
        if source.startswith('local://'):
            return 'local'
        elif source.startswith('s3://'):
            return 's3'
        elif source.startswith('hf://'):
            return 'huggingface'
        elif source.startswith('http://') or source.startswith('https://'):
            return 'http'
        else:
            return 'unknown'

    def _generate_fallback_metrics(self) -> str:
        """Generate Prometheus-style text format from in-memory metrics."""
        lines = []
        
        with self._lock:
            # Counters
            for metric_name, counter in self._counters.items():
                lines.append(f"# TYPE {metric_name} counter")
                for labels_str, value in counter.items():
                    lines.append(f"{metric_name}{labels_str} {value}")
            
            # Gauges
            for metric_name, gauge in self._gauges.items():
                lines.append(f"# TYPE {metric_name} gauge")
                for labels, value in gauge.items():
                    lines.append(f"{metric_name}{{model_family=\"{labels}\"}} {value}")
        
        return '\n'.join(lines)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def initialize_metrics(registry: Optional[CollectorRegistry] = None) -> MetricsCollector:
    """Initialize the global metrics collector with optional custom registry."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(registry)
    return _metrics_collector