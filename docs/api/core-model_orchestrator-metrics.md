---
title: core.model_orchestrator.metrics
category: api
tags: [api, core]
---

# core.model_orchestrator.metrics

Prometheus metrics for Model Orchestrator observability.

This module provides comprehensive metrics collection for downloads, errors,
storage usage, and performance tracking with limited cardinality.

## Classes

### MetricType

Types of metrics collected by the orchestrator.

### MetricEvent

Represents a metric event with labels and value.

#### Methods

##### __post_init__(self: Any)



### MetricsCollector

Collects and manages Prometheus metrics for the Model Orchestrator.

Provides both Prometheus integration when available and in-memory
fallback for environments without prometheus_client.

#### Methods

##### __init__(self: Any, registry: <ast.Subscript object at 0x0000019427A22890>)



##### _setup_prometheus_metrics(self: Any)

Initialize Prometheus metrics with limited cardinality labels.

##### record_download_started(self: Any, model_id: str, variant: str, source: str)

Record that a download has started.

##### record_download_completed(self: Any, model_id: str, variant: str, source: str, duration: float, bytes_downloaded: int)

Record a successful download completion.

##### record_download_failed(self: Any, model_id: str, variant: str, source: str, error_code: str)

Record a failed download.

##### record_storage_usage(self: Any, model_family: str, bytes_used: int, files_count: int)

Record current storage usage for a model family.

##### record_lock_timeout(self: Any, model_id: str)

Record a lock timeout event.

##### record_integrity_failure(self: Any, model_id: str, file_path: str)

Record an integrity check failure.

##### record_gc_run(self: Any, trigger_reason: str, bytes_reclaimed: int)

Record a garbage collection run.

##### record_deduplication_completed(self: Any, model_id: str, bytes_saved: int, links_created: int, dedup_type: str)

Record a completed deduplication operation.

##### get_metrics_text(self: Any) -> str

Get metrics in Prometheus text format.

##### get_metrics_dict(self: Any) -> <ast.Subscript object at 0x000001942A246470>

Get metrics as a dictionary for JSON serialization.

##### _sanitize_label(self: Any, value: str) -> str

Sanitize label values to prevent cardinality explosion.

##### _extract_source_type(self: Any, source: str) -> str

Extract source type from URL to limit cardinality.

##### _generate_fallback_metrics(self: Any) -> str

Generate Prometheus-style text format from in-memory metrics.

### CollectorRegistry



## Constants

### PROMETHEUS_AVAILABLE

Type: `bool`

Value: `True`

### DOWNLOAD_STARTED

Type: `str`

Value: `download_started`

### DOWNLOAD_COMPLETED

Type: `str`

Value: `download_completed`

### DOWNLOAD_FAILED

Type: `str`

Value: `download_failed`

### ERROR_OCCURRED

Type: `str`

Value: `error_occurred`

### STORAGE_USAGE

Type: `str`

Value: `storage_usage`

### LOCK_ACQUIRED

Type: `str`

Value: `lock_acquired`

### LOCK_TIMEOUT

Type: `str`

Value: `lock_timeout`

### INTEGRITY_CHECK

Type: `str`

Value: `integrity_check`

### GC_RUN

Type: `str`

Value: `gc_run`

### PROMETHEUS_AVAILABLE

Type: `bool`

Value: `False`

