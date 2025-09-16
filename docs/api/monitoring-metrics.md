---
title: monitoring.metrics
category: api
tags: [api, monitoring]
---

# monitoring.metrics

Application metrics collection and monitoring.

## Classes

### SystemMetrics

System resource metrics.

### ApplicationMetrics

Application-specific metrics.

### PerformanceMetrics

Performance metrics.

### MetricsCollector

Collects and stores application metrics.

#### Methods

##### __init__(self: Any, history_size: int)



##### collect_system_metrics(self: Any) -> SystemMetrics

Collect current system metrics.

##### collect_application_metrics(self: Any) -> ApplicationMetrics

Collect current application metrics.

##### collect_performance_metrics(self: Any) -> PerformanceMetrics

Collect current performance metrics.

##### record_request(self: Any, endpoint: str, response_time: float, status_code: int) -> None

Record HTTP request metrics.

##### record_generation(self: Any, duration: float, success: bool, error_type: <ast.Subscript object at 0x00000194288F4250>) -> None

Record generation metrics.

##### update_queue_size(self: Any, size: int) -> None

Update current queue size.

##### update_active_generations(self: Any, count: int) -> None

Update active generation count.

##### get_latest_metrics(self: Any) -> <ast.Subscript object at 0x00000194280121A0>

Get latest metrics from all categories.

##### get_metrics_history(self: Any, minutes: int) -> <ast.Subscript object at 0x0000019428010730>

Get metrics history for specified time period.

##### get_health_status(self: Any) -> <ast.Subscript object at 0x00000194288F20E0>

Get overall system health status.

##### export_metrics(self: Any, filepath: str) -> None

Export metrics to JSON file.

### MetricsMonitor

Background metrics monitoring service.

#### Methods

##### __init__(self: Any, collector: MetricsCollector, collection_interval: int)



##### start(self: Any) -> None

Start metrics collection in background thread.

##### stop(self: Any) -> None

Stop metrics collection.

##### _collect_loop(self: Any) -> None

Main collection loop.

## Constants

### NVIDIA_AVAILABLE

Type: `bool`

Value: `True`

### NVIDIA_AVAILABLE

Type: `bool`

Value: `False`

