---
title: core.model_health_monitor
category: api
tags: [api, core]
---

# core.model_health_monitor

Model Health Monitor
Provides integrity checking, performance monitoring, corruption detection,
and automated health checks for WAN2.2 models.

## Classes

### HealthStatus

Model health status enumeration

### CorruptionType

Types of corruption that can be detected

### IntegrityResult

Result of model integrity check

### PerformanceMetrics

Performance metrics for model operations

### PerformanceHealth

Health assessment based on performance metrics

### CorruptionReport

Detailed corruption analysis report

### SystemHealthReport

Overall system health report

### HealthCheckConfig

Configuration for health monitoring

### ModelHealthMonitor

Comprehensive model health monitoring system with integrity checking,
performance monitoring, corruption detection, and automated health checks.

#### Methods

##### __init__(self: Any, models_dir: <ast.Subscript object at 0x0000019428AC0E20>, config: <ast.Subscript object at 0x0000019428AC1BA0>)

Initialize the model health monitor.

Args:
    models_dir: Directory containing models
    config: Health check configuration

##### add_health_callback(self: Any, callback: <ast.Subscript object at 0x0000019428AC2800>)

Add a callback for health check results

##### add_corruption_callback(self: Any, callback: <ast.Subscript object at 0x0000019428AC3D90>)

Add a callback for corruption detection

##### _get_model_path(self: Any, model_id: str) -> Path

Get the path to a model directory

##### _detect_model_type(self: Any, model_id: str) -> str

Detect model type from ID

##### _calculate_trend_slope(self: Any, values: <ast.Subscript object at 0x0000019427B6C250>) -> float

Calculate trend slope using simple linear regression

## Constants

### PERFORMANCE_MONITORING_AVAILABLE

Type: `bool`

Value: `True`

### HEALTHY

Type: `str`

Value: `healthy`

### DEGRADED

Type: `str`

Value: `degraded`

### CORRUPTED

Type: `str`

Value: `corrupted`

### MISSING

Type: `str`

Value: `missing`

### UNKNOWN

Type: `str`

Value: `unknown`

### FILE_MISSING

Type: `str`

Value: `file_missing`

### CHECKSUM_MISMATCH

Type: `str`

Value: `checksum_mismatch`

### INCOMPLETE_DOWNLOAD

Type: `str`

Value: `incomplete_download`

### INVALID_FORMAT

Type: `str`

Value: `invalid_format`

### PERMISSION_ERROR

Type: `str`

Value: `permission_error`

### DISK_ERROR

Type: `str`

Value: `disk_error`

### PERFORMANCE_MONITORING_AVAILABLE

Type: `bool`

Value: `False`

