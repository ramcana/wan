---
title: core.performance_monitoring_system
category: api
tags: [api, core]
---

# core.performance_monitoring_system

Performance Monitoring and Optimization System for Enhanced Model Availability

This module provides comprehensive performance tracking for download operations,
health checks, fallback strategies, and system resource usage monitoring.

## Classes

### PerformanceMetricType

Types of performance metrics tracked

### PerformanceMetric

Individual performance metric data

### SystemResourceSnapshot

System resource usage snapshot

### PerformanceReport

Comprehensive performance report

### PerformanceTracker

Tracks individual performance metrics

#### Methods

##### __init__(self: Any)



##### start_operation(self: Any, metric_type: PerformanceMetricType, operation_name: str, metadata: <ast.Subscript object at 0x0000019431B07F70>) -> str

Start tracking a performance operation

##### end_operation(self: Any, operation_id: str, success: bool, error_message: <ast.Subscript object at 0x0000019431B07130>, additional_metadata: <ast.Subscript object at 0x0000019431B07070>) -> <ast.Subscript object at 0x0000019431ADDAE0>

End tracking a performance operation

##### _capture_resource_usage(self: Any) -> <ast.Subscript object at 0x0000019431ADCEB0>

Capture current system resource usage

##### get_metrics_by_type(self: Any, metric_type: PerformanceMetricType, hours_back: int) -> <ast.Subscript object at 0x0000019431ADC640>

Get metrics of specific type within time window

##### get_all_metrics(self: Any, hours_back: int) -> <ast.Subscript object at 0x0000019431B5FF40>

Get all metrics within time window

### SystemResourceMonitor

Monitors system resource usage continuously

#### Methods

##### __init__(self: Any, sample_interval: int)



##### _capture_snapshot(self: Any) -> SystemResourceSnapshot

Capture current system resource snapshot

##### get_resource_history(self: Any, hours_back: int) -> <ast.Subscript object at 0x000001943450DC60>

Get resource usage history

##### get_current_usage(self: Any) -> SystemResourceSnapshot

Get current resource usage

### PerformanceAnalyzer

Analyzes performance data and provides optimization recommendations

#### Methods

##### __init__(self: Any, tracker: PerformanceTracker, resource_monitor: SystemResourceMonitor)



##### generate_performance_report(self: Any, hours_back: int) -> PerformanceReport

Generate comprehensive performance report

##### _calculate_resource_summary(self: Any, resource_history: <ast.Subscript object at 0x0000019434546E90>) -> <ast.Subscript object at 0x0000019434500C10>

Calculate resource usage summary statistics

##### _identify_bottlenecks(self: Any, metrics: <ast.Subscript object at 0x0000019434500A60>, resource_history: <ast.Subscript object at 0x00000194345009A0>) -> <ast.Subscript object at 0x0000019434586FB0>

Identify performance bottlenecks

##### _generate_recommendations(self: Any, metrics: <ast.Subscript object at 0x0000019434586E60>, resource_history: <ast.Subscript object at 0x0000019434586DA0>, bottlenecks: <ast.Subscript object at 0x0000019434586CE0>) -> <ast.Subscript object at 0x00000194318D2290>

Generate optimization recommendations

### PerformanceMonitoringSystem

Main performance monitoring and optimization system

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x00000194318D0B50>)



##### _load_config(self: Any, config_path: <ast.Subscript object at 0x00000194344444C0>) -> <ast.Subscript object at 0x000001942EFBDF90>

Load performance monitoring configuration

##### track_download_operation(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019431AD2260>) -> str

Track a download operation

##### track_health_check(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019431AD2740>) -> str

Track a health check operation

##### track_fallback_strategy(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019431AD2C20>) -> str

Track a fallback strategy operation

##### track_analytics_collection(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019431AD3100>) -> str

Track an analytics collection operation

##### track_model_operation(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019431AD35E0>) -> str

Track a model operation

##### end_tracking(self: Any, operation_id: str, success: bool, error_message: <ast.Subscript object at 0x0000019431AD3B50>, additional_metadata: <ast.Subscript object at 0x0000019431AD3C10>) -> <ast.Subscript object at 0x0000019432E3C040>

End tracking an operation

##### get_performance_report(self: Any, hours_back: int) -> PerformanceReport

Get comprehensive performance report

##### get_dashboard_data(self: Any, force_refresh: bool) -> <ast.Subscript object at 0x0000019432E3DE40>

Get performance data for dashboard display

##### _calculate_resource_trends(self: Any) -> <ast.Subscript object at 0x0000019432E3FAC0>

Calculate resource usage trends

## Constants

### DOWNLOAD_OPERATION

Type: `str`

Value: `download_operation`

### HEALTH_CHECK

Type: `str`

Value: `health_check`

### FALLBACK_STRATEGY

Type: `str`

Value: `fallback_strategy`

### ANALYTICS_COLLECTION

Type: `str`

Value: `analytics_collection`

### SYSTEM_RESOURCE

Type: `str`

Value: `system_resource`

### MODEL_OPERATION

Type: `str`

Value: `model_operation`

