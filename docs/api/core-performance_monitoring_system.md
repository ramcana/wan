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



##### start_operation(self: Any, metric_type: PerformanceMetricType, operation_name: str, metadata: <ast.Subscript object at 0x0000019428820520>) -> str

Start tracking a performance operation

##### end_operation(self: Any, operation_id: str, success: bool, error_message: <ast.Subscript object at 0x000001942880F6A0>, additional_metadata: <ast.Subscript object at 0x000001942880F5E0>) -> <ast.Subscript object at 0x0000019428826050>

End tracking a performance operation

##### _capture_resource_usage(self: Any) -> <ast.Subscript object at 0x0000019428825420>

Capture current system resource usage

##### get_metrics_by_type(self: Any, metric_type: PerformanceMetricType, hours_back: int) -> <ast.Subscript object at 0x0000019428824BB0>

Get metrics of specific type within time window

##### get_all_metrics(self: Any, hours_back: int) -> <ast.Subscript object at 0x00000194288244F0>

Get all metrics within time window

### SystemResourceMonitor

Monitors system resource usage continuously

#### Methods

##### __init__(self: Any, sample_interval: int)



##### _capture_snapshot(self: Any) -> SystemResourceSnapshot

Capture current system resource snapshot

##### get_resource_history(self: Any, hours_back: int) -> <ast.Subscript object at 0x00000194284761D0>

Get resource usage history

##### get_current_usage(self: Any) -> SystemResourceSnapshot

Get current resource usage

### PerformanceAnalyzer

Analyzes performance data and provides optimization recommendations

#### Methods

##### __init__(self: Any, tracker: PerformanceTracker, resource_monitor: SystemResourceMonitor)



##### generate_performance_report(self: Any, hours_back: int) -> PerformanceReport

Generate comprehensive performance report

##### _calculate_resource_summary(self: Any, resource_history: <ast.Subscript object at 0x000001942846B400>) -> <ast.Subscript object at 0x0000019428471180>

Calculate resource usage summary statistics

##### _identify_bottlenecks(self: Any, metrics: <ast.Subscript object at 0x0000019428470FD0>, resource_history: <ast.Subscript object at 0x0000019428470F10>) -> <ast.Subscript object at 0x0000019428497520>

Identify performance bottlenecks

##### _generate_recommendations(self: Any, metrics: <ast.Subscript object at 0x00000194284973D0>, resource_history: <ast.Subscript object at 0x0000019428497310>, bottlenecks: <ast.Subscript object at 0x0000019428497250>) -> <ast.Subscript object at 0x00000194283E0850>

Generate optimization recommendations

### PerformanceMonitoringSystem

Main performance monitoring and optimization system

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x00000194283E0400>)



##### _load_config(self: Any, config_path: <ast.Subscript object at 0x000001942CD5F670>) -> <ast.Subscript object at 0x000001942CC88A30>

Load performance monitoring configuration

##### track_download_operation(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019428846260>) -> str

Track a download operation

##### track_health_check(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019428845330>) -> str

Track a health check operation

##### track_fallback_strategy(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019428845AB0>) -> str

Track a fallback strategy operation

##### track_analytics_collection(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019428847D60>) -> str

Track an analytics collection operation

##### track_model_operation(self: Any, operation_name: str, metadata: <ast.Subscript object at 0x0000019428847100>) -> str

Track a model operation

##### end_tracking(self: Any, operation_id: str, success: bool, error_message: <ast.Subscript object at 0x00000194288D8100>, additional_metadata: <ast.Subscript object at 0x00000194288D81C0>) -> <ast.Subscript object at 0x00000194288D85B0>

End tracking an operation

##### get_performance_report(self: Any, hours_back: int) -> PerformanceReport

Get comprehensive performance report

##### get_dashboard_data(self: Any, force_refresh: bool) -> <ast.Subscript object at 0x00000194288DA3B0>

Get performance data for dashboard display

##### _calculate_resource_trends(self: Any) -> <ast.Subscript object at 0x000001942A1A0070>

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

