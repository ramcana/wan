---
title: core.performance_monitor
category: api
tags: [api, core]
---

# core.performance_monitor

Performance monitoring system for real AI model integration.
Tracks generation performance, resource usage, and provides optimization recommendations.

## Classes

### PerformanceMetrics

Performance metrics for a generation task.

#### Methods

##### __post_init__(self: Any)



### SystemPerformanceSnapshot

System performance snapshot at a point in time.

### PerformanceAnalysis

Analysis of performance trends and recommendations.

### PerformanceMonitor

Monitors and analyzes system and generation performance.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x0000019434503DF0>)



##### _init_gpu_monitoring(self: Any)

Initialize GPU monitoring if available.

##### start_monitoring(self: Any)

Start continuous system monitoring.

##### stop_monitoring(self: Any)

Stop continuous system monitoring.

##### _monitoring_loop(self: Any)

Main monitoring loop.

##### _recheck_gpu_availability(self: Any)

Re-check GPU availability during monitoring

##### _capture_system_snapshot(self: Any) -> SystemPerformanceSnapshot

Capture current system performance snapshot.

##### start_task_monitoring(self: Any, task_id: str, model_type: str, resolution: str, steps: int) -> PerformanceMetrics

Start monitoring a generation task.

##### update_task_metrics(self: Any, task_id: str)

Update metrics for an active task.

##### complete_task_monitoring(self: Any, task_id: str, success: bool, error_category: <ast.Subscript object at 0x0000019431A074F0>) -> <ast.Subscript object at 0x0000019431A04BB0>

Complete monitoring for a task and calculate final metrics.

##### _calculate_resource_usage(self: Any, metrics: PerformanceMetrics)

Calculate resource usage for a task from system snapshots.

##### get_performance_analysis(self: Any, time_window_hours: int) -> PerformanceAnalysis

Analyze performance over a time window.

##### _analyze_bottlenecks(self: Any, metrics: <ast.Subscript object at 0x000001942EF27460>) -> <ast.Subscript object at 0x00000194318ECF10>

Analyze system bottlenecks from metrics.

##### _generate_optimization_recommendations(self: Any, metrics: <ast.Subscript object at 0x00000194318ECFA0>, bottlenecks: <ast.Subscript object at 0x00000194318EEFE0>) -> <ast.Subscript object at 0x000001942FD79F60>

Generate optimization recommendations based on analysis.

##### _calculate_resource_efficiency(self: Any, metrics: <ast.Subscript object at 0x000001942FD79E10>) -> float

Calculate overall resource efficiency score (0-1).

##### _calculate_performance_trends(self: Any, metrics: <ast.Subscript object at 0x000001942EFBDAB0>) -> <ast.Subscript object at 0x0000019434509120>

Calculate performance trends over time.

##### get_current_system_status(self: Any) -> <ast.Subscript object at 0x0000019434509C90>

Get current system performance status.

##### export_metrics(self: Any, filepath: str, time_window_hours: int)

Export performance metrics to JSON file.

