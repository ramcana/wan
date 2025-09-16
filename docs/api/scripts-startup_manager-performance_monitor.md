---
title: scripts.startup_manager.performance_monitor
category: api
tags: [api, scripts]
---

# scripts.startup_manager.performance_monitor

Performance monitoring system for startup manager.

This module provides comprehensive timing metrics collection, success/failure rate tracking,
resource usage monitoring, and trend analysis for the startup process.

## Classes

### MetricType

Types of metrics collected.

### StartupPhase

Startup phases for timing measurement.

### TimingMetric

Individual timing measurement.

#### Methods

##### finish(self: Any, success: bool, error_message: <ast.Subscript object at 0x0000019429C69A20>)

Mark the timing as finished.

### ResourceSnapshot

System resource usage snapshot.

#### Methods

##### capture(cls: Any) -> ResourceSnapshot

Capture current system resource usage.

### StartupSession

Complete startup session data.

#### Methods

##### finish(self: Any, success: bool)

Mark the session as finished.

### PerformanceStats

Aggregated performance statistics.

### PerformanceMonitor

Comprehensive performance monitoring system.

Features:
- Timing metrics collection for each startup phase
- Success/failure rate tracking with trend analysis
- Resource usage monitoring during startup
- Historical data storage and analysis
- Performance optimization suggestions

#### Methods

##### __init__(self: Any, data_dir: <ast.Subscript object at 0x000001942CCF7190>, max_sessions: int, resource_sampling_interval: float)

Initialize performance monitor.

Args:
    data_dir: Directory to store performance data
    max_sessions: Maximum number of sessions to keep in memory
    resource_sampling_interval: Interval for resource sampling in seconds

##### start_session(self: Any, metadata: <ast.Subscript object at 0x0000019428D66E00>) -> str

Start a new startup session.

Args:
    metadata: Optional metadata for the session
    
Returns:
    Session ID

##### finish_session(self: Any, success: bool)

Finish the current startup session.

Args:
    success: Whether the startup was successful

##### start_timing(self: Any, operation: str, metadata: <ast.Subscript object at 0x0000019428D67250>) -> str

Start timing an operation.

Args:
    operation: Name of the operation
    metadata: Optional metadata for the timing
    
Returns:
    Timing ID

##### finish_timing(self: Any, timing_id: str, success: bool, error_message: <ast.Subscript object at 0x0000019428D65240>)

Finish timing an operation.

Args:
    timing_id: ID returned from start_timing
    success: Whether the operation was successful
    error_message: Error message if operation failed

##### time_operation(self: Any, operation: str, metadata: <ast.Subscript object at 0x000001942A290220>)

Context manager for timing operations.

Args:
    operation: Name of the operation
    metadata: Optional metadata for the timing
    
Usage:
    with monitor.time_operation("environment_validation"):
        # Do validation work
        pass

##### record_error(self: Any, error_message: str, operation: <ast.Subscript object at 0x000001942A291A20>)

Record an error during startup.

Args:
    error_message: Error message
    operation: Operation where error occurred

##### get_performance_stats(self: Any, days: int) -> PerformanceStats

Get aggregated performance statistics.

Args:
    days: Number of days to include in statistics
    
Returns:
    Performance statistics

##### get_resource_usage_summary(self: Any) -> <ast.Subscript object at 0x000001942CB082E0>

Get resource usage summary for current or last session.

Returns:
    Resource usage summary

##### _start_resource_monitoring(self: Any)

Start background resource monitoring.

##### _stop_resource_monitoring(self: Any)

Stop background resource monitoring.

##### _resource_monitor_loop(self: Any)

Background loop for resource monitoring.

##### _analyze_trend(self: Any, sessions: <ast.Subscript object at 0x000001942CB081C0>) -> str

Analyze performance trend from recent sessions.

Args:
    sessions: List of recent sessions
    
Returns:
    Trend direction: "improving", "degrading", or "stable"

##### _load_historical_data(self: Any)

Load historical performance data from disk.

##### _save_session_data(self: Any, session: StartupSession)

Save session data to disk.

##### _save_performance_history(self: Any)

Save performance history to disk.

### TimingContext

Context manager for timing operations.

#### Methods

##### __init__(self: Any, monitor: PerformanceMonitor, operation: str, metadata: <ast.Subscript object at 0x000001942A269000>)



##### __enter__(self: Any)



##### __exit__(self: Any, exc_type: Any, exc_val: Any, exc_tb: Any)



## Constants

### TIMING

Type: `str`

Value: `timing`

### SUCCESS_RATE

Type: `str`

Value: `success_rate`

### RESOURCE_USAGE

Type: `str`

Value: `resource_usage`

### ERROR_COUNT

Type: `str`

Value: `error_count`

### PHASE_DURATION

Type: `str`

Value: `phase_duration`

### ENVIRONMENT_VALIDATION

Type: `str`

Value: `environment_validation`

### PORT_MANAGEMENT

Type: `str`

Value: `port_management`

### PROCESS_STARTUP

Type: `str`

Value: `process_startup`

### HEALTH_VERIFICATION

Type: `str`

Value: `health_verification`

### TOTAL_STARTUP

Type: `str`

Value: `total_startup`

