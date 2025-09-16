---
title: scripts.startup_manager.process_manager
category: api
tags: [api, scripts]
---

# scripts.startup_manager.process_manager

Process Manager for WAN22 server startup management.
Handles server process lifecycle, health monitoring, and cleanup.

## Classes

### ProcessStatus

Process status enumeration.

### ProcessInfo

Information about a managed process.

### ProcessResult

Result of a process operation.

#### Methods

##### success_result(cls: Any, process_info: ProcessInfo) -> ProcessResult

Create a successful process result.

##### failure_result(cls: Any, error_message: str, details: <ast.Subscript object at 0x000001942CE91720>) -> ProcessResult

Create a failed process result.

### HealthMonitor

Health monitoring for server processes.

#### Methods

##### __init__(self: Any, check_interval: float)



##### add_process(self: Any, process_info: ProcessInfo)

Add a process to monitor.

##### remove_process(self: Any, process_name: str)

Remove a process from monitoring.

##### start_monitoring(self: Any)

Start health monitoring in background thread.

##### stop_monitoring(self: Any)

Stop health monitoring.

##### _monitor_loop(self: Any)

Main monitoring loop.

##### _check_process_health(self: Any, process_info: ProcessInfo)

Check health of a single process.

### ProcessManager

Manages server process lifecycle and health monitoring.

#### Methods

##### __init__(self: Any, config: StartupConfig)



##### start_backend(self: Any, port: int, backend_config: <ast.Subscript object at 0x00000194285AD450>) -> ProcessResult

Start FastAPI backend server.

##### start_frontend(self: Any, port: int, frontend_config: <ast.Subscript object at 0x000001942856CEB0>) -> ProcessResult

Start React frontend development server.

##### _detect_package_manager(self: Any, frontend_dir: Path) -> <ast.Subscript object at 0x000001942C5C7D30>

Detect which package manager to use (npm or yarn).

##### _start_process(self: Any, process_info: ProcessInfo) -> ProcessResult

Start a process with the given configuration.

##### get_process_status(self: Any, process_name: str) -> <ast.Subscript object at 0x000001942C5C5CF0>

Get current status of a process.

##### is_process_healthy(self: Any, process_name: str) -> bool

Check if a process is healthy.

##### wait_for_health(self: Any, process_name: str, timeout: float) -> bool

Wait for a process to become healthy.

##### get_all_processes(self: Any) -> <ast.Subscript object at 0x000001942CBE4A60>

Get information about all managed processes.

##### cleanup(self: Any)

Clean up all processes and monitoring.

##### stop_process(self: Any, process_name: str, force: bool) -> bool

Stop a specific process.

##### graceful_shutdown(self: Any, process_name: str, timeout: float) -> bool

Gracefully shutdown a process with SIGTERM/SIGKILL escalation.

##### cleanup_zombie_processes(self: Any) -> <ast.Subscript object at 0x000001942CE06C20>

Clean up zombie processes and file locks.

##### restart_process(self: Any, process_name: str, max_attempts: int) -> ProcessResult

Restart a process with exponential backoff.

##### _restart_generic_process(self: Any, process_info: ProcessInfo) -> ProcessResult

Restart a generic process using stored configuration.

##### auto_restart_failed_processes(self: Any) -> <ast.Subscript object at 0x000001942CBB66B0>

Automatically restart failed processes that have auto_restart enabled.

##### get_process_metrics(self: Any, process_name: str) -> <ast.Subscript object at 0x000001942CBB7EE0>

Get detailed metrics for a process.

##### set_auto_restart(self: Any, process_name: str, enabled: bool) -> bool

Enable or disable auto-restart for a process.

##### reset_restart_count(self: Any, process_name: str) -> bool

Reset the restart count for a process.

## Constants

### STARTING

Type: `str`

Value: `starting`

### RUNNING

Type: `str`

Value: `running`

### FAILED

Type: `str`

Value: `failed`

### STOPPED

Type: `str`

Value: `stopped`

### UNKNOWN

Type: `str`

Value: `unknown`

