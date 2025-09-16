---
title: scripts.startup_manager.logger
category: api
tags: [api, scripts]
---

# scripts.startup_manager.logger

Comprehensive logging system for the startup manager.

This module provides structured logging with multiple outputs, rotation policies,
and different verbosity levels for debugging and troubleshooting.

## Classes

### LogLevel

Log levels with color mapping.

### LogEntry

Structured log entry for JSON logging.

### ColoredFormatter

Custom formatter that adds colors to console output.

#### Methods

##### __init__(self: Any, fmt: str, datefmt: str)



##### format(self: Any, record: logging.LogRecord) -> str



### JSONFormatter

Custom formatter for JSON output.

#### Methods

##### format(self: Any, record: logging.LogRecord) -> str



### StartupLogger

Comprehensive logging system for startup manager.

Features:
- Multiple output formats (console, file, JSON)
- Log rotation and cleanup
- Colored console output
- Structured logging for programmatic analysis
- Different verbosity levels

#### Methods

##### __init__(self: Any, name: str, log_dir: <ast.Subscript object at 0x000001942CE12F80>, console_level: str, file_level: str, json_level: str, max_file_size: int, backup_count: int, cleanup_days: int)

Initialize the logging system.

Args:
    name: Logger name
    log_dir: Directory for log files
    console_level: Console logging level
    file_level: File logging level
    json_level: JSON logging level
    max_file_size: Maximum size per log file in bytes
    backup_count: Number of backup files to keep
    cleanup_days: Days after which to clean up old logs

##### _setup_console_handler(self: Any)

Setup colored console handler.

##### _setup_file_handler(self: Any)

Setup rotating file handler.

##### _setup_json_handler(self: Any)

Setup JSON handler for structured logging.

##### _cleanup_old_logs(self: Any)

Clean up log files older than cleanup_days.

##### debug(self: Any, message: str, extra_data: <ast.Subscript object at 0x0000019429C8BA00>)

Log debug message.

##### info(self: Any, message: str, extra_data: <ast.Subscript object at 0x0000019429C8B4F0>)

Log info message.

##### warning(self: Any, message: str, extra_data: <ast.Subscript object at 0x0000019427BB7DF0>)

Log warning message.

##### error(self: Any, message: str, extra_data: <ast.Subscript object at 0x000001942C622E60>)

Log error message.

##### critical(self: Any, message: str, extra_data: <ast.Subscript object at 0x000001942C623970>)

Log critical message.

##### _log(self: Any, level: int, message: str, extra_data: <ast.Subscript object at 0x000001942C622A40>)

Internal logging method with extra data support.

##### log_startup_phase(self: Any, phase: str, details: <ast.Subscript object at 0x000001942C621DE0>)

Log startup phase with structured data.

##### log_error_with_context(self: Any, error: Exception, context: <ast.Subscript object at 0x000001942C622320>)

Log error with full context for debugging.

##### log_performance_metric(self: Any, operation: str, duration: float, details: <ast.Subscript object at 0x000001942C621B10>)

Log performance metrics.

##### log_system_info(self: Any, info: <ast.Subscript object at 0x000001942CE2E470>)

Log system information.

##### set_console_level(self: Any, level: str)

Dynamically change console logging level.

##### get_log_files(self: Any) -> <ast.Subscript object at 0x000001942CE2F790>

Get list of current log files.

## Constants

### DEBUG

Type: `unknown`

### INFO

Type: `unknown`

### WARNING

Type: `unknown`

### ERROR

Type: `unknown`

### CRITICAL

Type: `unknown`

