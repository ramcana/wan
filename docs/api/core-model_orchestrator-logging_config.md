---
title: core.model_orchestrator.logging_config
category: api
tags: [api, core]
---

# core.model_orchestrator.logging_config

Structured logging configuration for the Model Orchestrator system.

## Classes

### StructuredFormatter

Custom formatter that outputs structured JSON logs.

#### Methods

##### __init__(self: Any, include_extra: bool)



##### format(self: Any, record: logging.LogRecord) -> str

Format log record as structured JSON.

##### formatTime(self: Any, record: logging.LogRecord, datefmt: <ast.Subscript object at 0x0000019431929750>) -> str

Format timestamp in ISO format.

##### _is_sensitive_field(self: Any, field_name: str) -> bool

Check if a field contains sensitive information.

##### _mask_sensitive_value(self: Any, value: str) -> str

Mask sensitive values in logs.

### CorrelationIdFilter

Filter to add correlation ID to log records.

#### Methods

##### filter(self: Any, record: logging.LogRecord) -> bool

Add correlation ID to the log record if not already present.

### ModelOrchestratorLogger

Centralized logger configuration for the Model Orchestrator.

#### Methods

##### __init__(self: Any, name: str)



##### configure(self: Any, level: str, structured: bool, output_file: <ast.Subscript object at 0x000001942F29BC70>) -> logging.Logger

Configure the logger with structured output.

##### get_logger(self: Any, name: <ast.Subscript object at 0x00000194345E0460>) -> logging.Logger

Get a logger instance.

### LogContext

Context manager for setting correlation ID and additional context.

#### Methods

##### __init__(self: Any, correlation_id: <ast.Subscript object at 0x00000194345E2740>)



##### __enter__(self: Any)



##### __exit__(self: Any, exc_type: Any, exc_val: Any, exc_tb: Any)



##### log(self: Any, level: str, message: str)

Log a message with the current context.

### PerformanceTimer

Context manager for measuring and logging operation performance.

#### Methods

##### __init__(self: Any, operation: str, logger: <ast.Subscript object at 0x000001942FC01ED0>)



##### __enter__(self: Any)



##### __exit__(self: Any, exc_type: Any, exc_val: Any, exc_tb: Any)



##### duration(self: Any) -> <ast.Subscript object at 0x000001943460A2F0>

Get the duration of the operation if completed.

