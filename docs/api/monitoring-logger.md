---
title: monitoring.logger
category: api
tags: [api, monitoring]
---

# monitoring.logger

Production logging system with structured logging and monitoring.

## Classes

### StructuredFormatter

Custom formatter for structured JSON logging.

#### Methods

##### format(self: Any, record: logging.LogRecord) -> str

Format log record as structured JSON.

### PerformanceLogger

Logger for performance metrics and monitoring.

#### Methods

##### __init__(self: Any)



##### log_request(self: Any, method: str, path: str, status_code: int, duration: float, user_id: <ast.Subscript object at 0x000001942FBF1E70>) -> None

Log HTTP request metrics.

##### log_generation(self: Any, task_id: str, model_type: str, duration: float, success: bool, error_type: <ast.Subscript object at 0x000001942FBF1F60>) -> None

Log generation task metrics.

##### log_system_metrics(self: Any, cpu_percent: float, ram_used_gb: float, gpu_percent: float, vram_used_mb: float) -> None

Log system resource metrics.

##### log_queue_metrics(self: Any, queue_size: int, processing_count: int, completed_count: int, failed_count: int) -> None

Log queue metrics.

### ErrorLogger

Logger for error tracking and monitoring.

#### Methods

##### __init__(self: Any)



##### log_error(self: Any, error: Exception, context: <ast.Subscript object at 0x0000019431A06860>, user_id: <ast.Subscript object at 0x0000019431A068C0>, request_id: <ast.Subscript object at 0x0000019431A07B50>) -> None

Log error with context information.

##### log_validation_error(self: Any, field: str, value: Any, message: str, request_id: <ast.Subscript object at 0x0000019431A064A0>) -> None

Log validation error.

##### log_generation_error(self: Any, task_id: str, model_type: str, error: Exception, prompt: <ast.Subscript object at 0x0000019431A04D60>) -> None

Log generation-specific error.

