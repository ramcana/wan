---
title: core.model_orchestrator.gpu_health
category: api
tags: [api, core]
---

# core.model_orchestrator.gpu_health

GPU-based health checks for WAN2.2 models.

This module provides smoke tests for t2v/i2v/ti2v models using minimal
GPU operations to validate model functionality without full inference.

## Classes

### HealthStatus

Health check status values.

### HealthCheckResult

Result of a health check operation.

### GPUHealthChecker

Performs lightweight GPU health checks for WAN2.2 models.

Uses minimal denoise steps at low resolution to validate model
functionality without consuming excessive GPU resources.

#### Methods

##### __init__(self: Any, device: <ast.Subscript object at 0x0000019434376F50>, timeout: float)



##### check_model_health(self: Any, model_id: str, model_path: str) -> HealthCheckResult

Perform health check for a specific model.

Args:
    model_id: Model identifier (e.g., "t2v-A14B")
    model_path: Path to the model directory
    
Returns:
    HealthCheckResult with status and details

##### _check_t2v_model(self: Any, model_id: str, model_path: str) -> HealthCheckResult

Health check for text-to-video models.

##### _check_i2v_model(self: Any, model_id: str, model_path: str) -> HealthCheckResult

Health check for image-to-video models.

##### _check_ti2v_model(self: Any, model_id: str, model_path: str) -> HealthCheckResult

Health check for text+image-to-video models.

##### _check_generic_model(self: Any, model_id: str, model_path: str) -> HealthCheckResult

Generic health check for unknown model types.

##### _validate_t2v_components(self: Any, model_path: str) -> bool

Validate T2V model components without loading full model.

##### _validate_i2v_components(self: Any, model_path: str) -> bool

Validate I2V model components.

##### _validate_ti2v_components(self: Any, model_path: str) -> bool

Validate TI2V model components (dual conditioning).

##### _validate_model_files(self: Any, model_path: str) -> bool

Basic validation of model files.

##### _run_t2v_smoke_test(self: Any, model_path: str) -> bool

Run minimal T2V inference test.

##### _run_i2v_smoke_test(self: Any, model_path: str) -> bool

Run minimal I2V inference test.

##### _run_ti2v_smoke_test(self: Any, model_path: str) -> bool

Run minimal TI2V inference test with dual conditioning.

##### _get_gpu_memory_used(self: Any) -> <ast.Subscript object at 0x000001942F341DE0>

Get current GPU memory usage in bytes.

##### _timeout_context(self: Any, timeout: float)

Context manager for operation timeout.

##### _clear_old_cache_entries(self: Any)

Clear expired cache entries.

##### clear_cache(self: Any)

Clear all cached health check results.

##### get_system_health(self: Any) -> <ast.Subscript object at 0x00000194340C6020>

Get overall system health information.

## Constants

### HEALTHY

Type: `str`

Value: `healthy`

### DEGRADED

Type: `str`

Value: `degraded`

### UNHEALTHY

Type: `str`

Value: `unhealthy`

### UNKNOWN

Type: `str`

Value: `unknown`

