---
title: core.fallback_recovery_system
category: api
tags: [api, core]
---

# core.fallback_recovery_system

Fallback and Recovery System for Real AI Model Integration

This module implements comprehensive fallback and recovery mechanisms that automatically
handle failures in model loading, generation pipeline, and system optimization.

## Classes

### RecoveryAction

Types of recovery actions that can be performed

### FailureType

Types of failures that can trigger recovery

### RecoveryAttempt

Information about a recovery attempt

### SystemHealthStatus

Current system health status

### FallbackRecoverySystem

Comprehensive fallback and recovery system that handles various failure scenarios
and automatically attempts recovery using existing infrastructure

#### Methods

##### __init__(self: Any, generation_service: Any, websocket_manager: Any)



##### _initialize_recovery_strategies(self: Any) -> <ast.Subscript object at 0x0000019430122E30>

Initialize recovery strategies for different failure types

##### start_health_monitoring(self: Any)

Start continuous system health monitoring

##### stop_health_monitoring(self: Any)

Stop continuous system health monitoring

##### _health_monitor_worker(self: Any)

Background worker for continuous health monitoring

##### _is_in_cooldown(self: Any, failure_type: FailureType) -> bool

Check if we're in cooldown period for a failure type

##### _get_cooldown_remaining(self: Any, failure_type: FailureType) -> float

Get remaining cooldown time in seconds

##### get_recovery_statistics(self: Any) -> <ast.Subscript object at 0x0000019431B350C0>

Get statistics about recovery attempts

##### reset_recovery_state(self: Any)

Reset recovery state (useful for testing or manual recovery)

## Constants

### FALLBACK_TO_MOCK

Type: `str`

Value: `fallback_to_mock`

### RETRY_MODEL_DOWNLOAD

Type: `str`

Value: `retry_model_download`

### APPLY_VRAM_OPTIMIZATION

Type: `str`

Value: `apply_vram_optimization`

### RESTART_PIPELINE

Type: `str`

Value: `restart_pipeline`

### CLEAR_GPU_CACHE

Type: `str`

Value: `clear_gpu_cache`

### REDUCE_GENERATION_PARAMS

Type: `str`

Value: `reduce_generation_params`

### ENABLE_CPU_OFFLOAD

Type: `str`

Value: `enable_cpu_offload`

### SYSTEM_HEALTH_CHECK

Type: `str`

Value: `system_health_check`

### MODEL_LOADING_FAILURE

Type: `str`

Value: `model_loading_failure`

### VRAM_EXHAUSTION

Type: `str`

Value: `vram_exhaustion`

### GENERATION_PIPELINE_ERROR

Type: `str`

Value: `generation_pipeline_error`

### HARDWARE_OPTIMIZATION_FAILURE

Type: `str`

Value: `hardware_optimization_failure`

### SYSTEM_RESOURCE_ERROR

Type: `str`

Value: `system_resource_error`

### NETWORK_ERROR

Type: `str`

Value: `network_error`

