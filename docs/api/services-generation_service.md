---
title: services.generation_service
category: api
tags: [api, services]
---

# services.generation_service

Enhanced Generation service with real AI model integration
Integrates with existing Wan2.2 system using ModelIntegrationBridge and RealGenerationPipeline

## Classes

### TaskSubmissionResult

Structured response for generation task submissions.

### ModelType

Centralized model type definitions

#### Methods

##### normalize(cls: Any, mt: str) -> str

Normalize model type string to canonical form with sane default

### VRAMMonitor

Enhanced VRAM monitoring and management for generation tasks

#### Methods

##### __init__(self: Any, total_vram_gb: float, optimal_usage_gb: float, system_optimizer: Any)



##### get_current_vram_usage(self: Any) -> <ast.Subscript object at 0x0000019434444E80>

Get current VRAM usage statistics with multi-GPU support

##### check_vram_availability(self: Any, required_gb: float) -> <ast.Subscript object at 0x00000194344440D0>

Check VRAM availability with structured suggestions

### GenerationService

Enhanced service for managing video generation tasks with real AI integration

#### Methods

##### __init__(self: Any)



##### shutdown(self: Any, timeout: float)

Clean shutdown of the generation service

##### _ensure_processing_thread(self: Any)

Ensure the background processing thread is running.

##### _estimate_wan_model_vram_requirements(self: Any, model_type: str, resolution: str) -> float

Estimate VRAM requirements for WAN models based on type and resolution

##### _create_fallback_error_handler(self: Any)

Create a fallback error handler if existing system is not available

##### _determine_failure_type(self: Any, error: Exception, model_type: str) -> FailureType

Determine the type of failure based on the error and context

##### _process_queue_worker(self: Any)

Background worker that processes generation tasks from the database

### FallbackErrorHandler



#### Methods

##### handle_error(self: Any, error: Any, context: Any)

Handle errors with basic categorization

##### _categorize_error(self: Any, error_message: Any)

Categorize error for better handling

## Constants

### T2V_A14B

Type: `str`

Value: `t2v-A14B`

### I2V_A14B

Type: `str`

Value: `i2v-A14B`

### TI2V_5B

Type: `str`

Value: `ti2v-5b`

