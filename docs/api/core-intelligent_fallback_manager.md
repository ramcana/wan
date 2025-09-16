---
title: core.intelligent_fallback_manager
category: api
tags: [api, core]
---

# core.intelligent_fallback_manager

Intelligent Fallback Manager for Enhanced Model Availability

This module provides smart alternatives when preferred models are unavailable,
implements model compatibility scoring algorithms, and manages fallback strategies
with request queuing and wait time estimation.

## Classes

### FallbackType

Types of fallback strategies available

### ModelCapability

Model capabilities for compatibility scoring

### GenerationRequirements

Requirements for a generation request

### ModelSuggestion

Suggestion for an alternative model

### FallbackStrategy

Strategy for handling unavailable models

### EstimatedWaitTime

Estimated wait time for model availability

### QueuedRequest

Queued generation request waiting for model availability

### QueueResult

Result of queuing a request

### IntelligentFallbackManager

Intelligent fallback manager that provides smart alternatives when preferred models
are unavailable, implements compatibility scoring, and manages request queuing.

#### Methods

##### __init__(self: Any, availability_manager: Any, models_dir: <ast.Subscript object at 0x000001942C855B10>)

Initialize the Intelligent Fallback Manager.

Args:
    availability_manager: ModelAvailabilityManager instance
    models_dir: Directory for storing model metadata

##### _initialize_model_capabilities(self: Any) -> <ast.Subscript object at 0x000001942C81F9D0>

Initialize model capabilities database

##### _initialize_compatibility_matrix(self: Any) -> <ast.Subscript object at 0x000001942C81F3D0>

Initialize model compatibility scoring matrix

##### _estimate_performance_difference(self: Any, requested_model: str, candidate_model: str) -> float

Estimate performance difference between models (-1.0 to 1.0)

##### _estimate_vram_requirement(self: Any, model_id: str, resolution: str) -> float

Estimate VRAM requirement for model and resolution

##### _estimate_generation_time(self: Any, model_id: str, requirements: GenerationRequirements) -> timedelta

Estimate generation time for model and requirements

##### _estimate_model_size(self: Any, model_id: str) -> float

Estimate model size in GB

##### _get_priority_value(self: Any, priority: str) -> int

Convert priority string to numeric value for sorting

## Constants

### PERFORMANCE_MONITORING_AVAILABLE

Type: `bool`

Value: `True`

### ALTERNATIVE_MODEL

Type: `str`

Value: `alternative_model`

### QUEUE_AND_WAIT

Type: `str`

Value: `queue_and_wait`

### MOCK_GENERATION

Type: `str`

Value: `mock_generation`

### DOWNLOAD_AND_RETRY

Type: `str`

Value: `download_and_retry`

### REDUCE_REQUIREMENTS

Type: `str`

Value: `reduce_requirements`

### HYBRID_APPROACH

Type: `str`

Value: `hybrid_approach`

### TEXT_TO_VIDEO

Type: `str`

Value: `text_to_video`

### IMAGE_TO_VIDEO

Type: `str`

Value: `image_to_video`

### TEXT_IMAGE_TO_VIDEO

Type: `str`

Value: `text_image_to_video`

### HIGH_RESOLUTION

Type: `str`

Value: `high_resolution`

### FAST_GENERATION

Type: `str`

Value: `fast_generation`

### HIGH_QUALITY

Type: `str`

Value: `high_quality`

### PERFORMANCE_MONITORING_AVAILABLE

Type: `bool`

Value: `False`

