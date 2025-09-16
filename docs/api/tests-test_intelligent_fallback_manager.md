---
title: tests.test_intelligent_fallback_manager
category: api
tags: [api, tests]
---

# tests.test_intelligent_fallback_manager

Unit tests for Intelligent Fallback Manager

Tests model compatibility scoring algorithms, alternative model suggestions,
fallback strategy decision engine, request queuing, and wait time calculations.

## Classes

### MockAvailabilityStatus

Mock availability status for testing

### MockModelStatus

Mock model status for testing

### MockAvailabilityManager

Mock availability manager for testing

#### Methods

##### __init__(self: Any)



##### set_model_status(self: Any, model_id: str, status: MockAvailabilityStatus)

Helper method to change model status for testing

### TestIntelligentFallbackManager

Test suite for IntelligentFallbackManager

#### Methods

##### test_initialization(self: Any, mock_availability_manager: Any)

Test proper initialization of the fallback manager

##### test_model_capabilities_initialization(self: Any, fallback_manager: Any)

Test that model capabilities are properly initialized

##### test_compatibility_matrix_initialization(self: Any, fallback_manager: Any)

Test that compatibility matrix is properly initialized

##### test_estimate_performance_difference(self: Any, fallback_manager: Any)

Test performance difference estimation

##### test_estimate_vram_requirement(self: Any, fallback_manager: Any)

Test VRAM requirement estimation

##### test_estimate_generation_time(self: Any, fallback_manager: Any)

Test generation time estimation

##### test_estimate_model_size(self: Any, fallback_manager: Any)

Test model size estimation

##### test_get_priority_value(self: Any, fallback_manager: Any)

Test priority value conversion

### TestGlobalInstanceManagement

Test global instance management functions

#### Methods

##### test_get_intelligent_fallback_manager_singleton(self: Any)

Test that get_intelligent_fallback_manager returns singleton

### TestErrorHandling

Test error handling in various scenarios

### TestEdgeCases

Test edge cases and boundary conditions

#### Methods

##### test_compatibility_scoring_with_unknown_models(self: Any, fallback_manager: Any)

Test compatibility scoring with unknown models

## Constants

### AVAILABLE

Type: `str`

Value: `available`

### MISSING

Type: `str`

Value: `missing`

### DOWNLOADING

Type: `str`

Value: `downloading`

### CORRUPTED

Type: `str`

Value: `corrupted`

