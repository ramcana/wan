---
title: core.enhanced_error_recovery
category: api
tags: [api, core]
---

# core.enhanced_error_recovery



## Classes

### EnhancedFailureType

Enhanced failure types with more granular categorization

### RecoveryStrategy

Enhanced recovery strategies

### ErrorSeverity

Error severity levels

### ErrorContext

Enhanced error context with detailed information

### RecoveryResult

Enhanced recovery result with detailed information

### RecoveryMetrics

Metrics for recovery success tracking and optimization

### EnhancedErrorRecovery

Enhanced Error Recovery System that extends FallbackRecoverySystem
with sophisticated error categorization and multi-strategy recovery

#### Methods

##### __init__(self: Any, base_recovery_system: <ast.Subscript object at 0x000001942CBC0850>, model_availability_manager: <ast.Subscript object at 0x000001942CBC0790>, intelligent_fallback_manager: <ast.Subscript object at 0x000001942CBC06D0>, model_health_monitor: <ast.Subscript object at 0x000001942CBC0610>, enhanced_downloader: <ast.Subscript object at 0x000001942CBC0550>, websocket_manager: <ast.Subscript object at 0x000001942CBC0490>)



##### _initialize_strategy_mapping(self: Any) -> <ast.Subscript object at 0x000001942CB897B0>

Initialize recovery strategy mapping for different failure types

##### _initialize_error_messages(self: Any) -> <ast.Subscript object at 0x000001942CBE0DF0>

Initialize user-friendly error messages and actionable steps

##### _convert_to_base_failure_type(self: Any, enhanced_type: EnhancedFailureType) -> FailureType

Convert enhanced failure type to base failure type

##### _update_strategy_success_rate(self: Any, strategy: RecoveryStrategy, success: bool)

Update success rate metrics for recovery strategies

##### _determine_failure_type(self: Any, error: Exception, context: <ast.Subscript object at 0x0000019428321C30>) -> EnhancedFailureType

Determine the enhanced failure type based on error and context

##### _determine_error_severity(self: Any, failure_type: EnhancedFailureType, error: Exception, context: <ast.Subscript object at 0x00000194283D8250>) -> ErrorSeverity

Determine error severity based on failure type and context

##### reset_metrics(self: Any)

Reset recovery metrics (for testing or maintenance)

## Constants

### MODEL_DOWNLOAD_FAILURE

Type: `str`

Value: `model_download_failure`

### MODEL_CORRUPTION_DETECTED

Type: `str`

Value: `model_corruption_detected`

### MODEL_VERSION_MISMATCH

Type: `str`

Value: `model_version_mismatch`

### MODEL_LOADING_TIMEOUT

Type: `str`

Value: `model_loading_timeout`

### MODEL_INTEGRITY_FAILURE

Type: `str`

Value: `model_integrity_failure`

### MODEL_COMPATIBILITY_ERROR

Type: `str`

Value: `model_compatibility_error`

### VRAM_EXHAUSTION

Type: `str`

Value: `vram_exhaustion`

### STORAGE_SPACE_INSUFFICIENT

Type: `str`

Value: `storage_space_insufficient`

### NETWORK_CONNECTIVITY_LOSS

Type: `str`

Value: `network_connectivity_loss`

### BANDWIDTH_LIMITATION

Type: `str`

Value: `bandwidth_limitation`

### HARDWARE_OPTIMIZATION_FAILURE

Type: `str`

Value: `hardware_optimization_failure`

### GENERATION_PIPELINE_ERROR

Type: `str`

Value: `generation_pipeline_error`

### SYSTEM_RESOURCE_ERROR

Type: `str`

Value: `system_resource_error`

### DEPENDENCY_MISSING

Type: `str`

Value: `dependency_missing`

### INVALID_PARAMETERS

Type: `str`

Value: `invalid_parameters`

### UNSUPPORTED_OPERATION

Type: `str`

Value: `unsupported_operation`

### PERMISSION_DENIED

Type: `str`

Value: `permission_denied`

### IMMEDIATE_RETRY

Type: `str`

Value: `immediate_retry`

### INTELLIGENT_FALLBACK

Type: `str`

Value: `intelligent_fallback`

### AUTOMATIC_REPAIR

Type: `str`

Value: `automatic_repair`

### PARAMETER_ADJUSTMENT

Type: `str`

Value: `parameter_adjustment`

### RESOURCE_OPTIMIZATION

Type: `str`

Value: `resource_optimization`

### USER_INTERVENTION

Type: `str`

Value: `user_intervention`

### GRACEFUL_DEGRADATION

Type: `str`

Value: `graceful_degradation`

### LOW

Type: `str`

Value: `low`

### MEDIUM

Type: `str`

Value: `medium`

### HIGH

Type: `str`

Value: `high`

### CRITICAL

Type: `str`

Value: `critical`

