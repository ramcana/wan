---
title: core.integrated_error_handler
category: api
tags: [api, core]
---

# core.integrated_error_handler

Integrated Error Handler for Real AI Model Integration

This module provides enhanced error handling that bridges the FastAPI backend
with the existing GenerationErrorHandler from the Wan2.2 infrastructure.

## Classes

### IntegratedErrorHandler

Enhanced error handler that integrates FastAPI backend with existing
GenerationErrorHandler infrastructure for comprehensive error management.

#### Methods

##### __init__(self: Any)



##### _setup_logging(self: Any)

Setup logging configuration for integrated error handling

##### _initialize_fastapi_error_patterns(self: Any) -> <ast.Subscript object at 0x000001942811A4D0>

Initialize FastAPI-specific error patterns

##### _initialize_recovery_strategies(self: Any) -> <ast.Subscript object at 0x0000019427C87970>

Initialize recovery strategies specific to FastAPI integration

##### _enhance_context_for_fastapi(self: Any, context: <ast.Subscript object at 0x0000019427C866B0>) -> <ast.Subscript object at 0x0000019427C858D0>

Enhance error context with FastAPI-specific information

##### _enhance_error_for_fastapi(self: Any, user_error: UserFriendlyError, context: <ast.Subscript object at 0x0000019427C856C0>) -> UserFriendlyError

Enhance existing error with FastAPI-specific recovery suggestions

##### _get_fastapi_recovery_suggestions(self: Any, category: ErrorCategory, context: <ast.Subscript object at 0x0000019427C40400>) -> <ast.Subscript object at 0x0000019427C33130>

Get FastAPI-specific recovery suggestions

##### _handle_error_integrated(self: Any, error: Exception, context: <ast.Subscript object at 0x0000019427C32F80>) -> UserFriendlyError

Handle error using integrated fallback implementation

##### _categorize_error_integrated(self: Any, error_message: str, context: <ast.Subscript object at 0x0000019427C31E10>) -> ErrorCategory

Categorize error using integrated patterns

##### _determine_severity_integrated(self: Any, error: Exception, category: ErrorCategory) -> ErrorSeverity

Determine error severity for integrated handling

##### _generate_error_message_integrated(self: Any, error: Exception, category: ErrorCategory, context: <ast.Subscript object at 0x0000019427C1EA40>) -> <ast.Subscript object at 0x0000019427C1E1D0>

Generate user-friendly error message for integrated handling

##### _generate_recovery_suggestions_integrated(self: Any, category: ErrorCategory, error: Exception, context: <ast.Subscript object at 0x0000019427C71C60>) -> <ast.Subscript object at 0x0000019427C70E80>

Generate recovery suggestions for integrated handling

##### _format_technical_details_integrated(self: Any, error: Exception, context: <ast.Subscript object at 0x0000019427C70CD0>) -> str

Format technical details for integrated handling

##### get_error_categories(self: Any) -> <ast.Subscript object at 0x000001942C65FC70>

Get list of available error categories

##### get_system_status(self: Any) -> <ast.Subscript object at 0x000001942C65EB60>

Get current system status for error context

##### _is_wan_model_error(self: Any, error: Exception, context: <ast.Subscript object at 0x0000019428088AC0>) -> bool

Check if this is a WAN model-specific error

### ErrorCategory



### ErrorSeverity



### RecoveryAction



#### Methods

##### __post_init__(self: Any)



### UserFriendlyError



#### Methods

##### __post_init__(self: Any)



## Constants

### TORCH_AVAILABLE

Type: `bool`

Value: `True`

### EXISTING_ERROR_HANDLER_AVAILABLE

Type: `bool`

Value: `True`

### TORCH_AVAILABLE

Type: `bool`

Value: `False`

### EXISTING_ERROR_HANDLER_AVAILABLE

Type: `bool`

Value: `False`

### VRAM_MEMORY

Type: `str`

Value: `vram_memory`

### MODEL_LOADING

Type: `str`

Value: `model_loading`

### GENERATION_PIPELINE

Type: `str`

Value: `generation_pipeline`

### INPUT_VALIDATION

Type: `str`

Value: `input_validation`

### SYSTEM_RESOURCE

Type: `str`

Value: `system_resource`

### UNKNOWN

Type: `str`

Value: `unknown`

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

