---
title: tests.test_integrated_error_handler
category: api
tags: [api, tests]
---

# tests.test_integrated_error_handler

Tests for Integrated Error Handler

Tests the enhanced error handling system that bridges FastAPI backend
with existing GenerationErrorHandler infrastructure.

## Classes

### TestIntegratedErrorHandler

Test the IntegratedErrorHandler class

#### Methods

##### handler(self: Any)

Create an IntegratedErrorHandler instance for testing

##### mock_existing_handler(self: Any)

Create a mock existing error handler

##### test_handler_initialization(self: Any, handler: Any)

Test that the handler initializes correctly

##### test_fastapi_error_patterns_initialization(self: Any, handler: Any)

Test that FastAPI-specific error patterns are initialized

##### test_enhance_context_for_fastapi(self: Any, handler: Any)

Test context enhancement for FastAPI integration

##### test_get_fastapi_recovery_suggestions(self: Any, handler: Any)

Test FastAPI-specific recovery suggestions

##### test_categorize_error_integrated(self: Any, handler: Any)

Test error categorization with integrated patterns

##### test_determine_severity_integrated(self: Any, handler: Any)

Test error severity determination

##### test_get_system_status(self: Any, handler: Any)

Test system status reporting

##### test_get_error_categories(self: Any, handler: Any)

Test error categories retrieval

### TestConvenienceFunctions

Test convenience functions for common error scenarios

### TestGlobalErrorHandler

Test global error handler instance management

#### Methods

##### test_get_integrated_error_handler(self: Any)

Test global error handler retrieval

### TestErrorHandlerIntegration

Test integration with existing error handling infrastructure

### TestErrorRecovery

Test automatic error recovery functionality

### TestErrorContextEnhancement

Test error context enhancement for FastAPI integration

#### Methods

##### test_context_enhancement_with_generation_service(self: Any)

Test context enhancement when generation service is provided

##### test_context_enhancement_without_generation_service(self: Any)

Test context enhancement without generation service

## Constants

### INFRASTRUCTURE_AVAILABLE

Type: `bool`

Value: `True`

### INFRASTRUCTURE_AVAILABLE

Type: `bool`

Value: `False`

