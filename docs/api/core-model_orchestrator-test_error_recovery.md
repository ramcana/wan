---
title: core.model_orchestrator.test_error_recovery
category: api
tags: [api, core]
---

# core.model_orchestrator.test_error_recovery

Tests for the error recovery and retry logic system.

## Classes

### TestRetryConfiguration

Test retry configuration and strategy selection.

#### Methods

##### test_default_configuration(self: Any, recovery_manager: Any)

Test default retry configuration.

##### test_custom_configuration(self: Any)

Test custom retry configuration.

##### test_error_specific_overrides(self: Any)

Test error-specific retry configuration overrides.

### TestFailureCategories

Test failure category classification.

#### Methods

##### test_transient_failures(self: Any, recovery_manager: Any)

Test classification of transient failures.

##### test_permanent_failures(self: Any, recovery_manager: Any)

Test classification of permanent failures.

##### test_resource_failures(self: Any, recovery_manager: Any)

Test classification of resource failures.

##### test_integrity_failures(self: Any, recovery_manager: Any)

Test classification of integrity failures.

### TestDelayCalculation

Test retry delay calculation strategies.

#### Methods

##### test_exponential_backoff(self: Any, recovery_manager: Any)

Test exponential backoff delay calculation.

##### test_linear_backoff(self: Any, recovery_manager: Any)

Test linear backoff delay calculation.

##### test_fixed_delay(self: Any, recovery_manager: Any)

Test fixed delay strategy.

##### test_no_retry(self: Any, recovery_manager: Any)

Test no retry strategy.

##### test_jitter_application(self: Any, recovery_manager: Any)

Test that jitter is applied when enabled.

### TestRetryLogic

Test the core retry logic.

#### Methods

##### test_successful_operation_no_retry(self: Any, recovery_manager: Any)

Test that successful operations don't retry.

##### test_transient_failure_retry(self: Any, recovery_manager: Any, custom_config: Any)

Test retry behavior for transient failures.

##### test_permanent_failure_no_retry(self: Any, recovery_manager: Any)

Test that permanent failures don't retry.

##### test_max_attempts_exceeded(self: Any, custom_config: Any)

Test behavior when max attempts are exceeded.

##### test_delay_between_retries(self: Any)

Test that delays are applied between retries.

### TestRecoveryStrategies

Test specific recovery strategies for different error types.

#### Methods

##### test_auth_failure_recovery(self: Any, recovery_manager: Any)

Test recovery strategy for authentication failures.

##### test_rate_limit_recovery(self: Any, recovery_manager: Any)

Test recovery strategy for rate limiting.

##### test_resource_failure_recovery(self: Any, recovery_manager: Any)

Test recovery strategy for resource failures.

##### test_integrity_failure_recovery(self: Any, recovery_manager: Any)

Test recovery strategy for integrity failures.

##### test_concurrency_failure_recovery(self: Any, recovery_manager: Any)

Test recovery strategy for concurrency failures.

### TestConvenienceFunctions

Test convenience functions and decorators.

#### Methods

##### test_retry_operation_function(self: Any, custom_config: Any)

Test the retry_operation convenience function.

##### test_with_retry_decorator(self: Any, custom_config: Any)

Test the with_retry decorator.

##### test_decorator_with_context_kwargs(self: Any)

Test decorator with additional context kwargs.

### TestRecoveryContext

Test recovery context management.

#### Methods

##### test_context_creation(self: Any, recovery_manager: Any)

Test recovery context creation and management.

##### test_context_attempt_tracking(self: Any)

Test that context tracks attempt numbers correctly.

### TestErrorRecoveryIntegration

Test integration of error recovery with other components.

#### Methods

##### test_recovery_with_logging(self: Any, recovery_manager: Any, caplog: Any)

Test that recovery operations are properly logged.

##### test_recovery_preserves_exception_details(self: Any, recovery_manager: Any)

Test that recovery preserves original exception details.

