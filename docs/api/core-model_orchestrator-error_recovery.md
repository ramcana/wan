---
title: core.model_orchestrator.error_recovery
category: api
tags: [api, core]
---

# core.model_orchestrator.error_recovery

Error recovery and retry logic for the Model Orchestrator system.

## Classes

### RetryStrategy

Retry strategies for different types of failures.

### FailureCategory

Categories of failures for recovery strategy selection.

### RetryConfig

Configuration for retry behavior.

### RecoveryContext

Context information for error recovery.

### ErrorRecoveryManager

Manages error recovery strategies and retry logic.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x000001942CE750C0>)



##### get_failure_category(self: Any, error: Exception) -> FailureCategory

Determine the failure category for an error.

##### get_retry_config(self: Any, error: Exception) -> RetryConfig

Get the retry configuration for a specific error.

##### calculate_delay(self: Any, attempt: int, config: RetryConfig) -> float

Calculate the delay before the next retry attempt.

##### should_retry(self: Any, error: Exception, attempt: int) -> bool

Determine if an operation should be retried.

##### recovery_context(self: Any, operation: str)

Create a recovery context for an operation.

##### retry_with_recovery(self: Any, func: <ast.Subscript object at 0x000001942CE23520>, context: RecoveryContext) -> T

Execute a function with retry logic and error recovery.

##### _apply_recovery_strategy(self: Any, error: Exception, context: RecoveryContext)

Apply recovery strategies based on the error type.

##### _handle_auth_failure(self: Any, error: Exception, context: RecoveryContext)

Handle authentication failures.

##### _handle_rate_limit(self: Any, error: Exception, context: RecoveryContext)

Handle rate limiting.

##### _handle_resource_failure(self: Any, error: Exception, context: RecoveryContext)

Handle resource failures like disk space.

##### _handle_integrity_failure(self: Any, error: Exception, context: RecoveryContext)

Handle integrity failures like checksum mismatches.

##### _handle_concurrency_failure(self: Any, error: Exception, context: RecoveryContext)

Handle concurrency failures like lock timeouts.

## Constants

### T

Type: `unknown`

### EXPONENTIAL_BACKOFF

Type: `str`

Value: `exponential_backoff`

### LINEAR_BACKOFF

Type: `str`

Value: `linear_backoff`

### FIXED_DELAY

Type: `str`

Value: `fixed_delay`

### NO_RETRY

Type: `str`

Value: `no_retry`

### TRANSIENT

Type: `str`

Value: `transient`

### AUTHENTICATION

Type: `str`

Value: `authentication`

### RATE_LIMIT

Type: `str`

Value: `rate_limit`

### INTEGRITY

Type: `str`

Value: `integrity`

### RESOURCE

Type: `str`

Value: `resource`

### PERMANENT

Type: `str`

Value: `permanent`

### CONCURRENCY

Type: `str`

Value: `concurrency`

