---
title: scripts.startup_manager.recovery_engine
category: api
tags: [api, scripts]
---

# scripts.startup_manager.recovery_engine

Recovery Engine for Server Startup Management System

This module provides error classification, recovery strategies, and intelligent
failure handling for the WAN22 server startup process.

## Classes

### ErrorType

Classification of different error types that can occur during startup

### RecoveryAction

Represents a specific recovery action that can be taken

### StartupError

Represents an error that occurred during startup

### RecoveryResult

Result of a recovery attempt

### ErrorPatternMatcher

Matches error messages to specific error types using patterns

#### Methods

##### __init__(self: Any)



##### classify_error(self: Any, error_message: str, exception: <ast.Subscript object at 0x0000019431A3ACE0>) -> ErrorType

Classify an error based on its message and exception type

### RetryStrategy

Implements exponential backoff retry logic

#### Methods

##### __init__(self: Any, max_attempts: int, base_delay: float, max_delay: float)



##### execute_with_retry(self: Any, operation: Callable, operation_name: str) -> Any

Execute an operation with exponential backoff retry

### RecoveryEngine

Main recovery engine that handles error classification and recovery strategies

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x0000019434304AC0>)



##### load_configuration(self: Any)

Load recovery engine configuration

##### save_configuration(self: Any)

Save current configuration

##### classify_error(self: Any, error_message: str, exception: <ast.Subscript object at 0x0000019432D7C2E0>, context: <ast.Subscript object at 0x0000019432D7C0A0>) -> StartupError

Classify an error and create a StartupError object

##### _get_suggested_actions(self: Any, error_type: ErrorType) -> <ast.Subscript object at 0x000001942F3C7AF0>

Get human-readable suggested actions for an error type

##### _is_auto_fixable(self: Any, error_type: ErrorType) -> bool

Determine if an error type can be automatically fixed

##### _get_error_details(self: Any, error_type: ErrorType, exception: <ast.Subscript object at 0x000001942F3C7820>, context: <ast.Subscript object at 0x000001942F3C74F0>) -> <ast.Subscript object at 0x000001942F436620>

Get detailed information about the error

##### _get_process_using_port(self: Any, port: int) -> <ast.Subscript object at 0x000001942F4357E0>

Get information about the process using a specific port

##### _register_recovery_actions(self: Any)

Register all available recovery actions

##### register_recovery_action(self: Any, error_type: ErrorType, action: RecoveryAction)

Register a recovery action for a specific error type

##### get_recovery_actions(self: Any, error_type: ErrorType) -> <ast.Subscript object at 0x000001942F415690>

Get available recovery actions for an error type

##### attempt_recovery(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942F4154E0>) -> RecoveryResult

Attempt to recover from an error

##### _update_success_rate(self: Any, action_name: str, success: bool)

Update success rate for a recovery action

##### _kill_process_on_port(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942F05EFE0>) -> RecoveryResult

Kill process using a specific port

##### _find_alternative_port(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001943035D690>) -> RecoveryResult

Find an alternative available port

##### _is_port_available(self: Any, port: int) -> bool

Check if a port is available

##### _try_alternative_ports(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001943035E6E0>) -> RecoveryResult

Try ports in safe range for permission issues

##### _check_firewall_exceptions(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001943035E350>) -> RecoveryResult

Check firewall exceptions and provide guidance

##### _install_missing_dependencies(self: Any, error: StartupError, context: <ast.Subscript object at 0x0000019433FD44C0>) -> RecoveryResult

Install missing dependencies

##### _activate_virtual_environment(self: Any, error: StartupError, context: <ast.Subscript object at 0x0000019433FD7010>) -> RecoveryResult

Activate virtual environment

##### _repair_config_file(self: Any, error: StartupError, context: <ast.Subscript object at 0x0000019431AFE6B0>) -> RecoveryResult

Repair or recreate configuration file

##### _create_virtual_environment(self: Any, error: StartupError, context: <ast.Subscript object at 0x0000019431AFCA90>) -> RecoveryResult

Create new virtual environment

##### _suggest_firewall_exception(self: Any, error: StartupError, context: <ast.Subscript object at 0x0000019431AFC3A0>) -> RecoveryResult

Provide firewall exception suggestions

##### _restart_with_different_params(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942FB37FD0>) -> RecoveryResult

Restart process with different parameters

##### _increase_timeout(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942FB36EC0>) -> RecoveryResult

Increase timeout values and retry

### FailurePattern

Represents a detected failure pattern

#### Methods

##### __init__(self: Any, pattern_id: str, description: str, frequency: int)



##### add_occurrence(self: Any, error_type: ErrorType, context: <ast.Subscript object at 0x000001942FB36800>)

Add a new occurrence of this failure pattern

##### add_recovery_result(self: Any, action_name: str, success: bool)

Record the result of a recovery attempt

##### get_success_rate_for_action(self: Any, action_name: str) -> float

Get success rate for a specific recovery action

##### get_most_successful_actions(self: Any) -> <ast.Subscript object at 0x0000019431B59C90>

Get recovery actions sorted by success rate for this pattern

### FallbackConfiguration

Manages fallback configurations when primary recovery methods fail

#### Methods

##### __init__(self: Any)



##### get_fallback_config(self: Any, error_type: ErrorType) -> <ast.Subscript object at 0x0000019431B58EE0>

Get fallback configuration for an error type

##### apply_fallback_config(self: Any, error_type: ErrorType, base_config: <ast.Subscript object at 0x0000019431B5B850>) -> <ast.Subscript object at 0x0000019431B5AD40>

Apply fallback configuration to base configuration

### IntelligentFailureHandler

Handles intelligent failure detection and learning

#### Methods

##### __init__(self: Any, recovery_engine: Any)



##### detect_failure_pattern(self: Any, error: StartupError, context: <ast.Subscript object at 0x0000019431B59390>) -> <ast.Subscript object at 0x000001943015CFD0>

Detect if this error matches an existing failure pattern

##### _create_pattern_signature(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001943015DEA0>) -> str

Create a unique signature for a failure pattern

##### _extract_error_keywords(self: Any, error_message: str) -> str

Extract key words from error message for pattern matching

##### _find_similar_pattern(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001943015F340>) -> <ast.Subscript object at 0x000001942FE354E0>

Find similar existing patterns

##### _should_create_pattern(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942FE34880>) -> bool

Determine if we should create a new failure pattern

##### _generate_pattern_description(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942FE34A00>) -> str

Generate a human-readable description for the pattern

##### prioritize_recovery_actions(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942FE34160>) -> <ast.Subscript object at 0x000001942FE371F0>

Prioritize recovery actions based on learned patterns and success rates

##### handle_recovery_failure(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942FE37D90>, failed_actions: <ast.Subscript object at 0x000001942FE37FA0>) -> RecoveryResult

Handle the case when all primary recovery actions fail

##### _get_manual_intervention_suggestions(self: Any, error_type: ErrorType) -> <ast.Subscript object at 0x000001942FDED630>

Get manual intervention suggestions for an error type

##### save_failure_patterns(self: Any)

Save learned failure patterns to persistent storage

##### load_failure_patterns(self: Any)

Load failure patterns from persistent storage

## Constants

### PORT_CONFLICT

Type: `str`

Value: `port_conflict`

### PERMISSION_DENIED

Type: `str`

Value: `permission_denied`

### DEPENDENCY_MISSING

Type: `str`

Value: `dependency_missing`

### CONFIG_INVALID

Type: `str`

Value: `config_invalid`

### PROCESS_FAILED

Type: `str`

Value: `process_failed`

### NETWORK_ERROR

Type: `str`

Value: `network_error`

### FIREWALL_BLOCKED

Type: `str`

Value: `firewall_blocked`

### VIRTUAL_ENV_ERROR

Type: `str`

Value: `virtual_env_error`

### TIMEOUT_ERROR

Type: `str`

Value: `timeout_error`

### UNKNOWN

Type: `str`

Value: `unknown`

