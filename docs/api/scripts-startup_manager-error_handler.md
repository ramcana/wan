---
title: scripts.startup_manager.error_handler
category: api
tags: [api, scripts]
---

# scripts.startup_manager.error_handler

Error handling and user guidance system for the startup manager.

This module provides:
- Structured error display with clear messages
- Interactive error resolution
- Context-sensitive help and troubleshooting
- Error classification and recovery suggestions

## Classes

### ErrorSeverity

Error severity levels

### ErrorCategory

Categories of startup errors

### RecoveryAction

Represents a recovery action that can be taken

### StartupError

Structured representation of a startup error

#### Methods

##### __post_init__(self: Any)

Post-initialization to set default recovery actions

##### _get_default_recovery_actions(self: Any) -> <ast.Subscript object at 0x0000019429CD8F10>

Get default recovery actions based on error category

##### _open_firewall_settings(self: Any) -> bool

Open Windows Firewall settings

##### _suggest_admin_restart(self: Any) -> bool

Suggest restarting with admin privileges

##### _check_permissions(self: Any) -> bool

Check file and folder permissions

##### _install_dependencies(self: Any) -> bool

Install missing dependencies

##### _view_logs(self: Any) -> bool

View detailed logs

##### _show_help(self: Any) -> bool

Show help documentation

### ErrorClassifier

Classifies errors and creates structured error objects

#### Methods

##### __init__(self: Any)



##### classify_error(self: Any, error: Exception, context: <ast.Subscript object at 0x000001942808C160>) -> StartupError

Classify an error and create a structured error object

##### _determine_severity(self: Any, error: Exception, category: ErrorCategory) -> ErrorSeverity

Determine error severity based on type and category

##### _extract_details(self: Any, error: Exception, context: <ast.Subscript object at 0x000001942C6DB250>) -> str

Extract user-friendly details from error and context

##### _format_technical_details(self: Any, error: Exception) -> str

Format technical details for debugging

##### _generate_error_code(self: Any, category: ErrorCategory, error_type: str) -> str

Generate a unique error code

##### _get_context_specific_actions(self: Any, error: StartupError, context: <ast.Subscript object at 0x000001942886D330>) -> <ast.Subscript object at 0x000001942886E4A0>

Get recovery actions specific to the context

##### _suggest_port_change(self: Any, current_port: int) -> bool

Suggest using a different port

##### _reset_config_file(self: Any, config_file: str) -> bool

Reset a configuration file to defaults

### ErrorDisplayManager

Manages the display of errors and user interaction

#### Methods

##### __init__(self: Any, cli: InteractiveCLI)



##### handle_error(self: Any, error: Exception, context: <ast.Subscript object at 0x000001942886ED40>) -> bool

Handle an error with full user interaction

##### display_error(self: Any, error: StartupError)

Display a structured error with rich formatting

##### display_error_enhanced(self: Any, error: StartupError)

Display error with enhanced formatting and context

##### _get_category_guidance(self: Any, category: ErrorCategory) -> str

Get quick guidance based on error category

##### _is_recurring_error(self: Any, error: StartupError) -> bool

Check if this is a recurring error pattern

##### _display_recurring_error_warning(self: Any, error: StartupError)

Display warning about recurring errors

##### offer_recovery_options(self: Any, error: StartupError) -> bool

Offer recovery options to the user

##### offer_recovery_options_enhanced(self: Any, error: StartupError) -> bool

Enhanced recovery options with better user interaction

##### _display_no_recovery_options(self: Any, error: StartupError)

Display message when no recovery options are available

##### _display_recovery_options_table(self: Any, actions: <ast.Subscript object at 0x0000019427EFD630>)

Display recovery options in a formatted table

##### _offer_auto_recovery(self: Any, auto_actions: <ast.Subscript object at 0x0000019427EFFFD0>) -> bool

Offer to run all automatic actions

##### _interactive_recovery_selection(self: Any, error: StartupError) -> bool

Interactive recovery action selection with enhanced UX

##### _execute_single_action(self: Any, action: RecoveryAction) -> bool

Execute a single recovery action with enhanced feedback

##### _execute_all_actions(self: Any, actions: <ast.Subscript object at 0x00000194288B0280>) -> bool

Execute all recovery actions in sequence

##### _display_batch_recovery_summary(self: Any, success_count: int, total_count: int, failed_actions: <ast.Subscript object at 0x0000019428908610>)

Display summary of batch recovery execution

##### show_contextual_help(self: Any, error: StartupError)

Show contextual help for the error

##### show_enhanced_help(self: Any, error: StartupError)

Show enhanced contextual help with interactive options

##### show_detailed_troubleshooting(self: Any, error: StartupError)

Show detailed troubleshooting steps

##### show_error_examples(self: Any, error: StartupError)

Show examples of similar errors and solutions

##### show_log_guidance(self: Any, error: StartupError)

Show guidance on checking logs

##### show_support_information(self: Any, error: StartupError)

Show support and reporting information

##### _get_windows_version(self: Any) -> str

Get Windows version information

##### show_general_troubleshooting(self: Any)

Show general troubleshooting tips

### HelpSystem

Provides context-sensitive help and troubleshooting guidance

#### Methods

##### __init__(self: Any)



##### get_help_for_error(self: Any, error: StartupError) -> <ast.Subscript object at 0x000001942B744B50>

Get help content for a specific error

##### get_troubleshooting_steps(self: Any, category: ErrorCategory) -> <ast.Subscript object at 0x000001942B746080>

Get detailed troubleshooting steps for an error category

##### get_error_examples(self: Any, category: ErrorCategory) -> <ast.Subscript object at 0x000001942A1622F0>

Get examples of common errors in a category

##### _get_environment_help(self: Any, error: StartupError) -> str

Get help for environment-related errors

##### _get_network_help(self: Any, error: StartupError) -> str

Get help for network-related errors

##### _get_permission_help(self: Any, error: StartupError) -> str

Get help for permission-related errors

##### _get_dependency_help(self: Any, error: StartupError) -> str

Get help for dependency-related errors

##### _get_configuration_help(self: Any, error: StartupError) -> str

Get help for configuration-related errors

##### _get_process_help(self: Any, error: StartupError) -> str

Get help for process-related errors

##### _get_system_help(self: Any, error: StartupError) -> str

Get help for system-related errors

##### _get_environment_troubleshooting(self: Any) -> str

Get detailed environment troubleshooting steps

##### _get_network_troubleshooting(self: Any) -> str

Get detailed network troubleshooting steps

##### _get_permission_troubleshooting(self: Any) -> str

Get detailed permission troubleshooting steps

##### _get_dependency_troubleshooting(self: Any) -> str

Get detailed dependency troubleshooting steps

##### _get_configuration_troubleshooting(self: Any) -> str

Get detailed configuration troubleshooting steps

##### _get_process_troubleshooting(self: Any) -> str

Get detailed process troubleshooting steps

##### _get_system_troubleshooting(self: Any) -> str

Get detailed system troubleshooting steps

##### _get_environment_examples(self: Any) -> str

Get examples of common environment errors

##### _get_network_examples(self: Any) -> str

Get examples of common network errors

##### _get_permission_examples(self: Any) -> str

Get examples of common permission errors

##### _get_dependency_examples(self: Any) -> str

Get examples of common dependency errors

##### _get_configuration_examples(self: Any) -> str

Get examples of common configuration errors

##### _get_process_examples(self: Any) -> str

Get examples of common process errors

##### _get_system_examples(self: Any) -> str

Get examples of common system errors

## Constants

### INFO

Type: `str`

Value: `info`

### WARNING

Type: `str`

Value: `warning`

### ERROR

Type: `str`

Value: `error`

### CRITICAL

Type: `str`

Value: `critical`

### ENVIRONMENT

Type: `str`

Value: `environment`

### CONFIGURATION

Type: `str`

Value: `configuration`

### NETWORK

Type: `str`

Value: `network`

### PERMISSION

Type: `str`

Value: `permission`

### DEPENDENCY

Type: `str`

Value: `dependency`

### PROCESS

Type: `str`

Value: `process`

### SYSTEM

Type: `str`

Value: `system`

### UNKNOWN

Type: `str`

Value: `unknown`

