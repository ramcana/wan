---
title: tools.config_manager.config_validator
category: api
tags: [api, tools]
---

# tools.config_manager.config_validator

Configuration Validation System

This module provides comprehensive validation for the unified configuration,
including schema validation, dependency checking, and consistency validation.

## Classes

### ValidationSeverity

Severity levels for validation issues

### ValidationIssue

Represents a configuration validation issue

### ValidationResult

Result of configuration validation

#### Methods

##### has_errors(self: Any) -> bool

Check if there are any errors or critical issues

##### get_issues_by_severity(self: Any, severity: ValidationSeverity) -> <ast.Subscript object at 0x000001942C57C640>

Get issues filtered by severity

### ConfigurationValidator

Comprehensive configuration validation system

#### Methods

##### __init__(self: Any, schema_path: <ast.Subscript object at 0x000001942C57C430>)



##### _load_schema(self: Any) -> <ast.Subscript object at 0x000001942C52F3D0>

Load the configuration schema

##### validate_config(self: Any, config: UnifiedConfig) -> ValidationResult

Perform comprehensive validation of a configuration

Args:
    config: Configuration to validate
    
Returns:
    Validation result with any issues found

##### _validate_schema(self: Any, config_dict: <ast.Subscript object at 0x0000019428D0FF10>) -> <ast.Subscript object at 0x0000019428D0F0D0>

Validate configuration against JSON schema

##### _generate_schema_fix_suggestion(self: Any, error: jsonschema.ValidationError) -> str

Generate fix suggestion for schema validation errors

##### _validate_port_ranges(self: Any, config: UnifiedConfig) -> <ast.Subscript object at 0x0000019428D726E0>

Validate port number ranges and conflicts

##### _validate_file_paths(self: Any, config: UnifiedConfig) -> <ast.Subscript object at 0x0000019428D70910>

Validate file and directory paths

##### _validate_memory_limits(self: Any, config: UnifiedConfig) -> <ast.Subscript object at 0x0000019428D43700>

Validate memory and VRAM limits

##### _validate_timeout_values(self: Any, config: UnifiedConfig) -> <ast.Subscript object at 0x000001942B395D50>

Validate timeout and duration values

##### _validate_dependency_consistency(self: Any, config: UnifiedConfig) -> <ast.Subscript object at 0x000001942B302D10>

Validate consistency between dependent settings

##### _validate_environment_consistency(self: Any, config: UnifiedConfig) -> <ast.Subscript object at 0x000001942B2F47C0>

Validate environment-specific consistency

##### _validate_security_settings(self: Any, config: UnifiedConfig) -> <ast.Subscript object at 0x000001942B334F40>

Validate security-related settings

##### _validate_performance_settings(self: Any, config: UnifiedConfig) -> <ast.Subscript object at 0x000001942C59CD00>

Validate performance-related settings

##### validate_config_file(self: Any, config_path: <ast.Subscript object at 0x000001942C59CE50>) -> ValidationResult

Validate a configuration file

Args:
    config_path: Path to configuration file
    
Returns:
    Validation result

##### generate_validation_report(self: Any, result: ValidationResult) -> str

Generate a human-readable validation report

Args:
    result: Validation result to report on
    
Returns:
    Formatted validation report

## Constants

### JSONSCHEMA_AVAILABLE

Type: `bool`

Value: `True`

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

### JSONSCHEMA_AVAILABLE

Type: `bool`

Value: `False`

