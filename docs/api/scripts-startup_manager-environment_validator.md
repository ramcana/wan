---
title: scripts.startup_manager.environment_validator
category: api
tags: [api, scripts]
---

# scripts.startup_manager.environment_validator

Environment Validator for WAN22 Startup Manager.
Validates system requirements, dependencies, and configurations before server startup.

## Classes

### ValidationStatus

Status of validation checks.

### ValidationIssue

Represents a validation issue found during environment checking.

### ValidationResult

Result of environment validation.

### DependencyValidator

Validates Python and Node.js dependencies and environments.

#### Methods

##### __init__(self: Any)



##### validate_python_environment(self: Any) -> <ast.Subscript object at 0x00000194344D50C0>

Validate Python version, virtual environment, and dependencies.

##### validate_node_environment(self: Any) -> <ast.Subscript object at 0x00000194344D6C50>

Validate Node.js and npm versions and frontend dependencies.

##### _check_python_version(self: Any) -> <ast.Subscript object at 0x00000194344D77F0>

Check if Python version meets minimum requirements.

##### _check_virtual_environment(self: Any) -> <ast.Subscript object at 0x00000194344D6D70>

Check if running in a virtual environment.

##### _check_backend_dependencies(self: Any) -> <ast.Subscript object at 0x000001942FBCF1F0>

Check if backend dependencies are installed.

##### _check_node_version(self: Any) -> <ast.Subscript object at 0x000001942FBCD7E0>

Check Node.js installation and version.

##### _check_npm_version(self: Any) -> <ast.Subscript object at 0x00000194344DB490>

Check npm installation and version.

##### _check_frontend_dependencies(self: Any) -> <ast.Subscript object at 0x00000194344D8760>

Check if frontend dependencies are installed.

##### _is_package_installed(self: Any, package_name: str) -> bool

Check if a Python package is installed.

### ConfigurationValidator

Validates and repairs configuration files.

#### Methods

##### __init__(self: Any)



##### validate_backend_config(self: Any) -> <ast.Subscript object at 0x0000019434447340>

Validate backend config.json file.

##### validate_frontend_config(self: Any) -> <ast.Subscript object at 0x000001942EFF9D80>

Validate frontend configuration files.

##### _validate_package_json(self: Any) -> <ast.Subscript object at 0x000001942EFFB220>

Validate frontend package.json file.

##### _validate_vite_config(self: Any) -> <ast.Subscript object at 0x000001943455EE90>

Validate frontend vite.config.ts file.

##### _validate_backend_config_values(self: Any, config_data: <ast.Subscript object at 0x000001943455D3F0>) -> <ast.Subscript object at 0x000001943455EB00>

Validate specific backend configuration values.

##### auto_repair_config(self: Any, issues: <ast.Subscript object at 0x000001943455ED40>) -> <ast.Subscript object at 0x00000194344D23E0>

Attempt to automatically repair configuration issues.

##### _create_default_backend_config(self: Any)

Create a default backend configuration file.

##### _create_default_package_json(self: Any)

Create a default frontend package.json file.

##### _create_default_vite_config(self: Any)

Create a default vite.config.ts file.

##### _add_missing_backend_config_fields(self: Any, missing_fields: <ast.Subscript object at 0x00000194344D34C0>)

Add missing fields to backend config.json.

##### _fix_device_setting(self: Any)

Fix invalid device setting in backend config.

##### _fix_resolution_settings(self: Any)

Fix invalid resolution settings in backend config.

### EnvironmentValidator

Main environment validator that orchestrates all validation checks.

#### Methods

##### __init__(self: Any)



##### validate_all(self: Any) -> ValidationResult

Run all environment validation checks.

##### auto_fix_issues(self: Any, issues: <ast.Subscript object at 0x000001942EF971F0>) -> <ast.Subscript object at 0x000001943032B640>

Attempt to automatically fix issues that are marked as auto-fixable.

##### _collect_system_info(self: Any) -> <ast.Subscript object at 0x0000019430307AF0>

Collect system information for debugging and logging.

## Constants

### PASSED

Type: `str`

Value: `passed`

### FAILED

Type: `str`

Value: `failed`

### WARNING

Type: `str`

Value: `warning`

### FIXED

Type: `str`

Value: `fixed`

