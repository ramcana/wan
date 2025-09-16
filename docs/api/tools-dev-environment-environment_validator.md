---
title: tools.dev-environment.environment_validator
category: api
tags: [api, tools]
---

# tools.dev-environment.environment_validator

Development Environment Validator

This module provides comprehensive validation and health checking
for the WAN22 development environment.

## Classes

### ValidationResult

Result of a validation check

### EnvironmentHealth

Overall environment health status

### EnvironmentValidator

Validates development environment setup and health

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x000001942A1BE9E0>)



##### validate_python_environment(self: Any) -> <ast.Subscript object at 0x00000194274A4AF0>

Validate Python environment

##### validate_nodejs_environment(self: Any) -> <ast.Subscript object at 0x0000019428DAC670>

Validate Node.js environment

##### validate_project_structure(self: Any) -> <ast.Subscript object at 0x0000019428DACB80>

Validate project structure

##### validate_development_tools(self: Any) -> <ast.Subscript object at 0x000001942A2E6D40>

Validate development tools

##### validate_ports_and_services(self: Any) -> <ast.Subscript object at 0x000001942A2E74C0>

Validate ports and services

##### validate_gpu_environment(self: Any) -> <ast.Subscript object at 0x000001942A2E4FA0>

Validate GPU environment (optional)

##### run_full_validation(self: Any) -> EnvironmentHealth

Run complete environment validation

##### _check_python_requirements(self: Any, requirements_file: Path) -> <ast.Subscript object at 0x000001942CB353C0>

Check which Python packages are missing

##### _is_port_available(self: Any, port: int) -> bool

Check if a port is available

##### export_health_report(self: Any, health: EnvironmentHealth, output_file: Path) -> None

Export health report to JSON file

