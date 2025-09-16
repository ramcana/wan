---
title: tools.dev-environment.dependency_detector
category: api
tags: [api, tools]
---

# tools.dev-environment.dependency_detector

Dependency Detection and Installation Guidance

This module provides automated dependency detection and installation guidance
for the WAN22 development environment.

## Classes

### DependencyInfo

Information about a dependency

### SystemInfo

System information

### DependencyDetector

Detects and validates development dependencies

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x0000019427BBAB30>)



##### _get_system_info(self: Any) -> SystemInfo

Get system information

##### detect_python_dependencies(self: Any) -> <ast.Subscript object at 0x0000019427B87760>

Detect Python dependencies

##### detect_nodejs_dependencies(self: Any) -> <ast.Subscript object at 0x0000019427B84E50>

Detect Node.js dependencies

##### detect_system_dependencies(self: Any) -> <ast.Subscript object at 0x0000019427B85870>

Detect system-level dependencies

##### _check_python_package(self: Any, package_name: str, requirement: str) -> DependencyInfo

Check if a Python package is installed

##### get_all_dependencies(self: Any) -> <ast.Subscript object at 0x0000019427B2FCD0>

Get all dependencies categorized by type

##### get_missing_dependencies(self: Any) -> <ast.Subscript object at 0x0000019427B2FA00>

Get only missing dependencies

##### generate_installation_guide(self: Any) -> str

Generate installation guide for missing dependencies

##### export_dependency_report(self: Any, output_file: Path) -> None

Export dependency report to JSON file

