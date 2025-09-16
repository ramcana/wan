---
title: tools.dev-environment.setup_dev_environment
category: api
tags: [api, tools]
---

# tools.dev-environment.setup_dev_environment

Automated Development Environment Setup

This module provides automated setup for the WAN22 development environment,
including dependency installation, configuration, and validation.

## Classes

### DevEnvironmentSetup

Automated development environment setup

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x000001942CE1FA60>, verbose: bool)



##### setup_python_environment(self: Any) -> bool

Setup Python environment

##### setup_nodejs_environment(self: Any) -> bool

Setup Node.js environment

##### setup_development_tools(self: Any) -> bool

Setup development tools

##### create_project_structure(self: Any) -> bool

Create missing project directories

##### setup_configuration_files(self: Any) -> bool

Setup configuration files

##### validate_setup(self: Any) -> bool

Validate the setup

##### run_full_setup(self: Any) -> bool

Run complete development environment setup

##### generate_setup_report(self: Any, output_file: Path) -> None

Generate setup report

