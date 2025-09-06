# Development Environment Automation

This directory contains tools for automated development environment setup, dependency management, and validation.

## Components

- `setup_dev_environment.py` - Main automated setup script
- `dependency_detector.py` - Dependency detection and installation guidance
- `environment_validator.py` - Development environment validation and health checking
- `dev_environment_cli.py` - Command-line interface for development environment management

## Usage

### Quick Setup

```bash
# Automated development environment setup
python tools/dev-environment/setup_dev_environment.py

# With verbose output
python tools/dev-environment/setup_dev_environment.py --verbose

# Validate existing environment
python tools/dev-environment/environment_validator.py --validate

# Check dependencies
python tools/dev-environment/dependency_detector.py --check
```

### CLI Interface

```bash
# Interactive setup
python tools/dev-environment/dev_environment_cli.py setup

# Validate environment
python tools/dev-environment/dev_environment_cli.py validate

# Check dependencies
python tools/dev-environment/dev_environment_cli.py deps

# Health check
python tools/dev-environment/dev_environment_cli.py health
```

## Features

- Automated Python and Node.js environment setup
- Dependency detection and installation guidance
- Environment validation and health checking
- Cross-platform support (Windows, macOS, Linux)
- Integration with existing project structure
- Detailed logging and error reporting
