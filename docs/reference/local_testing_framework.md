---
category: reference
last_updated: '2025-09-15T22:50:00.843973'
original_path: tools\project-structure-analyzer\example-analysis-output\documentation\local_testing_framework.md
tags:
- configuration
- api
- troubleshooting
- installation
- performance
title: Local Testing Framework
---

# Local Testing Framework

The Local Testing Framework is a specialized testing system designed to validate the WAN22 project's functionality in local development environments.

## Purpose

The Local Testing Framework serves several key purposes:

1. **Local Validation**: Tests the complete system in a local development environment
2. **Integration Testing**: Validates interactions between different components
3. **Performance Testing**: Measures system performance under various conditions
4. **Edge Case Testing**: Tests unusual or boundary conditions
5. **Regression Testing**: Ensures new changes don't break existing functionality

## Relationship to Main Application

The Local Testing Framework is **separate from but complementary to** the main WAN22 application:

- **Main Application** (`backend/`, `frontend/`, `core/`): The production video generation system
- **Local Testing Framework** (`local_testing_framework/`): Testing and validation tools

The framework **tests** the main application but is **not part** of the production deployment.

## Architecture

The framework is organized as a Python package with 12 files:

- `cli/`: Command-line interface
- `config_templates/`: Module directory
- `continuous_monitor.py`: Monitoring functionality
- `diagnostic_tool.py`: Diagnostic tools
- `docs/`: Documentation
- `edge_case_samples/`: Edge case test data
- `environment_validator.py`: Validation logic
- `examples/`: Example code
- `integration_tester.py`: Test runner
- `integration_test_report_generator.py`: Test runner
- `models/`: Data models
- `performance_tester.py`: Test runner
- `production_validator.py`: Validation logic
- `report_generator.py`: Report generation
- `sample_manager.py`: Management utilities
- `tests/`: Test files
- `test_manager.py`: Test runner
- `test_samples/`: Test data samples
- `__init__.py`: Utility module
- `__main__.py`: Main entry point
- `__pycache__/`: Module directory

## Usage

### Running Tests

```bash
# Run all tests
python -m local_testing_framework

# Run specific test type
python -m local_testing_framework --integration
python -m local_testing_framework --performance
```

### Configuration

The framework uses configuration files in `local_testing_framework/config_templates/` to define:

- Test parameters and thresholds
- Model configurations for testing
- Environment-specific settings

### Test Samples

Test data is stored in `local_testing_framework/test_samples/` and includes:

- Sample input prompts
- Expected output formats
- Edge case scenarios

## Integration with Main System

The Local Testing Framework integrates with the main application through:

### API Testing
- Tests backend API endpoints
- Validates request/response formats
- Checks error handling

### Model Testing
- Loads and tests AI models
- Validates generation quality
- Tests model switching

### Configuration Testing
- Validates configuration files
- Tests environment setup
- Checks dependency resolution

### Performance Monitoring
- Measures generation times
- Monitors resource usage
- Tracks system health
