# WAN2.2 Local Installation Test Suite

This directory contains the comprehensive automated testing framework for the WAN2.2 local installation system. The test suite includes unit tests, integration tests, and hardware simulation tests to ensure the installation system works correctly across various hardware configurations.

## Overview

The test framework is designed to:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test cross-component functionality and workflows
- **Hardware Simulation Tests**: Test various hardware configurations and optimizations
- **Automated Reporting**: Generate detailed test reports in JSON and HTML formats

## Test Structure

```
tests/
├── __init__.py                     # Package initialization
├── test_runner.py                  # Main automated test framework
├── unit_tests.py                   # Unit tests for individual components
├── integration_tests.py            # Integration tests for workflows
├── hardware_simulation_tests.py    # Hardware configuration simulation tests
├── test_config.py                  # Test configuration and utilities
└── README.md                       # This documentation
```

## Running Tests

### Quick Start

Run all tests with default settings:

```bash
# Windows
run_tests.bat

# Python directly
python run_automated_tests.py
```

### Test Suite Options

Run specific test suites:

```bash
# Unit tests only
python run_automated_tests.py --suite unit

# Integration tests only
python run_automated_tests.py --suite integration

# Hardware simulation tests only
python run_automated_tests.py --suite hardware

# All tests (default)
python run_automated_tests.py --suite all
```

### Additional Options

```bash
# Verbose output
python run_automated_tests.py --verbose

# Generate HTML report
python run_automated_tests.py --html-report

# Don't cleanup test files (for debugging)
python run_automated_tests.py --no-cleanup

# Custom output directory
python run_automated_tests.py --output-dir my_test_results

# Custom timeout (in seconds)
python run_automated_tests.py --timeout 3600
```

### Batch File Options

The Windows batch file supports simplified options:

```batch
run_tests.bat --unit --verbose      # Unit tests with verbose output
run_tests.bat --integration --html  # Integration tests with HTML report
run_tests.bat --hardware            # Hardware simulation tests only
run_tests.bat --help                # Show help
```

## Test Categories

### Unit Tests

Test individual components in isolation:

- **System Detection**: CPU, memory, GPU, storage, and OS detection
- **Dependency Management**: Python installation, virtual environment creation, package installation
- **Model Management**: Model downloading, validation, and configuration
- **Configuration Engine**: Hardware-aware configuration generation and validation
- **Validation Framework**: Installation validation and functionality testing
- **Error Handling**: Error categorization, recovery suggestions, and logging

### Integration Tests

Test cross-component functionality:

- **Detection to Configuration**: Hardware detection flowing into configuration generation
- **Dependency to Validation**: Dependency installation validation
- **Model to Configuration**: Model requirements affecting configuration optimization
- **Full Installation Flow**: Complete installation workflow simulation
- **Error Recovery**: Error handling and rollback integration

### Hardware Simulation Tests

Test various hardware configurations:

#### High-End Systems

- **Threadripper PRO 5995WX + RTX 4080**: 64C/128T, 128GB RAM, 16GB VRAM
- **Ryzen 9 5950X + RTX 4070**: 16C/32T, 64GB RAM, 12GB VRAM

#### Mid-Range Systems

- **Ryzen 7 5800X + RTX 3070**: 8C/16T, 32GB RAM, 8GB VRAM
- **Intel i7-12700K + RTX 3060**: 12C/20T, 32GB RAM, 12GB VRAM

#### Budget Systems

- **Ryzen 5 3600 + GTX 1660 Ti**: 6C/12T, 16GB RAM, 6GB VRAM
- **Intel i5-8400 + GTX 1060**: 6C/6T, 8GB RAM, 6GB VRAM

#### Special Configurations

- **APU System**: Ryzen 7 5700G (no dedicated GPU)
- **Legacy System**: Intel i5-4590 + GTX 970 (older hardware)

## Test Configuration

The test framework uses configuration in `test_config.py`:

```python
TEST_CONFIG = {
    "timeout": {
        "unit_test": 30,        # 30 seconds per unit test
        "integration_test": 120, # 2 minutes per integration test
        "hardware_test": 60,     # 1 minute per hardware test
        "total_suite": 1800      # 30 minutes total timeout
    },
    "hardware_profiles": {
        "test_high_end": True,
        "test_mid_range": True,
        "test_budget": True,
        "test_minimum": True,
        "test_no_gpu": True,
        "test_legacy": False     # May fail minimum requirements
    },
    "mock_downloads": True,      # Mock model downloads
    "mock_subprocess": True,     # Mock subprocess calls
    "create_temp_files": True,   # Create temporary test files
    "cleanup_after_tests": True,
    "verbose_output": False,
    "save_test_artifacts": True
}
```

## Test Reports

The framework generates comprehensive test reports:

### JSON Report (`test_report.json`)

```json
{
  "timestamp": "2024-01-15 14:30:25",
  "total_duration": 245.67,
  "overall_stats": {
    "total_tests": 45,
    "total_passed": 43,
    "total_failed": 2,
    "success_rate": 0.956
  },
  "suite_results": {
    "unit_tests": { ... },
    "integration_tests": { ... },
    "hardware_simulation_tests": { ... }
  },
  "summary": {
    "status": "PASSED",
    "critical_failures": [],
    "recommendations": []
  }
}
```

### HTML Report (`test_report.html`)

Interactive HTML report with:

- Test summary dashboard
- Suite-by-suite breakdown
- Individual test results
- Error details and recommendations

## Mock Data and Simulation

The test framework creates realistic mock data:

### Mock Model Structure

- Creates model directories for WAN2.2-T2V-A14B, WAN2.2-I2V-A14B, WAN2.2-TI2V-5B
- Generates mock model files (pytorch_model.bin, config.json, tokenizer.json)
- Simulates appropriate file sizes and metadata

### Mock Virtual Environment

- Creates Scripts/ directory with python.exe, pip.exe
- Generates site-packages/ with mock installed packages
- Simulates package metadata and version information

### Mock System Responses

- Mocks subprocess calls (nvidia-smi, python version checks)
- Simulates network requests for model downloads
- Provides realistic hardware detection responses

## Debugging Tests

### Verbose Output

Use `--verbose` flag to see detailed test execution:

```bash
python run_automated_tests.py --verbose
```

### Preserve Test Files

Use `--no-cleanup` to keep temporary test files for inspection:

```bash
python run_automated_tests.py --no-cleanup
```

### Individual Test Execution

Run specific test modules directly:

```bash
python -m unittest tests.unit_tests.TestSystemDetection.test_cpu_detection
python -m unittest tests.integration_tests.TestDetectionToConfigurationIntegration
python -m unittest tests.hardware_simulation_tests.TestHighEndHardwareConfigurations
```

## Exit Codes

The test framework uses specific exit codes:

- **0**: All tests passed (≥90% success rate)
- **1**: Some tests failed but core functionality works (≥70% success rate)
- **2**: Many tests failed, review required (<70% success rate)
- **130**: Test execution interrupted by user
- **3**: Test framework error

## Hardware Requirements for Testing

The test framework itself has minimal requirements:

- Python 3.9+
- ~100MB temporary disk space
- ~50MB RAM during execution

The framework simulates various hardware configurations without requiring the actual hardware.

## Continuous Integration

The test framework is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run WAN2.2 Installation Tests
  run: |
    cd local_installation
    python run_automated_tests.py --html-report --output-dir ci_results

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: local_installation/ci_results/
```

## Extending the Test Suite

### Adding New Unit Tests

1. Add test methods to appropriate classes in `unit_tests.py`
2. Follow naming convention: `test_<component>_<functionality>`
3. Use appropriate assertions and mock objects

### Adding New Integration Tests

1. Create test methods in `integration_tests.py`
2. Test cross-component workflows
3. Use realistic data flows and error scenarios

### Adding New Hardware Profiles

1. Add hardware profiles to `HardwareProfileGenerator` in `hardware_simulation_tests.py`
2. Create corresponding test methods
3. Verify appropriate optimizations are applied

### Custom Test Configuration

1. Modify `TEST_CONFIG` in `test_config.py`
2. Add new mock data generators as needed
3. Update report generation for new test types

## Troubleshooting

### Common Issues

**Import Errors**

- Ensure you're running from the `local_installation` directory
- Check that all required modules are in the `scripts/` directory

**Mock Failures**

- Verify mock data is being created correctly
- Check that patches are applied to the correct module paths

**Timeout Issues**

- Increase timeout values in `TEST_CONFIG`
- Use `--timeout` parameter for longer test runs

**Permission Errors**

- Ensure write permissions for temporary directories
- Run with appropriate privileges if needed

### Getting Help

1. Run with `--verbose` for detailed output
2. Check test reports for specific error messages
3. Review individual test modules for expected behavior
4. Use `--no-cleanup` to inspect temporary test files

## Performance Considerations

The test suite is optimized for speed:

- Uses mocking to avoid actual downloads and installations
- Creates minimal test data structures
- Runs tests in parallel where possible
- Provides configurable timeouts

Typical execution times:

- Unit tests: ~30-60 seconds
- Integration tests: ~60-120 seconds
- Hardware simulation tests: ~60-90 seconds
- Complete suite: ~3-5 minutes
