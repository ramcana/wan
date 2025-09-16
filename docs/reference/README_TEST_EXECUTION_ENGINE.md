---
category: reference
last_updated: '2025-09-15T22:50:00.668307'
original_path: tests\utils\README_TEST_EXECUTION_ENGINE.md
tags:
- configuration
- api
- troubleshooting
- installation
- performance
title: Test Execution Engine with Timeout Handling
---

# Test Execution Engine with Timeout Handling

A comprehensive test execution engine that provides robust test running capabilities with timeout handling, retry logic, parallel execution, and resource management.

## Features

### Core Capabilities

- **Configurable Timeouts**: Different timeout settings per test category (unit, integration, e2e, performance, reliability)
- **Automatic Retry Logic**: Exponential backoff retry for flaky tests
- **Parallel Execution**: Configurable parallel test execution with resource management
- **Resource Monitoring**: CPU and memory usage monitoring with throttling
- **Flaky Test Detection**: Statistical analysis to identify intermittently failing tests
- **Comprehensive Reporting**: Detailed test results with aggregation and analysis

### Test Categories

- **Unit Tests**: Fast, isolated tests (default: 30s timeout)
- **Integration Tests**: Component interaction tests (default: 120s timeout)
- **E2E Tests**: End-to-end workflow tests (default: 300s timeout)
- **Performance Tests**: Performance benchmarks (default: 600s timeout)
- **Reliability Tests**: System reliability tests (default: 900s timeout)

## Installation

The test execution engine requires Python 3.7+ and the following dependencies:

```bash
pip install psutil pyyaml
```

## Quick Start

### Basic Usage

```python
from utils.test_execution_engine import TestExecutionEngine

# Create engine with default configuration
engine = TestExecutionEngine()

# Run all tests
result = engine.run_tests()

# Generate report
report = engine.generate_report(result, "test_report.txt")
print(report)
```

### Command Line Usage

```bash
# Run all tests
python tests/utils/test_runner_cli.py

# Run only unit tests
python tests/utils/test_runner_cli.py --category unit

# Run with custom timeout
python tests/utils/test_runner_cli.py --timeout-unit 60

# Run with more workers
python tests/utils/test_runner_cli.py --workers 8

# Generate detailed report
python tests/utils/test_runner_cli.py --output test_report.txt --verbose
```

## Configuration

### YAML Configuration

Create a configuration file (`tests/config/execution_config.yaml`):

```yaml
# Timeout settings per test category (in seconds)
timeouts:
  unit: 30
  integration: 120
  e2e: 300
  performance: 600
  reliability: 900

# Retry configuration
retry:
  max_retries: 3
  delay_base: 1.0
  delay_max: 60.0

# Parallel execution settings
parallel:
  max_workers: 4
  memory_limit_mb: 2048

# Resource management thresholds
resources:
  cpu_threshold: 0.8
  memory_threshold: 0.8

# Flaky test detection
flaky_detection:
  failure_threshold: 2
  success_rate_threshold: 0.7
```

### Programmatic Configuration

```python
from utils.test_execution_engine import TestExecutionEngine, TestConfig, TestCategory

config = TestConfig(
    timeouts={
        TestCategory.UNIT: 15,
        TestCategory.INTEGRATION: 60,
        TestCategory.E2E: 180,
    },
    max_retries=2,
    max_workers=4,
    retry_delay_base=2.0,
    cpu_threshold=0.7,
    memory_threshold=0.8
)

engine = TestExecutionEngine(config)
```

## Advanced Features

### Retry Logic with Exponential Backoff

The engine automatically retries failed tests with exponential backoff:

- First retry: 1 second delay
- Second retry: 2 second delay
- Third retry: 4 second delay
- Maximum delay: 60 seconds (configurable)

### Flaky Test Detection

Tests are marked as flaky based on:

- Failure threshold: Number of failures before considering flaky
- Success rate threshold: Minimum success rate to not be considered flaky
- Historical tracking: Maintains history of recent test results

### Resource Management

The engine monitors system resources and throttles execution when:

- CPU usage exceeds threshold (default: 80%)
- Memory usage exceeds threshold (default: 80%)

### Parallel Execution

Tests are executed in parallel with:

- Configurable worker count (default: CPU count, max 4)
- Resource-aware scheduling
- Proper cleanup and error handling

## Test Categorization

Tests are automatically categorized based on file paths and names:

| Category    | Path Patterns                            | Timeout |
| ----------- | ---------------------------------------- | ------- |
| Unit        | `*/unit/*`, `*test_unit_*`               | 30s     |
| Integration | `*/integration/*`, `*test_integration_*` | 120s    |
| E2E         | `*/e2e/*`, `*test_e2e_*`                 | 300s    |
| Performance | `*/performance/*`, `*test_performance_*` | 600s    |
| Reliability | `*/reliability/*`, `*test_reliability_*` | 900s    |

## Reporting

### Text Report

```
Test Execution Report
====================

Summary:
  Total Tests: 25
  Passed: 23
  Failed: 1
  Timeout: 1
  Error: 0
  Skipped: 0
  Success Rate: 92.0%
  Total Duration: 45.67s

Resource Usage:
  Average CPU: 45.2%
  Average Memory: 62.8%

Flaky Tests: 2
  - tests/integration/test_flaky_api.py
  - tests/unit/test_intermittent.py

Failed Tests:
  - tests/unit/test_broken.py: Test failed with return code 1

Timeout Tests:
  - tests/e2e/test_slow.py: Test timed out after 300s
```

### JSON Report

Detailed JSON report with individual test results, timing information, and resource usage data.

## CLI Options

```
usage: test_runner_cli.py [-h] [--test-dir TEST_DIR] [--category {unit,integration,e2e,performance,reliability}]
                         [--pattern PATTERN] [--file FILE] [--workers WORKERS] [--timeout-unit TIMEOUT_UNIT]
                         [--timeout-integration TIMEOUT_INTEGRATION] [--timeout-e2e TIMEOUT_E2E]
                         [--timeout-performance TIMEOUT_PERFORMANCE] [--timeout-reliability TIMEOUT_RELIABILITY]
                         [--max-retries MAX_RETRIES] [--no-retry] [--cpu-threshold CPU_THRESHOLD]
                         [--memory-threshold MEMORY_THRESHOLD] [--output OUTPUT] [--config CONFIG]
                         [--verbose] [--quiet] [--json-output JSON_OUTPUT] [--dry-run] [--list-categories]

Test Execution Engine with Timeout Handling and Retry Logic

optional arguments:
  -h, --help            show this help message and exit
  --test-dir TEST_DIR   Test directory to scan (default: tests)
  --category {unit,integration,e2e,performance,reliability}
                        Run tests from specific categories (can be used multiple times)
  --pattern PATTERN     Run tests matching pattern (can be used multiple times)
  --file FILE           Run specific test files (can be used multiple times)
  --workers WORKERS     Maximum parallel workers
  --timeout-unit TIMEOUT_UNIT
                        Timeout for unit tests (seconds)
  --timeout-integration TIMEOUT_INTEGRATION
                        Timeout for integration tests (seconds)
  --timeout-e2e TIMEOUT_E2E
                        Timeout for e2e tests (seconds)
  --timeout-performance TIMEOUT_PERFORMANCE
                        Timeout for performance tests (seconds)
  --timeout-reliability TIMEOUT_RELIABILITY
                        Timeout for reliability tests (seconds)
  --max-retries MAX_RETRIES
                        Maximum retry attempts for failed tests
  --no-retry            Disable retry logic
  --cpu-threshold CPU_THRESHOLD
                        CPU usage threshold for throttling (0.0-1.0)
  --memory-threshold MEMORY_THRESHOLD
                        Memory usage threshold for throttling (0.0-1.0)
  --output OUTPUT       Output file for test report
  --config CONFIG       Configuration file path
  --verbose, -v         Verbose logging
  --quiet, -q           Quiet mode (minimal output)
  --json-output JSON_OUTPUT
                        Output results as JSON to specified file
  --dry-run             Show what tests would be run without executing them
  --list-categories     List available test categories and exit
```

## Examples

### Run Specific Test Categories

```bash
# Run only unit tests
python tests/utils/test_runner_cli.py --category unit

# Run unit and integration tests
python tests/utils/test_runner_cli.py --category unit --category integration
```

### Custom Timeouts

```bash
# Increase timeout for integration tests
python tests/utils/test_runner_cli.py --timeout-integration 300

# Set custom timeouts for all categories
python tests/utils/test_runner_cli.py \
  --timeout-unit 45 \
  --timeout-integration 180 \
  --timeout-e2e 600
```

### Pattern Matching

```bash
# Run tests matching pattern
python tests/utils/test_runner_cli.py --pattern "*api*"

# Run multiple patterns
python tests/utils/test_runner_cli.py --pattern "*unit*" --pattern "*integration*"
```

### Resource Management

```bash
# Limit parallel workers
python tests/utils/test_runner_cli.py --workers 2

# Adjust resource thresholds
python tests/utils/test_runner_cli.py --cpu-threshold 0.6 --memory-threshold 0.7
```

### Reporting

```bash
# Generate detailed report
python tests/utils/test_runner_cli.py --output test_report.txt --verbose

# Generate JSON report
python tests/utils/test_runner_cli.py --json-output results.json

# Quiet mode with JSON output
python tests/utils/test_runner_cli.py --quiet --json-output results.json
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install psutil pyyaml
      - name: Run tests
        run: |
          python tests/utils/test_runner_cli.py \
            --json-output test_results.json \
            --output test_report.txt \
            --verbose
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: |
            test_results.json
            test_report.txt
```

### Jenkins

```groovy
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh '''
                    python tests/utils/test_runner_cli.py \
                        --json-output test_results.json \
                        --output test_report.txt \
                        --workers 4
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'test_*.txt,test_*.json'
                    publishTestResults testResultsPattern: 'test_results.json'
                }
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Tests timing out**: Increase timeout for the appropriate category
2. **High resource usage**: Reduce worker count or adjust thresholds
3. **Flaky tests**: Check test isolation and external dependencies
4. **Import errors**: Ensure test paths are correctly configured

### Debug Mode

```bash
# Run with verbose logging
python tests/utils/test_runner_cli.py --verbose

# Dry run to see what would be executed
python tests/utils/test_runner_cli.py --dry-run
```

### Performance Tuning

- Adjust worker count based on system capabilities
- Set appropriate resource thresholds
- Use test categorization to optimize timeouts
- Monitor resource usage in reports

## Contributing

When contributing to the test execution engine:

1. Add tests for new features
2. Update configuration schema if needed
3. Update documentation
4. Ensure backward compatibility
5. Test with various Python versions

## License

This test execution engine is part of the WAN22 project and follows the same license terms.
