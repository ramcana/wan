# Test Suite Infrastructure and Orchestration

A comprehensive test management system that provides orchestrated test execution, coverage analysis, and test health monitoring for the WAN22 project.

## Overview

This system addresses the critical need for reliable test infrastructure by providing:

- **Test Orchestration**: Coordinated execution of test suites with category management
- **Test Runner Engine**: Robust test execution with timeout handling and progress monitoring
- **Coverage Analysis**: Comprehensive code coverage measurement and reporting
- **Test Auditing**: Automated identification and fixing of broken or miscategorized tests

## Components

### 1. Test Suite Orchestrator (`orchestrator.py`)

The main orchestrator coordinates test execution across different categories with parallel execution support.

**Key Features:**

- Category-based test organization (unit, integration, performance, e2e)
- Parallel and sequential execution modes
- Resource management and timeout handling
- Comprehensive result aggregation and reporting

**Usage:**

```python
from orchestrator import TestSuiteOrchestrator, TestCategory

orchestrator = TestSuiteOrchestrator()
results = await orchestrator.run_full_suite(
    categories=[TestCategory.UNIT, TestCategory.INTEGRATION],
    parallel=True
)
```

### 2. Test Runner Engine (`runner_engine.py`)

Handles the actual execution of tests with advanced monitoring and timeout management.

**Key Features:**

- Automatic test discovery and categorization
- Timeout handling with graceful process termination
- Progress monitoring with real-time callbacks
- Support for multiple test frameworks (pytest, unittest)

**Usage:**

```python
from runner_engine import TestRunnerEngine, TestExecutionContext

engine = TestRunnerEngine(config)
context = TestExecutionContext(
    category=TestCategory.UNIT,
    timeout=60,
    parallel=True
)
results = await engine.execute_category_tests(context)
```

### 3. Coverage Analyzer (`coverage_analyzer.py`)

Provides comprehensive code coverage measurement and analysis.

**Key Features:**

- Multi-format coverage reporting (HTML, JSON, Markdown)
- Threshold validation and enforcement
- Historical trend analysis
- Actionable recommendations for improvement

**Usage:**

```python
from coverage_analyzer import CoverageAnalyzer

analyzer = CoverageAnalyzer(config)
report = analyzer.measure_coverage([TestCategory.UNIT])
analyzer.generate_coverage_report(report, "coverage.html", "html")
```

### 4. Test Auditor (`test_auditor.py`)

Identifies and fixes issues in existing test files.

**Key Features:**

- Automated test file analysis
- Issue detection (broken tests, missing imports, syntax errors)
- Automatic categorization suggestions
- Fix generation and application

**Usage:**

```python
from test_auditor import TestAuditor

auditor = TestAuditor(config)
report = auditor.audit_all_tests()
auditor.fix_broken_tests(report, auto_fix=True)
```

## Configuration

Tests are configured via `tests/config/test-config.yaml`:

```yaml
test_categories:
  unit:
    timeout: 30
    parallel: true
    coverage_threshold: 70
    patterns:
      - "tests/unit/test_*.py"

  integration:
    timeout: 120
    parallel: false
    requires_services: ["backend"]
    patterns:
      - "tests/integration/test_*_integration.py"

coverage:
  minimum_threshold: 70
  exclude_patterns:
    - "*/tests/*"
    - "*/__pycache__/*"

parallel_execution:
  max_workers: 4
  categories_parallel: ["unit"]
  categories_sequential: ["integration", "performance", "e2e"]
```

## Directory Structure

```
tests/
├── config/
│   ├── test-config.yaml      # Main test configuration
│   ├── conftest.py          # Pytest configuration
│   └── pytest.ini          # Pytest settings
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── performance/             # Performance benchmarks
├── e2e/                    # End-to-end tests
├── fixtures/               # Test data and fixtures
└── utils/                  # Test utilities

tools/test-runner/
├── orchestrator.py         # Main orchestration
├── runner_engine.py        # Test execution engine
├── coverage_analyzer.py    # Coverage analysis
├── test_auditor.py        # Test auditing
├── run_test_audit.py      # Audit runner script
└── example_usage.py       # Usage examples
```

## Quick Start

### 1. Run Test Audit

Identify and fix issues in existing tests:

```bash
cd tools/test-runner
python run_test_audit.py
```

### 2. Run Full Test Suite

Execute all test categories with coverage:

```python
import asyncio
from orchestrator import TestSuiteOrchestrator

async def run_tests():
    orchestrator = TestSuiteOrchestrator()
    results = await orchestrator.run_full_suite()
    print(f"Success rate: {results.overall_summary.success_rate:.1f}%")

asyncio.run(run_tests())
```

### 3. Generate Coverage Report

Measure and report code coverage:

```python
from coverage_analyzer import CoverageAnalyzer
from orchestrator import TestCategory, TestConfig

config = TestConfig.load_from_file("tests/config/test-config.yaml")
analyzer = CoverageAnalyzer(config)

report = analyzer.measure_coverage([TestCategory.UNIT])
analyzer.generate_coverage_report(report, "coverage.html", "html")
```

## Command Line Usage

### Audit Tests

```bash
python tools/test-runner/run_test_audit.py
```

### Run Demo

```bash
python tools/test-runner/example_usage.py
```

## Integration with CI/CD

The system integrates with GitHub Actions and other CI/CD systems:

```yaml
# .github/workflows/test-suite.yml
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
          python-version: 3.11

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run test audit
        run: python tools/test-runner/run_test_audit.py

      - name: Run test suite
        run: |
          python -c "
          import asyncio
          from tools.test_runner import TestSuiteOrchestrator

          async def main():
              orchestrator = TestSuiteOrchestrator()
              results = await orchestrator.run_full_suite()
              exit(0 if results.overall_summary.success_rate >= 80 else 1)

          asyncio.run(main())
          "
```

## Features

### Test Categories

- **Unit Tests**: Fast, isolated component tests
- **Integration Tests**: Cross-component interaction tests
- **Performance Tests**: Benchmarks and load tests
- **End-to-End Tests**: Complete workflow validation

### Execution Modes

- **Parallel Execution**: Run compatible tests concurrently
- **Sequential Execution**: Run tests one at a time for stability
- **Mixed Mode**: Parallel for unit tests, sequential for integration

### Monitoring & Reporting

- **Real-time Progress**: Live updates during test execution
- **Comprehensive Reports**: JSON, HTML, and Markdown formats
- **Coverage Analysis**: Line, branch, and function coverage
- **Trend Analysis**: Historical coverage and performance tracking

### Error Handling

- **Timeout Management**: Graceful handling of long-running tests
- **Failure Recovery**: Continue execution despite individual test failures
- **Resource Management**: Prevent resource exhaustion during parallel execution

## Requirements

- Python 3.8+
- pytest
- pytest-cov
- pytest-asyncio
- PyYAML

## Installation

```bash
# Install required packages
pip install pytest pytest-cov pytest-asyncio pyyaml

# Ensure test configuration exists
cp tests/config/test-config.yaml.example tests/config/test-config.yaml
```

## Contributing

When adding new tests:

1. Place tests in the appropriate category directory
2. Follow naming convention: `test_*.py`
3. Include proper assertions and error handling
4. Update test configuration if needed
5. Run audit to verify categorization: `python tools/test-runner/run_test_audit.py`

## Troubleshooting

### Common Issues

**No tests discovered:**

- Check test file naming (must start with `test_`)
- Verify test directory structure
- Review patterns in `test-config.yaml`

**Coverage measurement fails:**

- Ensure `pytest-cov` is installed
- Check source directory paths in configuration
- Verify file permissions

**Timeout errors:**

- Increase timeout values in configuration
- Check for infinite loops or blocking operations
- Review resource usage during parallel execution

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance

The system is optimized for:

- **Fast Feedback**: Unit tests complete in under 30 seconds
- **Parallel Execution**: Utilizes multiple CPU cores effectively
- **Resource Efficiency**: Manages memory and process limits
- **Scalability**: Handles large test suites (1000+ tests)

## Metrics

Key performance indicators:

- **Test Execution Time**: Target <15 minutes for full suite
- **Coverage Threshold**: Minimum 70% code coverage
- **Success Rate**: Target >95% test pass rate
- **Audit Health**: Zero broken or miscategorized tests

---

For more examples and advanced usage, see `example_usage.py` and the individual module documentation.
