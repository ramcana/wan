# Test Suite Auditor

A comprehensive test suite analysis and auditing system that identifies broken, incomplete, and flaky tests while providing detailed performance profiling and coverage analysis.

## Features

### 🔍 Test Discovery Engine

- Automatically discovers all test files across the project
- Supports multiple test patterns and directory structures
- Categorizes tests by type (unit, integration, e2e, performance)

### 🛠️ Test Auditing System

- **Syntax Analysis**: Identifies syntax errors and import issues
- **Structure Analysis**: Detects empty tests and missing assertions
- **Dependency Analysis**: Finds missing imports and fixtures
- **Quality Assessment**: Evaluates test completeness and reliability

### ⚡ Performance Profiler

- Measures test execution times with configurable timeouts
- Identifies slow and hanging tests
- Provides performance optimization recommendations
- Supports parallel test execution with resource management

### 📊 Coverage Analyzer

- Comprehensive code coverage analysis
- Function and branch coverage tracking
- Coverage gap identification with severity levels
- Threshold management and violation detection

### 🎯 Test Runner Engine

- Isolated test execution with proper cleanup
- Retry logic for flaky tests with exponential backoff
- Timeout handling with configurable limits
- Parallel execution with resource management

### 📈 Health Scoring

- Overall test suite health score (0-100)
- Component-specific scoring (syntax, coverage, performance, reliability)
- Trend analysis and improvement tracking

### 📋 Action Planning

- Prioritized action plans for test suite improvements
- Effort estimation and impact analysis
- Critical issue identification and resolution guidance

## Installation

The test auditor requires Python 3.8+ and several dependencies:

```bash
# Install required packages
pip install pytest coverage pytest-json-report pytest-cov
```

## Usage

### Quick Start

Run a comprehensive analysis of your entire test suite:

```bash
python tools/test-auditor/orchestrator.py
```

### Command Line Interface

The test auditor provides several CLI tools:

#### 1. Comprehensive Analysis (Recommended)

```bash
# Run full analysis with all components
python tools/test-auditor/orchestrator.py --project-root . --output analysis.json

# Print summary only
python tools/test-auditor/orchestrator.py --summary-only
```

#### 2. Test Auditing Only

```bash
# Run basic audit
python tools/test-auditor/cli.py audit

# Save to specific file
python tools/test-auditor/cli.py audit --output my_report.json --format json

# Show summary of existing report
python tools/test-auditor/cli.py summary --report my_report.json

# Show issues filtered by severity
python tools/test-auditor/cli.py issues --severity critical

# Show performance analysis
python tools/test-auditor/cli.py performance --threshold 10.0

# Show file analysis
python tools/test-auditor/cli.py files --broken-only
```

#### 3. Test Execution with Monitoring

```bash
# Run tests with advanced monitoring
python tools/test-auditor/test_runner.py --parallel 4 --timeout 60

# Run specific test files
python tools/test-auditor/test_runner.py tests/unit/test_example.py tests/integration/

# Save execution results
python tools/test-auditor/test_runner.py --output execution_report.json
```

#### 4. Coverage Analysis

```bash
# Run coverage analysis
python tools/test-auditor/coverage_analyzer.py

# Set custom threshold
python tools/test-auditor/coverage_analyzer.py --threshold 85

# Analyze specific test files
python tools/test-auditor/coverage_analyzer.py --test-files tests/unit/ tests/integration/

# Save coverage report
python tools/test-auditor/coverage_analyzer.py --output coverage_report.json
```

### Python API

You can also use the test auditor programmatically:

```python
from pathlib import Path
from tools.test_auditor.orchestrator import TestSuiteOrchestrator

# Create orchestrator
project_root = Path(".")
orchestrator = TestSuiteOrchestrator(project_root)

# Run comprehensive analysis
analysis = orchestrator.run_comprehensive_analysis()

# Access results
print(f"Health Score: {analysis.health_score}")
print(f"Total Tests: {analysis.audit_report.total_tests}")
print(f"Coverage: {analysis.coverage_report.overall_percentage}%")

# Get action plan
for action in analysis.action_plan:
    print(f"[{action['priority']}] {action['title']}")
```

## Output Formats

### JSON Report

Detailed machine-readable report with all analysis data:

```json
{
  "audit_report": {
    "total_files": 45,
    "total_tests": 234,
    "passing_tests": 198,
    "failing_tests": 12,
    "critical_issues": [...],
    "recommendations": [...]
  },
  "execution_report": {
    "successful_files": 38,
    "failed_files": 7,
    "performance_summary": {...}
  },
  "coverage_report": {
    "overall_percentage": 78.5,
    "coverage_gaps": [...],
    "threshold_violations": [...]
  },
  "health_score": 82.3,
  "action_plan": [...]
}
```

### HTML Report

Human-readable HTML report with visualizations:

```bash
python tools/test-auditor/cli.py audit --format html --output report.html
```

### Text Report

Simple text summary for quick review:

```bash
python tools/test-auditor/cli.py audit --format text --output report.txt
```

## Configuration

### Test Discovery Configuration

The test auditor automatically discovers tests using these patterns:

- `test_*.py`
- `*_test.py`
- `tests.py`

Common test directories:

- `tests/`
- `backend/tests/`
- `frontend/src/tests/`
- `local_installation/tests/`

### Timeout Configuration

Default timeouts by test type:

- Unit tests: 30 seconds
- Integration tests: 60 seconds
- E2E tests: 120 seconds
- Performance tests: 300 seconds
- Stress tests: 600 seconds

### Coverage Thresholds

Default coverage thresholds:

- Overall coverage: 80%
- File coverage: 70%
- Function coverage: 75%
- Class coverage: 80%
- Branch coverage: 70%

### Health Score Weights

Health score calculation weights:

- Syntax health: 20%
- Test completeness: 15%
- Execution success: 25%
- Performance: 15%
- Coverage: 20%
- Reliability: 5%

## Understanding Results

### Health Score Interpretation

- **90-100**: Excellent - Test suite is in great shape
- **80-89**: Good - Minor improvements needed
- **70-79**: Fair - Several issues to address
- **60-69**: Poor - Significant problems exist
- **Below 60**: Critical - Major overhaul required

### Issue Severity Levels

- **Critical**: Must be fixed immediately (syntax errors, broken imports)
- **High**: Should be fixed soon (failing tests, missing assertions)
- **Medium**: Should be addressed (missing fixtures, slow tests)
- **Low**: Nice to have (minor optimizations, style issues)

### Action Plan Priorities

- **Critical**: Fix immediately to prevent system failure
- **High**: Address within current sprint/iteration
- **Medium**: Plan for next sprint/iteration
- **Low**: Long-term improvements and optimizations

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test Suite Health Check
on: [push, pull_request]

jobs:
  test-health:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install pytest coverage pytest-json-report pytest-cov

      - name: Run test suite analysis
        run: |
          python tools/test-auditor/orchestrator.py --output test_analysis.json

      - name: Upload analysis results
        uses: actions/upload-artifact@v2
        with:
          name: test-analysis
          path: test_analysis.json

      - name: Check health score
        run: |
          python -c "
          import json
          with open('test_analysis.json') as f:
              data = json.load(f)
          score = data['health_score']
          print(f'Health Score: {score}')
          exit(0 if score >= 70 else 1)
          "
```

### Pre-commit Hook

```bash
# Install pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running test suite health check..."
python tools/test-auditor/orchestrator.py --summary-only
if [ $? -ne 0 ]; then
    echo "Test suite health check failed. Please fix issues before committing."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all test dependencies are installed
2. **Permission Errors**: Check file permissions for test directories
3. **Timeout Issues**: Increase timeout limits for slow tests
4. **Coverage Collection Fails**: Ensure `coverage` package is installed

### Debug Mode

Enable verbose output for debugging:

```bash
export PYTEST_VERBOSE=1
python tools/test-auditor/orchestrator.py
```

### Performance Issues

If analysis is slow:

- Reduce parallel workers: `--parallel 2`
- Exclude large test directories
- Use test file filtering

## Contributing

To contribute to the test auditor:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all existing tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install pytest coverage pytest-json-report pytest-cov black flake8

# Run tests
python -m pytest tools/test-auditor/tests/

# Format code
black tools/test-auditor/

# Lint code
flake8 tools/test-auditor/
```

## License

This test auditor is part of the WAN22 project and follows the same license terms.
