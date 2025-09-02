# Test Quality Improvement Tools

Comprehensive test coverage analysis system that identifies untested code paths, enforces coverage thresholds for new code, generates detailed reports with actionable recommendations, and tracks coverage trends over time.

## Features

### 1. Comprehensive Coverage Analysis

- **Untested Code Path Detection**: Identifies functions, classes, and code blocks without test coverage
- **Branch Coverage Analysis**: Detects uncovered conditional branches and decision points
- **Function-Level Coverage**: Provides detailed coverage metrics for individual functions
- **File-Level Analysis**: Comprehensive per-file coverage reporting

### 2. Coverage Threshold Enforcement

- **Overall Coverage Thresholds**: Enforces minimum coverage requirements for the entire codebase
- **New Code Coverage**: Higher standards for newly added or modified code
- **Critical File Protection**: Special thresholds for critical system components
- **Automated Violation Detection**: Identifies and reports threshold violations

### 3. Detailed Reporting System

- **Actionable Recommendations**: Specific suggestions for improving coverage
- **Priority-Based Gap Analysis**: Categorizes coverage gaps by severity and impact
- **Multiple Report Formats**: JSON, HTML, and Markdown output options
- **Interactive Dashboards**: Visual coverage analysis and trends

### 4. Coverage Trend Tracking

- **Historical Data**: Tracks coverage changes over time
- **Trend Analysis**: Identifies improving, declining, or stable coverage patterns
- **Commit-Level Tracking**: Associates coverage data with specific commits
- **Performance Regression Detection**: Alerts when coverage significantly drops

## Installation

The coverage system is part of the test quality improvement tools:

```bash
# Install required dependencies
pip install coverage[toml] pytest

# The tools are ready to use from the project root
cd tools/test-quality
```

## Quick Start

### Comprehensive Analysis (Recommended)

```bash
# Run all test quality analyses
python test_quality_cli.py analyze-all

# Run with custom thresholds
python test_quality_cli.py analyze-all --coverage-threshold 85 --performance-threshold 3.0 --flakiness-threshold 0.05

# Fail CI if issues found
python test_quality_cli.py analyze-all --fail-on-issues
```

### Individual Analysis Tools

#### Coverage Analysis

```bash
# Run comprehensive coverage analysis
python test_quality_cli.py coverage analyze

# Analyze specific test files
python test_quality_cli.py coverage analyze --test-files tests/unit/test_*.py

# Set custom thresholds
python test_quality_cli.py coverage analyze --overall-threshold 85 --new-code-threshold 90

# View coverage trends
python test_quality_cli.py coverage trends --days 14
```

#### Performance Analysis

```bash
# Run performance analysis
python test_quality_cli.py performance analyze

# Set slow test threshold
python test_quality_cli.py performance analyze --slow-threshold 3.0

# Clear cache before analysis
python test_quality_cli.py performance analyze --clear-cache

# View performance trends for specific test
python test_quality_cli.py performance trends --test-id "tests/unit/test_slow.py::test_function"
```

#### Flaky Test Detection

```bash
# Run flaky test analysis
python test_quality_cli.py flaky analyze

# Analyze with custom threshold
python test_quality_cli.py flaky analyze --flakiness-threshold 0.05 --days 14

# Record test results for tracking
python test_quality_cli.py flaky record test_results.xml

# Manage quarantined tests
python test_quality_cli.py flaky quarantine --list
python test_quality_cli.py flaky quarantine --release "tests/unit/test_flaky.py::test_function"
```

## Configuration

### Coverage Thresholds

The system supports different threshold levels:

- **Overall Coverage**: Minimum coverage for the entire codebase (default: 80%)
- **New Code Coverage**: Higher threshold for new/modified code (default: 85%)
- **Critical Files**: Special threshold for critical components (default: 90%)

### Critical File Patterns

Files matching these patterns require higher coverage:

- `core/` - Core system components
- `api/` - API endpoints and handlers
- `services/` - Business logic services

### Exclusions

Files automatically excluded from coverage requirements:

- Test files (`test_*.py`, `*_test.py`)
- Virtual environments (`venv/`, `.venv/`)
- Cache directories (`__pycache__/`)
- Build artifacts

## Usage Examples

### CI/CD Integration

```bash
# In your CI pipeline
python coverage_cli.py analyze --fail-on-violation --output coverage_report.json

# Check if coverage meets requirements (exit code 0 = pass, 1 = fail)
if [ $? -eq 0 ]; then
    echo "Coverage requirements met ✅"
else
    echo "Coverage requirements not met ❌"
    exit 1
fi
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Checking coverage for new code..."
python tools/test-quality/coverage_cli.py thresholds --check-only --base-branch origin/main

if [ $? -ne 0 ]; then
    echo "Coverage check failed. Please add tests for new code."
    exit 1
fi
```

### Development Workflow

```bash
# Before starting work
python coverage_cli.py trends --days 7

# After making changes
python coverage_cli.py analyze --base-branch main

# Generate report for review
python coverage_cli.py report --format html --output coverage_review.html
```

## API Usage

### Programmatic Access

```python
from pathlib import Path
from coverage_system import ComprehensiveCoverageSystem

# Initialize system
project_root = Path.cwd()
system = ComprehensiveCoverageSystem(project_root)

# Configure thresholds
system.threshold_enforcer.set_thresholds(
    overall=85.0,
    new_code=90.0,
    critical=95.0
)

# Run analysis
result = system.run_comprehensive_analysis()

# Access results
coverage_report = result['basic_coverage']
threshold_result = result['threshold_enforcement']
detailed_analysis = result['detailed_analysis']

print(f"Overall coverage: {coverage_report['overall_percentage']:.1f}%")
print(f"Thresholds passed: {threshold_result['passed']}")
```

### Custom Analysis

```python
from coverage_system import CoverageTrendTracker, NewCodeCoverageAnalyzer

# Track trends
tracker = CoverageTrendTracker(project_root)
trends = tracker.get_trends(days=30)

# Analyze new code
analyzer = NewCodeCoverageAnalyzer(project_root)
new_code_results = analyzer.analyze_new_code_coverage(coverage_report, 'main')

for result in new_code_results:
    if not result.meets_threshold:
        print(f"New code in {result.file_path} needs more tests")
        print(f"Coverage: {result.new_code_coverage:.1f}% (required: {result.threshold_required:.1f}%)")
```

## Report Formats

### JSON Report Structure

```json
{
  "basic_coverage": {
    "overall_percentage": 82.5,
    "total_files": 45,
    "covered_files": 38,
    "coverage_gaps": [...]
  },
  "threshold_enforcement": {
    "passed": true,
    "violations": [],
    "new_code_results": [...]
  },
  "detailed_analysis": {
    "summary": {...},
    "file_analysis": [...],
    "gap_analysis": {...},
    "recommendations": [...],
    "trends": {...},
    "actionable_items": [...]
  }
}
```

### HTML Report Features

- Interactive coverage visualization
- Sortable file coverage table
- Coverage trend charts
- Drill-down capability for detailed analysis
- Responsive design for mobile viewing

### Markdown Report

- GitHub-compatible markdown format
- Emoji indicators for priority levels
- Table format for easy scanning
- Integration with pull request reviews

## Troubleshooting

### Common Issues

1. **Coverage data not collected**

   - Ensure `coverage` package is installed
   - Check that tests can run successfully
   - Verify test discovery is working

2. **Git integration not working**

   - Ensure you're in a git repository
   - Check that git commands are available
   - Verify branch names are correct

3. **Threshold violations**
   - Review specific violation messages
   - Use detailed reports to identify gaps
   - Focus on high-priority recommendations

### Debug Mode

```bash
# Enable verbose output
python coverage_cli.py analyze --verbose

# Check system status
python coverage_cli.py analyze --test-files tests/unit/test_simple.py
```

## Integration with Other Tools

### pytest Integration

```bash
# Run with pytest
python -m pytest --cov=. --cov-report=json:coverage.json
python coverage_cli.py report --input coverage.json
```

### IDE Integration

Most IDEs can be configured to run coverage analysis:

- **VS Code**: Use the Python extension with coverage support
- **PyCharm**: Built-in coverage runner integration
- **Vim/Neovim**: Coverage plugins available

## Contributing

To contribute to the coverage system:

1. Add new analyzers in `coverage_system.py`
2. Extend CLI commands in `coverage_cli.py`
3. Update documentation and examples
4. Add tests for new functionality

## License

This tool is part of the WAN22 project and follows the same license terms.
