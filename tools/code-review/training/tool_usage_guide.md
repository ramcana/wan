# Code Review Tools Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Code Review Tool](#code-review-tool)
3. [Refactoring Engine](#refactoring-engine)
4. [Technical Debt Tracker](#technical-debt-tracker)
5. [Command Line Interface](#command-line-interface)
6. [Integration with Development Workflow](#integration-with-development-workflow)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Getting Started

### Installation

The code review tools are part of the project's quality improvement suite. No additional installation is required if you have the project set up.

### Prerequisites

- Python 3.8 or higher
- Project dependencies installed
- Write access to the project directory (for technical debt tracking database)

### Quick Start

1. **Run a basic code review:**

   ```bash
   python -m tools.code-review.cli review --project-root .
   ```

2. **Generate refactoring suggestions:**

   ```bash
   python -m tools.code-review.cli refactor --project-root .
   ```

3. **View technical debt:**

   ```bash
   python -m tools.code-review.cli debt list
   ```

4. **Generate comprehensive report:**
   ```bash
   python -m tools.code-review.cli report --output-dir reports/
   ```

## Code Review Tool

### Overview

The code review tool automatically analyzes Python code for quality issues, categorizing them by severity and type. It provides actionable suggestions for improvement.

### Features

- **Complexity Analysis**: Identifies functions with high cyclomatic complexity
- **Maintainability Checks**: Finds long methods, missing documentation, and other maintainability issues
- **Performance Analysis**: Detects potential performance problems like nested loops
- **Security Scanning**: Identifies potentially dangerous function calls
- **Customizable Rules**: Configure thresholds and rules via configuration file

### Usage Examples

#### Basic File Review

```bash
# Review a single file
python -m tools.code-review.cli review --file src/main.py

# Review with specific severity filter
python -m tools.code-review.cli review --file src/main.py --severity high

# Output to JSON file
python -m tools.code-review.cli review --file src/main.py --output review_results.json --format json
```

#### Project-Wide Review

```bash
# Review entire project
python -m tools.code-review.cli review --project-root .

# Filter by category
python -m tools.code-review.cli review --project-root . --category complexity

# Review with multiple filters
python -m tools.code-review.cli review --project-root . --severity medium --category maintainability
```

#### Programmatic Usage

```python
from tools.code_review.code_reviewer import CodeReviewer

# Initialize reviewer
reviewer = CodeReviewer(project_root=".")

# Review single file
issues = reviewer.review_file("src/main.py")
for issue in issues:
    print(f"{issue.severity.value}: {issue.message}")

# Review entire project
result = reviewer.review_project()
print(f"Found {result['issues']} issues in {result['files_reviewed']} files")

# Generate detailed report
report = reviewer.generate_report("detailed_review.json")
```

### Configuration

Create a configuration file at `tools/code-review/config.json`:

```json
{
  "max_complexity": 10,
  "max_function_length": 50,
  "max_class_length": 200,
  "min_documentation_coverage": 80,
  "performance_thresholds": {
    "nested_loops": 3,
    "database_queries_per_function": 5
  },
  "security_patterns": [
    "eval\\(",
    "exec\\(",
    "subprocess\\.call",
    "os\\.system"
  ]
}
```

### Understanding Results

#### Issue Severity Levels

- **CRITICAL**: Issues that could cause security vulnerabilities or system failures
- **HIGH**: Issues that significantly impact maintainability or performance
- **MEDIUM**: Issues that moderately impact code quality
- **LOW**: Minor style or documentation issues
- **INFO**: Informational suggestions for improvement

#### Issue Categories

- **COMPLEXITY**: High cyclomatic complexity, deeply nested code
- **MAINTAINABILITY**: Long methods, missing documentation, poor naming
- **PERFORMANCE**: Inefficient algorithms, resource leaks
- **SECURITY**: Potentially dangerous function calls, input validation issues
- **STYLE**: Formatting, naming conventions
- **DOCUMENTATION**: Missing or inadequate documentation
- **TESTING**: Test-related issues
- **ARCHITECTURE**: Structural and design issues

## Refactoring Engine

### Overview

The refactoring engine analyzes code and provides intelligent suggestions for improving structure, readability, and maintainability.

### Features

- **Extract Method**: Suggests breaking down long methods
- **Extract Class**: Identifies classes with too many responsibilities
- **Simplify Conditionals**: Recommends simplifying complex conditional logic
- **Improve Naming**: Suggests better names for variables and functions
- **Remove Duplication**: Identifies duplicate code patterns
- **Optimize Imports**: Suggests import organization improvements

### Usage Examples

#### Basic Refactoring Analysis

```bash
# Analyze entire project
python -m tools.code-review.cli refactor --project-root .

# Analyze single file
python -m tools.code-review.cli refactor --file src/complex_module.py

# Filter by refactoring type
python -m tools.code-review.cli refactor --project-root . --type extract_method

# Filter by priority
python -m tools.code-review.cli refactor --project-root . --priority 2
```

#### Programmatic Usage

```python
from tools.code_review.refactoring_engine import RefactoringEngine

# Initialize engine
engine = RefactoringEngine(project_root=".")

# Analyze single file
suggestions = engine.analyze_file("src/complex_module.py")
for suggestion in suggestions:
    print(f"Priority {suggestion.priority}: {suggestion.title}")
    print(f"  Benefits: {', '.join(suggestion.benefits)}")
    print(f"  Effort: {suggestion.effort_estimate}")

# Generate comprehensive report
report = engine.generate_suggestions_report("refactoring_report.json")
```

### Understanding Suggestions

#### Refactoring Types

1. **Extract Method**: Break down long or complex methods

   - **When suggested**: Methods longer than 30 lines or with high complexity
   - **Benefits**: Improved readability, better testability, code reuse
   - **Effort**: Medium to high

2. **Extract Class**: Split large classes with multiple responsibilities

   - **When suggested**: Classes longer than 200 lines or with 15+ methods
   - **Benefits**: Single Responsibility Principle, better organization
   - **Effort**: High

3. **Simplify Conditional**: Reduce complex conditional logic

   - **When suggested**: Deeply nested conditions or complex boolean expressions
   - **Benefits**: Improved readability, easier debugging
   - **Effort**: Low to medium

4. **Improve Naming**: Use more descriptive names

   - **When suggested**: Short, generic, or unclear names
   - **Benefits**: Self-documenting code, better comprehension
   - **Effort**: Low

5. **Remove Duplication**: Eliminate duplicate code
   - **When suggested**: Similar code blocks found in multiple places
   - **Benefits**: Reduced maintenance, consistent behavior
   - **Effort**: Medium

#### Priority Levels

- **Priority 1**: Critical refactoring needed (high complexity, major issues)
- **Priority 2**: Important improvements (moderate complexity, significant benefits)
- **Priority 3**: Recommended improvements (minor issues, good practices)
- **Priority 4**: Optional improvements (style, minor optimizations)

## Technical Debt Tracker

### Overview

The technical debt tracker helps teams identify, prioritize, and manage technical debt systematically.

### Features

- **Debt Item Management**: Add, update, and track debt items
- **Priority Scoring**: Automatic priority calculation based on multiple factors
- **Metrics and Analytics**: Comprehensive debt metrics and trends
- **Recommendations**: Intelligent suggestions for debt reduction
- **History Tracking**: Audit trail of all debt item changes

### Usage Examples

#### Managing Debt Items

```bash
# List all debt items
python -m tools.code-review.cli debt list

# Filter by category
python -m tools.code-review.cli debt list --category code_quality

# Filter by severity and status
python -m tools.code-review.cli debt list --severity high --status identified

# Add new debt item
python -m tools.code-review.cli debt add \
  --file src/legacy_module.py \
  --title "Refactor complex authentication logic" \
  --description "The authentication logic is overly complex and hard to test" \
  --category architecture \
  --severity high \
  --effort 16 \
  --business-impact "Difficult to add new authentication methods"

# Update debt item status
python -m tools.code-review.cli debt update \
  --id abc123def456 \
  --status in_progress \
  --assignee "john.doe" \
  --notes "Started refactoring, created new auth service"

# Mark debt item as resolved
python -m tools.code-review.cli debt update \
  --id abc123def456 \
  --status resolved \
  --notes "Completed refactoring, all tests passing"
```

#### Debt Metrics and Analysis

```bash
# View debt metrics
python -m tools.code-review.cli debt metrics

# Export metrics to file
python -m tools.code-review.cli debt metrics --output debt_metrics.json
```

#### Programmatic Usage

```python
from tools.code_review.technical_debt_tracker import (
    TechnicalDebtTracker, TechnicalDebtItem, DebtCategory, DebtSeverity, DebtStatus
)
from datetime import datetime

# Initialize tracker
tracker = TechnicalDebtTracker()

# Add debt item
debt_item = TechnicalDebtItem(
    id="",  # Will be auto-generated
    title="Complex database query needs optimization",
    description="The user search query is slow and complex",
    file_path="src/user_service.py",
    line_start=45,
    line_end=75,
    category=DebtCategory.PERFORMANCE,
    severity=DebtSeverity.HIGH,
    status=DebtStatus.IDENTIFIED,
    created_date=datetime.now(),
    updated_date=datetime.now(),
    estimated_effort_hours=8.0,
    business_impact="Slow user search affects user experience",
    technical_impact="Database performance bottleneck",
    priority_score=0  # Will be calculated automatically
)

item_id = tracker.add_debt_item(debt_item)
print(f"Added debt item: {item_id}")

# Get prioritized debt items
priority_items = tracker.get_prioritized_debt_items(limit=10)
for item in priority_items:
    print(f"Priority {item.priority_score:.1f}: {item.title}")

# Calculate metrics
metrics = tracker.calculate_debt_metrics()
print(f"Total debt items: {metrics.total_items}")
print(f"Total estimated hours: {metrics.total_estimated_hours}")
print(f"Average age: {metrics.average_age_days:.1f} days")

# Generate recommendations
recommendations = tracker.generate_recommendations()
for rec in recommendations:
    print(f"{rec.recommendation_type}: {rec.description}")
```

### Understanding Debt Metrics

#### Key Metrics

- **Total Items**: Number of tracked debt items
- **Total Estimated Hours**: Sum of effort estimates for all items
- **Average Age**: Average age of debt items in days
- **Oldest Item**: Age of the oldest unresolved debt item
- **Debt Trend**: Whether debt is increasing, decreasing, or stable
- **Resolution Rate**: Number of items resolved per week

#### Priority Scoring

The priority score is calculated based on:

- **Severity Weight**: Critical (10), High (7), Medium (4), Low (1)
- **Category Weight**: Security (3x), Performance (2.5x), Architecture (2x), etc.
- **Age Factor**: Older items get higher priority (up to 2x for items over 1 year)
- **Effort Factor**: Prefer items with reasonable effort (1-40 hours get bonus)

## Command Line Interface

### Global Options

```bash
# Get help for any command
python -m tools.code-review.cli --help
python -m tools.code-review.cli review --help
python -m tools.code-review.cli debt --help
```

### Review Command

```bash
# Basic usage
python -m tools.code-review.cli review [OPTIONS]

# Options:
--project-root PATH    # Project root directory (default: .)
--file PATH           # Specific file to review
--output PATH         # Output file for results
--format FORMAT       # Output format: json, text (default: text)
--severity LEVEL      # Filter by minimum severity
--category CATEGORY   # Filter by category
```

### Refactor Command

```bash
# Basic usage
python -m tools.code-review.cli refactor [OPTIONS]

# Options:
--project-root PATH    # Project root directory (default: .)
--file PATH           # Specific file to analyze
--output PATH         # Output file for suggestions
--type TYPE           # Filter by refactoring type
--priority LEVEL      # Filter by minimum priority
```

### Debt Command

```bash
# List debt items
python -m tools.code-review.cli debt list [OPTIONS]

# Add debt item
python -m tools.code-review.cli debt add [OPTIONS]

# Update debt item
python -m tools.code-review.cli debt update [OPTIONS]

# Show metrics
python -m tools.code-review.cli debt metrics [OPTIONS]
```

### Report Command

```bash
# Generate comprehensive report
python -m tools.code-review.cli report [OPTIONS]

# Options:
--project-root PATH    # Project root directory (default: .)
--output-dir PATH     # Output directory (default: reports)
--format FORMAT       # Report format: json, html (default: json)
```

## Integration with Development Workflow

### Pre-commit Hooks

Add code review checks to your pre-commit configuration:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: code-review
        name: Code Review Check
        entry: python -m tools.code-review.cli review --severity high
        language: system
        pass_filenames: false
        always_run: true
```

### CI/CD Integration

#### GitHub Actions Example

```yaml
# .github/workflows/code-quality.yml
name: Code Quality Check

on: [push, pull_request]

jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run code review
        run: |
          python -m tools.code-review.cli review --project-root . --format json --output review_results.json
          python -m tools.code-review.cli refactor --project-root . --output refactor_suggestions.json

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: code-quality-results
          path: |
            review_results.json
            refactor_suggestions.json
```

### IDE Integration

#### VS Code Integration

Create a VS Code task for running code reviews:

```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Code Review",
      "type": "shell",
      "command": "python",
      "args": [
        "-m",
        "tools.code-review.cli",
        "review",
        "--file",
        "${file}",
        "--format",
        "text"
      ],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    }
  ]
}
```

### Team Workflow Integration

#### Daily Standup Integration

```bash
# Generate daily debt report
python -m tools.code-review.cli debt metrics > daily_debt_report.txt

# Show high-priority items
python -m tools.code-review.cli debt list --severity high --limit 5
```

#### Sprint Planning Integration

```bash
# Generate comprehensive report for sprint planning
python -m tools.code-review.cli report --output-dir sprint_reports/

# Identify refactoring opportunities
python -m tools.code-review.cli refactor --priority 1 --output high_priority_refactoring.json
```

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors

**Problem**: Python can't find the code review modules.

**Solution**:

- Ensure you're running from the project root directory
- Check that the `tools/code-review/` directory exists
- Verify Python path includes the project root

#### 2. Database permission errors

**Problem**: Can't create or write to the technical debt database.

**Solution**:

- Check write permissions in the project directory
- Ensure the database file isn't locked by another process
- Try running with appropriate permissions

#### 3. Large project performance issues

**Problem**: Code review takes too long on large projects.

**Solution**:

- Use file-specific reviews for development
- Configure exclusion patterns for generated code
- Run full project reviews in CI/CD only

#### 4. False positive security warnings

**Problem**: Security analyzer flags legitimate code.

**Solution**:

- Review the security patterns in configuration
- Add exceptions for known safe patterns
- Use comments to document why code is safe

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from tools.code_review.code_reviewer import CodeReviewer
reviewer = CodeReviewer(project_root=".")
# Debug output will show detailed analysis steps
```

### Performance Optimization

#### For Large Projects

1. **Use file filtering**: Focus on specific directories or file patterns
2. **Parallel processing**: The tools support concurrent file analysis
3. **Incremental analysis**: Only analyze changed files in CI/CD
4. **Caching**: Results can be cached for unchanged files

#### Configuration Tuning

```json
{
  "performance_mode": true,
  "max_file_size_mb": 1,
  "exclude_patterns": [
    "*/migrations/*",
    "*/tests/fixtures/*",
    "*/vendor/*",
    "*/__pycache__/*"
  ],
  "parallel_workers": 4
}
```

## Advanced Usage

### Custom Analyzers

Extend the code review system with custom analyzers:

```python
from tools.code_review.code_reviewer import CodeReviewer, CodeIssue, ReviewSeverity, IssueCategory
import ast

class CustomSecurityAnalyzer:
    def analyze(self, file_path: str, tree: ast.AST, content: str) -> List[CodeIssue]:
        issues = []

        # Custom security check
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if self._is_hardcoded_password(node):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity=ReviewSeverity.CRITICAL,
                        category=IssueCategory.SECURITY,
                        message="Hardcoded password detected",
                        suggestion="Use environment variables or secure configuration",
                        rule_id="HARDCODED_PASSWORD"
                    ))

        return issues

    def _is_hardcoded_password(self, node):
        # Custom logic to detect hardcoded passwords
        pass

# Extend the reviewer
reviewer = CodeReviewer()
reviewer.custom_analyzer = CustomSecurityAnalyzer()
```

### Batch Processing

Process multiple projects or repositories:

```python
import os
from pathlib import Path
from tools.code_review.code_reviewer import CodeReviewer

def batch_review(project_paths):
    results = {}

    for project_path in project_paths:
        print(f"Reviewing {project_path}...")
        reviewer = CodeReviewer(project_path)
        result = reviewer.review_project()
        results[project_path] = result

        # Generate individual report
        report_path = Path(project_path) / "code_review_report.json"
        reviewer.generate_report(str(report_path))

    return results

# Review multiple projects
projects = ["./project1", "./project2", "./project3"]
batch_results = batch_review(projects)
```

### Integration with External Tools

#### SonarQube Integration

Export results in SonarQube format:

```python
def export_to_sonarqube_format(issues, output_path):
    sonar_issues = []

    for issue in issues:
        sonar_issue = {
            "engineId": "code-review-tool",
            "ruleId": issue.rule_id,
            "severity": issue.severity.value.upper(),
            "type": "CODE_SMELL",
            "primaryLocation": {
                "message": issue.message,
                "filePath": issue.file_path,
                "textRange": {
                    "startLine": issue.line_number,
                    "startColumn": issue.column
                }
            }
        }
        sonar_issues.append(sonar_issue)

    with open(output_path, 'w') as f:
        json.dump({"issues": sonar_issues}, f, indent=2)
```

#### Slack Integration

Send debt metrics to Slack:

```python
import requests
import json

def send_debt_metrics_to_slack(webhook_url, metrics):
    message = {
        "text": "Daily Technical Debt Report",
        "attachments": [
            {
                "color": "warning" if metrics.total_items > 10 else "good",
                "fields": [
                    {"title": "Total Items", "value": str(metrics.total_items), "short": True},
                    {"title": "Total Hours", "value": f"{metrics.total_estimated_hours:.1f}", "short": True},
                    {"title": "Average Age", "value": f"{metrics.average_age_days:.1f} days", "short": True},
                    {"title": "Trend", "value": metrics.debt_trend, "short": True}
                ]
            }
        ]
    }

    requests.post(webhook_url, json=message)
```

This comprehensive guide should help teams effectively use the code review and refactoring assistance tools to improve their code quality and manage technical debt systematically.
