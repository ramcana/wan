# Code Review and Refactoring Assistance Tools

A comprehensive suite of tools for automated code review, refactoring suggestions, and technical debt management. This system helps development teams maintain high code quality, identify improvement opportunities, and systematically manage technical debt.

## Features

### 🔍 Automated Code Review

- **Multi-dimensional Analysis**: Complexity, maintainability, performance, and security
- **Configurable Rules**: Customizable thresholds and detection patterns
- **Severity Classification**: Issues categorized by impact and urgency
- **Actionable Suggestions**: Clear recommendations for fixing identified issues

### 🔧 Intelligent Refactoring Suggestions

- **Pattern Recognition**: Identifies common refactoring opportunities
- **Priority-based Recommendations**: Suggestions ranked by impact and effort
- **Multiple Refactoring Types**: Extract method, simplify conditionals, improve naming, and more
- **Benefit Analysis**: Clear explanation of improvements for each suggestion

### 📊 Technical Debt Tracking

- **Comprehensive Tracking**: Systematic debt identification and management
- **Priority Scoring**: Intelligent prioritization based on multiple factors
- **Metrics and Analytics**: Detailed insights into debt trends and resolution rates
- **Recommendation Engine**: Smart suggestions for debt reduction strategies

### 🛠 Developer-Friendly Tools

- **Command Line Interface**: Easy-to-use CLI for all operations
- **Programmatic API**: Full Python API for custom integrations
- **Multiple Output Formats**: JSON, text, and HTML reporting
- **CI/CD Integration**: Ready for continuous integration workflows

## Quick Start

### Basic Usage

```bash
# Review entire project
python -m tools.code-review.cli review --project-root .

# Generate refactoring suggestions
python -m tools.code-review.cli refactor --project-root .

# View technical debt
python -m tools.code-review.cli debt list

# Generate comprehensive report
python -m tools.code-review.cli report --output-dir reports/
```

### Review a Single File

```bash
python -m tools.code-review.cli review --file src/main.py --format json
```

### Add Technical Debt Item

```bash
python -m tools.code-review.cli debt add \
  --file src/legacy_module.py \
  --title "Complex authentication logic needs refactoring" \
  --severity high \
  --category architecture \
  --effort 16
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Project dependencies installed
- Write access to project directory (for technical debt database)

### Configuration

The tools use a configuration file at `tools/code-review/config.json`. Key settings include:

```json
{
  "max_complexity": 10,
  "max_function_length": 50,
  "max_class_length": 200,
  "security_patterns": ["eval\\(", "exec\\("],
  "exclude_patterns": ["*/tests/*", "*/__pycache__/*"]
}
```

## Tool Components

### 1. Code Reviewer (`code_reviewer.py`)

Analyzes Python code for quality issues across multiple dimensions:

- **Complexity Analysis**: Cyclomatic complexity calculation
- **Maintainability Checks**: Function length, documentation coverage
- **Performance Analysis**: Nested loops, inefficient patterns
- **Security Scanning**: Dangerous function calls, potential vulnerabilities

**Example Usage:**

```python
from tools.code_review.code_reviewer import CodeReviewer

reviewer = CodeReviewer(project_root=".")
issues = reviewer.review_file("src/main.py")
for issue in issues:
    print(f"{issue.severity.value}: {issue.message}")
```

### 2. Refactoring Engine (`refactoring_engine.py`)

Provides intelligent refactoring suggestions based on code analysis:

- **Extract Method**: Break down long or complex methods
- **Extract Class**: Split large classes with multiple responsibilities
- **Simplify Conditionals**: Reduce complex conditional logic
- **Improve Naming**: Suggest better variable and function names
- **Remove Duplication**: Identify and eliminate duplicate code

**Example Usage:**

```python
from tools.code_review.refactoring_engine import RefactoringEngine

engine = RefactoringEngine(project_root=".")
suggestions = engine.analyze_file("src/complex_module.py")
for suggestion in suggestions:
    print(f"Priority {suggestion.priority}: {suggestion.title}")
```

### 3. Technical Debt Tracker (`technical_debt_tracker.py`)

Comprehensive system for tracking and managing technical debt:

- **Debt Item Management**: Add, update, and track debt items
- **Priority Scoring**: Automatic calculation based on severity, category, age, and effort
- **Metrics and Analytics**: Comprehensive debt metrics and trend analysis
- **Recommendations**: Intelligent suggestions for debt reduction

**Example Usage:**

```python
from tools.code_review.technical_debt_tracker import TechnicalDebtTracker

tracker = TechnicalDebtTracker()
metrics = tracker.calculate_debt_metrics()
print(f"Total debt items: {metrics.total_items}")
print(f"Total estimated hours: {metrics.total_estimated_hours}")
```

### 4. Command Line Interface (`cli.py`)

Unified CLI providing access to all functionality:

```bash
# Code review commands
python -m tools.code-review.cli review [OPTIONS]

# Refactoring commands
python -m tools.code-review.cli refactor [OPTIONS]

# Technical debt commands
python -m tools.code-review.cli debt [SUBCOMMAND] [OPTIONS]

# Reporting commands
python -m tools.code-review.cli report [OPTIONS]
```

## Integration Examples

### Pre-commit Hooks

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

### GitHub Actions

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
      - name: Run code review
        run: |
          python -m tools.code-review.cli review --project-root . --format json --output review_results.json
          python -m tools.code-review.cli refactor --project-root . --output refactor_suggestions.json
```

### VS Code Integration

```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Code Review",
      "type": "shell",
      "command": "python",
      "args": ["-m", "tools.code-review.cli", "review", "--file", "${file}"],
      "group": "build"
    }
  ]
}
```

## Understanding Results

### Issue Severity Levels

- **CRITICAL**: Security vulnerabilities, system failures
- **HIGH**: Significant maintainability or performance issues
- **MEDIUM**: Moderate code quality impacts
- **LOW**: Minor style or documentation issues
- **INFO**: Informational suggestions

### Issue Categories

- **COMPLEXITY**: High cyclomatic complexity, deeply nested code
- **MAINTAINABILITY**: Long methods, missing documentation, poor naming
- **PERFORMANCE**: Inefficient algorithms, resource leaks
- **SECURITY**: Dangerous function calls, input validation issues
- **STYLE**: Formatting, naming conventions
- **DOCUMENTATION**: Missing or inadequate documentation
- **TESTING**: Test-related issues
- **ARCHITECTURE**: Structural and design issues

### Refactoring Priority Levels

- **Priority 1**: Critical refactoring needed (high complexity, major issues)
- **Priority 2**: Important improvements (moderate complexity, significant benefits)
- **Priority 3**: Recommended improvements (minor issues, good practices)
- **Priority 4**: Optional improvements (style, minor optimizations)

### Technical Debt Priority Scoring

Priority scores are calculated based on:

- **Severity Weight**: Critical (10), High (7), Medium (4), Low (1)
- **Category Weight**: Security (3x), Performance (2.5x), Architecture (2x)
- **Age Factor**: Older items get higher priority (up to 2x multiplier)
- **Effort Factor**: Reasonable effort items get priority bonus

## Advanced Usage

### Custom Analyzers

Extend the system with custom analysis rules:

```python
from tools.code_review.code_reviewer import CodeReviewer, CodeIssue
import ast

class CustomAnalyzer:
    def analyze(self, file_path: str, tree: ast.AST, content: str):
        # Custom analysis logic
        issues = []
        # ... analyze and create issues
        return issues

reviewer = CodeReviewer()
reviewer.custom_analyzer = CustomAnalyzer()
```

### Batch Processing

Process multiple projects:

```python
def batch_review(project_paths):
    results = {}
    for project_path in project_paths:
        reviewer = CodeReviewer(project_path)
        results[project_path] = reviewer.review_project()
    return results
```

### External Tool Integration

Export results for external tools:

```python
def export_to_sonarqube(issues, output_path):
    # Convert to SonarQube format
    sonar_issues = [convert_issue(issue) for issue in issues]
    with open(output_path, 'w') as f:
        json.dump({"issues": sonar_issues}, f)
```

## Configuration Options

### Code Review Settings

```json
{
  "max_complexity": 10,
  "max_function_length": 50,
  "max_class_length": 200,
  "min_documentation_coverage": 80
}
```

### Performance Settings

```json
{
  "performance_mode": true,
  "parallel_workers": 4,
  "cache_results": true,
  "max_file_size_mb": 1
}
```

### Security Patterns

```json
{
  "security_patterns": [
    "eval\\(",
    "exec\\(",
    "subprocess\\.call",
    "os\\.system"
  ]
}
```

### Exclusion Patterns

```json
{
  "exclude_patterns": [
    "*/migrations/*",
    "*/tests/fixtures/*",
    "*/vendor/*",
    "*/__pycache__/*"
  ]
}
```

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure you're running from the project root
2. **Database permission errors**: Check write permissions in project directory
3. **Performance issues**: Use file-specific reviews for large projects
4. **False positive warnings**: Configure exclusion patterns appropriately

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

For large projects:

- Use file filtering and exclusion patterns
- Enable parallel processing
- Use incremental analysis in CI/CD
- Cache results for unchanged files

## Training and Best Practices

### Documentation

- [Best Practices Guide](training/best_practices.md): Comprehensive guide to code quality best practices
- [Tool Usage Guide](training/tool_usage_guide.md): Detailed instructions for using the tools effectively

### Key Principles

1. **Readability First**: Code should be written for humans to read
2. **Simplicity**: Simple solutions are often the best solutions
3. **Consistency**: Maintain consistent patterns throughout the codebase
4. **Continuous Improvement**: Regular refactoring and debt reduction
5. **Team Collaboration**: Use tools to facilitate better code reviews

## Contributing

### Adding New Analyzers

1. Create analyzer class with `analyze()` method
2. Return list of `CodeIssue` objects
3. Register analyzer with main reviewer
4. Add tests and documentation

### Extending Refactoring Patterns

1. Add new `RefactoringPattern` to engine
2. Implement detection logic
3. Provide clear suggestions and benefits
4. Test with real code examples

### Improving Debt Tracking

1. Enhance priority scoring algorithm
2. Add new debt categories or metrics
3. Improve recommendation engine
4. Add integration with external tools

## License

This code review and refactoring assistance system is part of the WAN22 project quality improvement initiative. See the main project license for details.

## Support

For questions, issues, or contributions:

1. Check the troubleshooting section
2. Review the training documentation
3. Create an issue in the project repository
4. Consult the team's coding standards and practices

---

**Remember**: These tools are designed to assist, not replace, human judgment. Use the suggestions as starting points for discussion and improvement, always considering the specific context and requirements of your project.
