---
title: tools.code_quality.models
category: api
tags: [api, tools]
---

# tools.code_quality.models

Data models for code quality checking system.

## Classes

### QualityIssueType

Types of quality issues that can be detected.

Attributes:
    FORMATTING: Code formatting issues (spacing, indentation, etc.)
    STYLE: Code style issues (naming conventions, etc.)
    DOCUMENTATION: Missing or inadequate documentation
    TYPE_HINTS: Missing or incorrect type hints
    COMPLEXITY: Code complexity issues
    IMPORTS: Import organization and usage issues
    NAMING: Variable and function naming issues

### QualitySeverity

Severity levels for quality issues.

Attributes:
    ERROR: Critical issues that must be fixed
    WARNING: Important issues that should be addressed
    INFO: Minor issues or suggestions for improvement

### QualityIssue

Represents a single code quality issue.

Attributes:
    file_path: Path to the file containing the issue
    line_number: Line number where the issue occurs
    column: Column number where the issue occurs
    issue_type: Type of quality issue
    severity: Severity level of the issue
    message: Human-readable description of the issue
    rule_code: Unique identifier for the rule that detected this issue
    suggestion: Optional suggestion for fixing the issue
    auto_fixable: Whether this issue can be automatically fixed

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942851CAF0>

Convert issue to dictionary format.

Returns:
    Dict[str, Any]: Dictionary representation of the quality issue.

### QualityMetrics

Code quality metrics for a file or project.

Attributes:
    total_lines: Total number of lines in the code
    code_lines: Number of lines containing actual code
    comment_lines: Number of lines containing comments
    blank_lines: Number of blank lines
    functions_count: Total number of functions
    classes_count: Total number of classes
    modules_count: Total number of modules
    documented_functions: Number of functions with documentation
    documented_classes: Number of classes with documentation
    documented_modules: Number of modules with documentation
    type_annotated_functions: Number of functions with type annotations
    complexity_score: Average cyclomatic complexity score
    maintainability_index: Maintainability index score

#### Methods

##### documentation_coverage(self: Any) -> float

Calculate documentation coverage percentage.

Returns:
    float: Documentation coverage as a percentage (0-100).

##### type_hint_coverage(self: Any) -> float

Calculate type hint coverage percentage.

Returns:
    float: Type hint coverage as a percentage (0-100).

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942894C6D0>

Convert metrics to dictionary format.

Returns:
    Dict[str, Any]: Dictionary representation of the quality metrics.

### QualityReport

Comprehensive quality report for code analysis.

Attributes:
    timestamp: When the report was generated
    project_path: Path to the analyzed project
    issues: List of all quality issues found
    metrics: Quality metrics for the analyzed code
    files_analyzed: Number of files that were analyzed
    total_issues: Total number of issues found
    errors: Number of error-level issues
    warnings: Number of warning-level issues
    infos: Number of info-level issues
    auto_fixable_issues: Number of issues that can be automatically fixed

#### Methods

##### add_issue(self: Any, issue: QualityIssue) -> None

Add a quality issue to the report.

##### get_issues_by_type(self: Any, issue_type: QualityIssueType) -> <ast.Subscript object at 0x000001942894D270>

Get all issues of a specific type.

Args:
    issue_type (QualityIssueType): The type of issues to filter by.

Returns:
    List[QualityIssue]: List of issues matching the specified type.

##### get_issues_by_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942894D4E0>

Get all issues for a specific file.

Args:
    file_path (Path): The file path to filter issues by.

Returns:
    List[QualityIssue]: List of issues for the specified file.

##### quality_score(self: Any) -> float

Calculate overall quality score (0-100).

Returns:
    float: Quality score from 0 to 100, where 100 is perfect quality.

##### to_dict(self: Any) -> <ast.Subscript object at 0x0000019427B5C370>

Convert report to dictionary format.

Returns:
    Dict[str, Any]: Dictionary representation of the quality report.

### QualityConfig

Configuration for quality checking.

Attributes:
    line_length: Maximum line length for formatting
    use_black: Whether to use Black for code formatting
    use_isort: Whether to use isort for import sorting
    use_autopep8: Whether to use autopep8 for formatting
    require_module_docstrings: Whether module docstrings are required
    require_class_docstrings: Whether class docstrings are required
    require_function_docstrings: Whether function docstrings are required
    min_docstring_length: Minimum length for docstrings
    require_return_types: Whether return type annotations are required
    require_parameter_types: Whether parameter type annotations are required
    strict_mode: Whether to use strict type checking
    max_cyclomatic_complexity: Maximum allowed cyclomatic complexity
    max_function_length: Maximum allowed function length in lines

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x0000019427B5D900>

Convert config to dictionary format.

Returns:
    Dict[str, Any]: Dictionary representation of the quality configuration.

## Constants

### FILE_PATH_KEY

Type: `str`

Value: `file_path`

### LINE_NUMBER_KEY

Type: `str`

Value: `line_number`

### COLUMN_KEY

Type: `str`

Value: `column`

### ISSUE_TYPE_KEY

Type: `str`

Value: `issue_type`

### SEVERITY_KEY

Type: `str`

Value: `severity`

### MESSAGE_KEY

Type: `str`

Value: `message`

### RULE_CODE_KEY

Type: `str`

Value: `rule_code`

### SUGGESTION_KEY

Type: `str`

Value: `suggestion`

### AUTO_FIXABLE_KEY

Type: `str`

Value: `auto_fixable`

### TOTAL_LINES_KEY

Type: `str`

Value: `total_lines`

### CODE_LINES_KEY

Type: `str`

Value: `code_lines`

### COMMENT_LINES_KEY

Type: `str`

Value: `comment_lines`

### BLANK_LINES_KEY

Type: `str`

Value: `blank_lines`

### FUNCTIONS_COUNT_KEY

Type: `str`

Value: `functions_count`

### CLASSES_COUNT_KEY

Type: `str`

Value: `classes_count`

### MODULES_COUNT_KEY

Type: `str`

Value: `modules_count`

### DOCUMENTED_FUNCTIONS_KEY

Type: `str`

Value: `documented_functions`

### DOCUMENTED_CLASSES_KEY

Type: `str`

Value: `documented_classes`

### DOCUMENTED_MODULES_KEY

Type: `str`

Value: `documented_modules`

### TYPE_ANNOTATED_FUNCTIONS_KEY

Type: `str`

Value: `type_annotated_functions`

### COMPLEXITY_SCORE_KEY

Type: `str`

Value: `complexity_score`

### MAINTAINABILITY_INDEX_KEY

Type: `str`

Value: `maintainability_index`

### DOCUMENTATION_COVERAGE_KEY

Type: `str`

Value: `documentation_coverage`

### TYPE_HINT_COVERAGE_KEY

Type: `str`

Value: `type_hint_coverage`

### TIMESTAMP_KEY

Type: `str`

Value: `timestamp`

### PROJECT_PATH_KEY

Type: `str`

Value: `project_path`

### FILES_ANALYZED_KEY

Type: `str`

Value: `files_analyzed`

### TOTAL_ISSUES_KEY

Type: `str`

Value: `total_issues`

### ERRORS_KEY

Type: `str`

Value: `errors`

### WARNINGS_KEY

Type: `str`

Value: `warnings`

### INFOS_KEY

Type: `str`

Value: `infos`

### AUTO_FIXABLE_ISSUES_KEY

Type: `str`

Value: `auto_fixable_issues`

### QUALITY_SCORE_KEY

Type: `str`

Value: `quality_score`

### METRICS_KEY

Type: `str`

Value: `metrics`

### ISSUES_KEY

Type: `str`

Value: `issues`

### FORMATTING_KEY

Type: `str`

Value: `formatting`

### DOCUMENTATION_KEY

Type: `str`

Value: `documentation`

### TYPE_HINTS_KEY

Type: `str`

Value: `type_hints`

### COMPLEXITY_KEY

Type: `str`

Value: `complexity`

### FORMATTING

Type: `str`

Value: `formatting`

### STYLE

Type: `str`

Value: `style`

### DOCUMENTATION

Type: `str`

Value: `documentation`

### TYPE_HINTS

Type: `str`

Value: `type_hints`

### COMPLEXITY

Type: `str`

Value: `complexity`

### IMPORTS

Type: `str`

Value: `imports`

### NAMING

Type: `str`

Value: `naming`

### ERROR

Type: `str`

Value: `error`

### WARNING

Type: `str`

Value: `warning`

### INFO

Type: `str`

Value: `info`

