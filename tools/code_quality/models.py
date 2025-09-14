"""
Data models for code quality checking system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Dictionary key constants
FILE_PATH_KEY = "file_path"
LINE_NUMBER_KEY = "line_number"
COLUMN_KEY = "column"
ISSUE_TYPE_KEY = "issue_type"
SEVERITY_KEY = "severity"
MESSAGE_KEY = "message"
RULE_CODE_KEY = "rule_code"
SUGGESTION_KEY = "suggestion"
AUTO_FIXABLE_KEY = "auto_fixable"

TOTAL_LINES_KEY = "total_lines"
CODE_LINES_KEY = "code_lines"
COMMENT_LINES_KEY = "comment_lines"
BLANK_LINES_KEY = "blank_lines"
FUNCTIONS_COUNT_KEY = "functions_count"
CLASSES_COUNT_KEY = "classes_count"
MODULES_COUNT_KEY = "modules_count"
DOCUMENTED_FUNCTIONS_KEY = "documented_functions"
DOCUMENTED_CLASSES_KEY = "documented_classes"
DOCUMENTED_MODULES_KEY = "documented_modules"
TYPE_ANNOTATED_FUNCTIONS_KEY = "type_annotated_functions"
COMPLEXITY_SCORE_KEY = "complexity_score"
MAINTAINABILITY_INDEX_KEY = "maintainability_index"
DOCUMENTATION_COVERAGE_KEY = "documentation_coverage"
TYPE_HINT_COVERAGE_KEY = "type_hint_coverage"

TIMESTAMP_KEY = "timestamp"
PROJECT_PATH_KEY = "project_path"
FILES_ANALYZED_KEY = "files_analyzed"
TOTAL_ISSUES_KEY = "total_issues"
ERRORS_KEY = "errors"
WARNINGS_KEY = "warnings"
INFOS_KEY = "infos"
AUTO_FIXABLE_ISSUES_KEY = "auto_fixable_issues"
QUALITY_SCORE_KEY = "quality_score"
METRICS_KEY = "metrics"
ISSUES_KEY = "issues"

FORMATTING_KEY = "formatting"
DOCUMENTATION_KEY = "documentation"
TYPE_HINTS_KEY = "type_hints"
COMPLEXITY_KEY = "complexity"


class QualityIssueType(Enum):
    """Types of quality issues that can be detected.

    Attributes:
        FORMATTING: Code formatting issues (spacing, indentation, etc.)
        STYLE: Code style issues (naming conventions, etc.)
        DOCUMENTATION: Missing or inadequate documentation
        TYPE_HINTS: Missing or incorrect type hints
        COMPLEXITY: Code complexity issues
        IMPORTS: Import organization and usage issues
        NAMING: Variable and function naming issues
    """

    FORMATTING = "formatting"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TYPE_HINTS = "type_hints"
    COMPLEXITY = "complexity"
    IMPORTS = "imports"
    NAMING = "naming"


class QualitySeverity(Enum):
    """Severity levels for quality issues.

    Attributes:
        ERROR: Critical issues that must be fixed
        WARNING: Important issues that should be addressed
        INFO: Minor issues or suggestions for improvement
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class QualityIssue:
    """Represents a single code quality issue.

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
    """

    file_path: Path
    line_number: int
    column: int
    issue_type: QualityIssueType
    severity: QualitySeverity
    message: str
    rule_code: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the quality issue.
        """
        return {
            FILE_PATH_KEY: str(self.file_path),
            LINE_NUMBER_KEY: self.line_number,
            COLUMN_KEY: self.column,
            ISSUE_TYPE_KEY: self.issue_type.value,
            SEVERITY_KEY: self.severity.value,
            MESSAGE_KEY: self.message,
            RULE_CODE_KEY: self.rule_code,
            SUGGESTION_KEY: self.suggestion,
            AUTO_FIXABLE_KEY: self.auto_fixable,
        }


@dataclass
class QualityMetrics:
    """Code quality metrics for a file or project.

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
    """

    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    functions_count: int = 0
    classes_count: int = 0
    modules_count: int = 0
    documented_functions: int = 0
    documented_classes: int = 0
    documented_modules: int = 0
    type_annotated_functions: int = 0
    complexity_score: float = 0.0
    maintainability_index: float = 0.0

    @property
    def documentation_coverage(self) -> float:
        """Calculate documentation coverage percentage.

        Returns:
            float: Documentation coverage as a percentage (0-100).
        """
        total_items = self.functions_count + self.classes_count + self.modules_count
        if total_items == 0:
            return 100.0
        documented_items = (
            self.documented_functions
            + self.documented_classes
            + self.documented_modules
        )
        return (documented_items / total_items) * 100.0

    @property
    def type_hint_coverage(self) -> float:
        """Calculate type hint coverage percentage.

        Returns:
            float: Type hint coverage as a percentage (0-100).
        """
        if self.functions_count == 0:
            return 100.0
        return (self.type_annotated_functions / self.functions_count) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the quality metrics.
        """
        return {
            TOTAL_LINES_KEY: self.total_lines,
            CODE_LINES_KEY: self.code_lines,
            COMMENT_LINES_KEY: self.comment_lines,
            BLANK_LINES_KEY: self.blank_lines,
            FUNCTIONS_COUNT_KEY: self.functions_count,
            CLASSES_COUNT_KEY: self.classes_count,
            MODULES_COUNT_KEY: self.modules_count,
            DOCUMENTED_FUNCTIONS_KEY: self.documented_functions,
            DOCUMENTED_CLASSES_KEY: self.documented_classes,
            DOCUMENTED_MODULES_KEY: self.documented_modules,
            TYPE_ANNOTATED_FUNCTIONS_KEY: self.type_annotated_functions,
            COMPLEXITY_SCORE_KEY: self.complexity_score,
            MAINTAINABILITY_INDEX_KEY: self.maintainability_index,
            DOCUMENTATION_COVERAGE_KEY: self.documentation_coverage,
            TYPE_HINT_COVERAGE_KEY: self.type_hint_coverage,
        }


@dataclass
class QualityReport:
    """Comprehensive quality report for code analysis.

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
    """

    timestamp: datetime = field(default_factory=datetime.now)
    project_path: Path = field(default_factory=lambda: Path.cwd())
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    files_analyzed: int = 0
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    auto_fixable_issues: int = 0

    def add_issue(self, issue: QualityIssue) -> None:
        """Add a quality issue to the report."""
        self.issues.append(issue)
        self.total_issues += 1

        if issue.severity == QualitySeverity.ERROR:
            self.errors += 1
        elif issue.severity == QualitySeverity.WARNING:
            self.warnings += 1
        else:
            self.infos += 1

        if issue.auto_fixable:
            self.auto_fixable_issues += 1

    def get_issues_by_type(self, issue_type: QualityIssueType) -> List[QualityIssue]:
        """Get all issues of a specific type.

        Args:
            issue_type (QualityIssueType): The type of issues to filter by.

        Returns:
            List[QualityIssue]: List of issues matching the specified type.
        """
        return [issue for issue in self.issues if issue.issue_type == issue_type]

    def get_issues_by_file(self, file_path: Path) -> List[QualityIssue]:
        """Get all issues for a specific file.

        Args:
            file_path (Path): The file path to filter issues by.

        Returns:
            List[QualityIssue]: List of issues for the specified file.
        """
        return [issue for issue in self.issues if issue.file_path == file_path]

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-100).

        Returns:
            float: Quality score from 0 to 100, where 100 is perfect quality.
        """
        if self.files_analyzed == 0:
            return 100.0

        # Base score starts at 100
        score = 100.0

        # Deduct points for issues
        error_penalty = self.errors * 5.0
        warning_penalty = self.warnings * 2.0
        info_penalty = self.infos * 0.5

        total_penalty = error_penalty + warning_penalty + info_penalty

        # Normalize penalty by number of files
        normalized_penalty = total_penalty / self.files_analyzed

        score = max(0.0, score - normalized_penalty)

        # Bonus for good documentation and type hints
        doc_bonus = (self.metrics.documentation_coverage / 100.0) * 5.0
        type_bonus = (self.metrics.type_hint_coverage / 100.0) * 5.0

        return min(100.0, score + doc_bonus + type_bonus)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the quality report.
        """
        return {
            TIMESTAMP_KEY: self.timestamp.isoformat(),
            PROJECT_PATH_KEY: str(self.project_path),
            FILES_ANALYZED_KEY: self.files_analyzed,
            TOTAL_ISSUES_KEY: self.total_issues,
            ERRORS_KEY: self.errors,
            WARNINGS_KEY: self.warnings,
            INFOS_KEY: self.infos,
            AUTO_FIXABLE_ISSUES_KEY: self.auto_fixable_issues,
            QUALITY_SCORE_KEY: self.quality_score,
            METRICS_KEY: self.metrics.to_dict(),
            ISSUES_KEY: [issue.to_dict() for issue in self.issues],
        }


@dataclass
class QualityConfig:
    """Configuration for quality checking.

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
    """

    # Formatting settings
    line_length: int = 88
    use_black: bool = True
    use_isort: bool = True
    use_autopep8: bool = False

    # Documentation settings
    require_module_docstrings: bool = True
    require_class_docstrings: bool = True
    require_function_docstrings: bool = True
    min_docstring_length: int = 10

    # Type hint settings
    require_return_types: bool = True
    require_parameter_types: bool = True
    strict_mode: bool = False

    # Complexity settings
    max_cyclomatic_complexity: int = 10
    max_function_length: int = 50
    max_class_length: int = 500
    max_module_length: int = 1000

    # Style settings
    max_line_length: int = 88
    ignore_rules: List[str] = field(default_factory=list)

    # File patterns
    include_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["__pycache__", "*.pyc", ".git"]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the quality configuration.
        """
        return {
            FORMATTING_KEY: {
                "line_length": self.line_length,
                "use_black": self.use_black,
                "use_isort": self.use_isort,
                "use_autopep8": self.use_autopep8,
            },
            DOCUMENTATION_KEY: {
                "require_module_docstrings": self.require_module_docstrings,
                "require_class_docstrings": self.require_class_docstrings,
                "require_function_docstrings": self.require_function_docstrings,
                "min_docstring_length": self.min_docstring_length,
            },
            TYPE_HINTS_KEY: {
                "require_return_types": self.require_return_types,
                "require_parameter_types": self.require_parameter_types,
                "strict_mode": self.strict_mode,
            },
            COMPLEXITY_KEY: {
                "max_cyclomatic_complexity": self.max_cyclomatic_complexity,
                "max_function_length": self.max_function_length,
                "max_class_length": self.max_class_length,
                "max_module_length": self.max_module_length,
            },
            "style": {
                "max_line_length": self.max_line_length,
                "ignore_rules": self.ignore_rules,
            },
            "files": {
                "include_patterns": self.include_patterns,
                "exclude_patterns": self.exclude_patterns,
            },
        }
