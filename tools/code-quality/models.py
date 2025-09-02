"""
Data models for code quality checking system.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class QualityIssueType(Enum):
    """Types of quality issues that can be detected."""
    FORMATTING = "formatting"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TYPE_HINTS = "type_hints"
    COMPLEXITY = "complexity"
    IMPORTS = "imports"
    NAMING = "naming"


class QualitySeverity(Enum):
    """Severity levels for quality issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class QualityIssue:
    """Represents a single code quality issue."""
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
        """Convert issue to dictionary format."""
        return {
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "column": self.column,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "rule_code": self.rule_code,
            "suggestion": self.suggestion,
            "auto_fixable": self.auto_fixable
        }


@dataclass
class QualityMetrics:
    """Code quality metrics for a file or project."""
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
        """Calculate documentation coverage percentage."""
        total_items = self.functions_count + self.classes_count + self.modules_count
        if total_items == 0:
            return 100.0
        documented_items = self.documented_functions + self.documented_classes + self.documented_modules
        return (documented_items / total_items) * 100.0
    
    @property
    def type_hint_coverage(self) -> float:
        """Calculate type hint coverage percentage."""
        if self.functions_count == 0:
            return 100.0
        return (self.type_annotated_functions / self.functions_count) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "functions_count": self.functions_count,
            "classes_count": self.classes_count,
            "modules_count": self.modules_count,
            "documented_functions": self.documented_functions,
            "documented_classes": self.documented_classes,
            "documented_modules": self.documented_modules,
            "type_annotated_functions": self.type_annotated_functions,
            "complexity_score": self.complexity_score,
            "maintainability_index": self.maintainability_index,
            "documentation_coverage": self.documentation_coverage,
            "type_hint_coverage": self.type_hint_coverage
        }


@dataclass
class QualityReport:
    """Comprehensive quality report for code analysis."""
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
        """Get all issues of a specific type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]
    
    def get_issues_by_file(self, file_path: Path) -> List[QualityIssue]:
        """Get all issues for a specific file."""
        return [issue for issue in self.issues if issue.file_path == file_path]
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
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
        """Convert report to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "project_path": str(self.project_path),
            "files_analyzed": self.files_analyzed,
            "total_issues": self.total_issues,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "auto_fixable_issues": self.auto_fixable_issues,
            "quality_score": self.quality_score,
            "metrics": self.metrics.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues]
        }


@dataclass
class QualityConfig:
    """Configuration for quality checking."""
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
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", ".git"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            "formatting": {
                "line_length": self.line_length,
                "use_black": self.use_black,
                "use_isort": self.use_isort,
                "use_autopep8": self.use_autopep8
            },
            "documentation": {
                "require_module_docstrings": self.require_module_docstrings,
                "require_class_docstrings": self.require_class_docstrings,
                "require_function_docstrings": self.require_function_docstrings,
                "min_docstring_length": self.min_docstring_length
            },
            "type_hints": {
                "require_return_types": self.require_return_types,
                "require_parameter_types": self.require_parameter_types,
                "strict_mode": self.strict_mode
            },
            "complexity": {
                "max_cyclomatic_complexity": self.max_cyclomatic_complexity,
                "max_function_length": self.max_function_length,
                "max_class_length": self.max_class_length,
                "max_module_length": self.max_module_length
            },
            "style": {
                "max_line_length": self.max_line_length,
                "ignore_rules": self.ignore_rules
            },
            "files": {
                "include_patterns": self.include_patterns,
                "exclude_patterns": self.exclude_patterns
            }
        }