"""
Demonstration of the code quality checking system.
"""

import ast
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional


# Simplified models for demonstration
class QualityIssueType(Enum):
    FORMATTING = "formatting"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TYPE_HINTS = "type_hints"
    COMPLEXITY = "complexity"


class QualitySeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class QualityIssue:
    file_path: Path
    line_number: int
    column: int
    issue_type: QualityIssueType
    severity: QualitySeverity
    message: str
    rule_code: str
    suggestion: Optional[str] = None


@dataclass
class QualityMetrics:
    total_lines: int = 0
    functions_count: int = 0
    classes_count: int = 0
    documented_functions: int = 0
    documented_classes: int = 0
    complexity_score: float = 0.0
    
    @property
    def documentation_coverage(self) -> float:
        total_items = self.functions_count + self.classes_count
        if total_items == 0:
            return 100.0
        documented_items = self.documented_functions + self.documented_classes
        return (documented_items / total_items) * 100.0


@dataclass
class QualityReport:
    timestamp: datetime
    project_path: Path
    issues: List[QualityIssue]
    metrics: QualityMetrics
    files_analyzed: int = 0
    
    @property
    def total_issues(self) -> int:
        return len(self.issues)
    
    @property
    def errors(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == QualitySeverity.ERROR)
    
    @property
    def warnings(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == QualitySeverity.WARNING)
    
    @property
    def quality_score(self) -> float:
        if self.files_analyzed == 0:
            return 100.0
        
        # Simple scoring algorithm
        base_score = 100.0
        error_penalty = self.errors * 10.0
        warning_penalty = self.warnings * 5.0
        
        score = base_score - error_penalty - warning_penalty
        
        # Bonus for good documentation
        doc_bonus = (self.metrics.documentation_coverage / 100.0) * 10.0
        
        return max(0.0, min(100.0, score + doc_bonus))


class SimpleQualityChecker:
    """Simplified quality checker for demonstration."""
    
    def check_quality(self, file_path: Path) -> QualityReport:
        """Check quality of a Python file."""
        issues = []
        metrics = QualityMetrics()
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic metrics
        lines = content.split('\n')
        metrics.total_lines = len(lines)
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            issues.append(QualityIssue(
                file_path=file_path,
                line_number=e.lineno or 1,
                column=e.offset or 1,
                issue_type=QualityIssueType.STYLE,
                severity=QualitySeverity.ERROR,
                message=f"Syntax error: {e.msg}",
                rule_code="SYNTAX_ERROR"
            ))
            tree = None
        
        if tree:
            # Analyze functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics.functions_count += 1
                    
                    # Check for docstring
                    docstring = ast.get_docstring(node)
                    if docstring:
                        metrics.documented_functions += 1
                    else:
                        issues.append(QualityIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            issue_type=QualityIssueType.DOCUMENTATION,
                            severity=QualitySeverity.WARNING,
                            message=f"Function '{node.name}' missing docstring",
                            rule_code="MISSING_DOCSTRING",
                            suggestion="Add docstring describing function purpose"
                        ))
                    
                    # Check complexity (simplified)
                    complexity = self._calculate_complexity(node)
                    if complexity > 10:
                        issues.append(QualityIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            issue_type=QualityIssueType.COMPLEXITY,
                            severity=QualitySeverity.WARNING,
                            message=f"Function '{node.name}' too complex (complexity {complexity})",
                            rule_code="HIGH_COMPLEXITY",
                            suggestion="Consider breaking function into smaller functions"
                        ))
                    
                    metrics.complexity_score += complexity
                
                elif isinstance(node, ast.ClassDef):
                    metrics.classes_count += 1
                    
                    # Check for docstring
                    docstring = ast.get_docstring(node)
                    if docstring:
                        metrics.documented_classes += 1
                    else:
                        issues.append(QualityIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            issue_type=QualityIssueType.DOCUMENTATION,
                            severity=QualitySeverity.WARNING,
                            message=f"Class '{node.name}' missing docstring",
                            rule_code="MISSING_DOCSTRING",
                            suggestion="Add docstring describing class purpose"
                        ))
            
            # Average complexity
            if metrics.functions_count > 0:
                metrics.complexity_score /= metrics.functions_count
        
        # Check basic formatting issues
        for line_num, line in enumerate(lines, 1):
            if len(line) > 88:
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=89,
                    issue_type=QualityIssueType.FORMATTING,
                    severity=QualitySeverity.WARNING,
                    message=f"Line too long ({len(line)} > 88 characters)",
                    rule_code="LINE_TOO_LONG",
                    suggestion="Break line into multiple lines"
                ))
            
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=len(line.rstrip()) + 1,
                    issue_type=QualityIssueType.FORMATTING,
                    severity=QualitySeverity.INFO,
                    message="Trailing whitespace",
                    rule_code="TRAILING_WHITESPACE",
                    suggestion="Remove trailing whitespace"
                ))
        
        return QualityReport(
            timestamp=datetime.now(),
            project_path=file_path,
            issues=issues,
            metrics=metrics,
            files_analyzed=1
        )
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        
        return complexity


def demonstrate_quality_checking():
    """Demonstrate the quality checking system."""
    print("🔍 Code Quality Checking System Demo")
    print("=" * 50)
    
    # Create test files with different quality levels
    test_files = {
        "good_quality.py": '''
"""High-quality module with proper documentation."""

from typing import List


def calculate_average(numbers: List[float]) -> float:
    """
    Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers to average
        
    Returns:
        The average value
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


class Calculator:
    """A simple calculator class."""
    
    def __init__(self) -> None:
        """Initialize the calculator."""
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
''',
        
        "poor_quality.py": '''
def bad_function(x,y,z,a,b,c):
    if x:
        if y:
            if z:
                if a:
                    if b:
                        if c:
                            return "deeply nested and no docs"
    return None

class BadClass:
    def method1(self):
        pass
    def method2(self):
        pass
    def method3(self):
        pass
''',
        
        "syntax_error.py": '''
def broken_function(
    print("This has a syntax error"
    return "missing parenthesis"
'''
    }
    
    checker = SimpleQualityChecker()
    
    for filename, content in test_files.items():
        print(f"\n📄 Analyzing {filename}")
        print("-" * 30)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = Path(f.name)
        
        try:
            # Analyze the file
            report = checker.check_quality(temp_file)
            
            # Display results
            print(f"Quality Score: {report.quality_score:.1f}/100")
            print(f"Total Issues: {report.total_issues}")
            print(f"  Errors: {report.errors}")
            print(f"  Warnings: {report.warnings}")
            print(f"Documentation Coverage: {report.metrics.documentation_coverage:.1f}%")
            print(f"Average Complexity: {report.metrics.complexity_score:.1f}")
            
            if report.issues:
                print("\nTop Issues:")
                for i, issue in enumerate(report.issues[:3], 1):
                    print(f"  {i}. Line {issue.line_number}: {issue.message}")
                    if issue.suggestion:
                        print(f"     💡 {issue.suggestion}")
            
            # Quality assessment
            if report.quality_score >= 90:
                print("🟢 Excellent code quality!")
            elif report.quality_score >= 75:
                print("🟡 Good code quality")
            elif report.quality_score >= 50:
                print("🟠 Needs improvement")
            else:
                print("🔴 Poor code quality")
        
        finally:
            temp_file.unlink(missing_ok=True)
    
    print("\n✅ Demo completed successfully!")
    print("\nThe Code Quality Checking System can:")
    print("• Detect missing documentation")
    print("• Identify complex functions")
    print("• Find formatting issues")
    print("• Calculate quality metrics")
    print("• Provide improvement suggestions")
    print("• Generate comprehensive reports")


if __name__ == "__main__":
    demonstrate_quality_checking()