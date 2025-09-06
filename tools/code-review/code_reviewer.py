"""
Code Review and Refactoring Assistance System

This module provides automated code review suggestions, refactoring recommendations,
and technical debt tracking to improve code quality and maintainability.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
import json
import subprocess
from datetime import datetime


class ReviewSeverity(Enum):
    """Severity levels for code review issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    """Categories of code review issues"""
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ARCHITECTURE = "architecture"


@dataclass
class CodeIssue:
    """Represents a code review issue"""
    file_path: str
    line_number: int
    column: int
    severity: ReviewSeverity
    category: IssueCategory
    message: str
    suggestion: str
    rule_id: str
    context: str = ""
    fix_effort: str = "medium"  # low, medium, high
    technical_debt_score: int = 1


@dataclass
class RefactoringRecommendation:
    """Represents a refactoring recommendation"""
    file_path: str
    start_line: int
    end_line: int
    recommendation_type: str
    description: str
    benefits: List[str]
    effort_estimate: str
    priority: int
    code_before: str = ""
    code_after: str = ""


@dataclass
class TechnicalDebtItem:
    """Represents a technical debt item"""
    id: str
    file_path: str
    description: str
    category: IssueCategory
    severity: ReviewSeverity
    estimated_effort: str
    business_impact: str
    created_date: datetime
    priority_score: float
    related_issues: List[str] = field(default_factory=list)


class CodeReviewer:
    """Main code review and refactoring assistance system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues: List[CodeIssue] = []
        self.recommendations: List[RefactoringRecommendation] = []
        self.technical_debt: List[TechnicalDebtItem] = []
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize analyzers
        self.complexity_analyzer = ComplexityAnalyzer()
        self.maintainability_analyzer = MaintainabilityAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load code review configuration"""
        config_path = self.project_root / "tools" / "code-review" / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
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
    
    def review_file(self, file_path: str) -> List[CodeIssue]:
        """Review a single file and return issues"""
        if not file_path.endswith('.py'):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            issues = []
            issues.extend(self.complexity_analyzer.analyze(file_path, tree, content))
            issues.extend(self.maintainability_analyzer.analyze(file_path, tree, content))
            issues.extend(self.performance_analyzer.analyze(file_path, tree, content))
            issues.extend(self.security_analyzer.analyze(file_path, tree, content))
            
            return issues
            
        except Exception as e:
            return [CodeIssue(
                file_path=file_path,
                line_number=1,
                column=1,
                severity=ReviewSeverity.HIGH,
                category=IssueCategory.MAINTAINABILITY,
                message=f"Failed to parse file: {str(e)}",
                suggestion="Fix syntax errors or encoding issues",
                rule_id="PARSE_ERROR"
            )]
    
    def review_project(self, include_patterns: List[str] = None) -> Dict[str, Any]:
        """Review entire project"""
        if include_patterns is None:
            include_patterns = ["**/*.py"]
        
        all_issues = []
        all_recommendations = []
        
        for pattern in include_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    issues = self.review_file(str(file_path))
                    all_issues.extend(issues)
                    
                    # Generate refactoring recommendations
                    recommendations = self._generate_refactoring_recommendations(str(file_path), issues)
                    all_recommendations.extend(recommendations)
        
        self.issues = all_issues
        self.recommendations = all_recommendations
        
        # Update technical debt tracking
        self._update_technical_debt()
        
        return {
            "issues": len(all_issues),
            "recommendations": len(all_recommendations),
            "technical_debt_items": len(self.technical_debt),
            "summary": self._generate_summary()
        }
    
    def _generate_refactoring_recommendations(self, file_path: str, issues: List[CodeIssue]) -> List[RefactoringRecommendation]:
        """Generate refactoring recommendations based on issues"""
        recommendations = []
        
        # Group issues by type to generate targeted recommendations
        complexity_issues = [i for i in issues if i.category == IssueCategory.COMPLEXITY]
        maintainability_issues = [i for i in issues if i.category == IssueCategory.MAINTAINABILITY]
        
        # Complex function refactoring
        for issue in complexity_issues:
            if "high complexity" in issue.message.lower():
                recommendations.append(RefactoringRecommendation(
                    file_path=file_path,
                    start_line=issue.line_number,
                    end_line=issue.line_number + 10,  # Estimate
                    recommendation_type="extract_method",
                    description=f"Extract complex logic into smaller methods",
                    benefits=[
                        "Improved readability",
                        "Better testability",
                        "Reduced complexity"
                    ],
                    effort_estimate="medium",
                    priority=3
                ))
        
        # Long method refactoring
        for issue in maintainability_issues:
            if "long method" in issue.message.lower():
                recommendations.append(RefactoringRecommendation(
                    file_path=file_path,
                    start_line=issue.line_number,
                    end_line=issue.line_number + 20,  # Estimate
                    recommendation_type="split_method",
                    description="Split long method into smaller, focused methods",
                    benefits=[
                        "Improved maintainability",
                        "Better code organization",
                        "Easier debugging"
                    ],
                    effort_estimate="high",
                    priority=2
                ))
        
        return recommendations
    
    def _update_technical_debt(self):
        """Update technical debt tracking based on current issues"""
        debt_items = []
        
        # Group issues by file and severity
        file_issues = {}
        for issue in self.issues:
            if issue.file_path not in file_issues:
                file_issues[issue.file_path] = []
            file_issues[issue.file_path].append(issue)
        
        # Create technical debt items for high-impact issues
        for file_path, issues in file_issues.items():
            critical_issues = [i for i in issues if i.severity == ReviewSeverity.CRITICAL]
            high_issues = [i for i in issues if i.severity == ReviewSeverity.HIGH]
            
            if critical_issues or len(high_issues) > 3:
                debt_item = TechnicalDebtItem(
                    id=f"debt_{hash(file_path)}_{datetime.now().strftime('%Y%m%d')}",
                    file_path=file_path,
                    description=f"File has {len(critical_issues)} critical and {len(high_issues)} high severity issues",
                    category=IssueCategory.MAINTAINABILITY,
                    severity=ReviewSeverity.HIGH if critical_issues else ReviewSeverity.MEDIUM,
                    estimated_effort="high" if critical_issues else "medium",
                    business_impact="High maintenance cost, potential bugs",
                    created_date=datetime.now(),
                    priority_score=self._calculate_priority_score(issues),
                    related_issues=[i.rule_id for i in critical_issues + high_issues]
                )
                debt_items.append(debt_item)
        
        self.technical_debt = debt_items
    
    def _calculate_priority_score(self, issues: List[CodeIssue]) -> float:
        """Calculate priority score for technical debt"""
        score = 0.0
        
        for issue in issues:
            if issue.severity == ReviewSeverity.CRITICAL:
                score += 10
            elif issue.severity == ReviewSeverity.HIGH:
                score += 5
            elif issue.severity == ReviewSeverity.MEDIUM:
                score += 2
            else:
                score += 1
        
        return score
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate review summary"""
        severity_counts = {}
        category_counts = {}
        
        for issue in self.issues:
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
            category_counts[issue.category.value] = category_counts.get(issue.category.value, 0) + 1
        
        return {
            "total_issues": len(self.issues),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "refactoring_recommendations": len(self.recommendations),
            "technical_debt_score": sum(item.priority_score for item in self.technical_debt),
            "files_reviewed": len(set(issue.file_path for issue in self.issues))
        }
    
    def generate_report(self, output_path: str = "code_review_report.json"):
        """Generate comprehensive code review report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "summary": self._generate_summary(),
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "column": issue.column,
                    "severity": issue.severity.value,
                    "category": issue.category.value,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "rule_id": issue.rule_id,
                    "context": issue.context,
                    "fix_effort": issue.fix_effort,
                    "technical_debt_score": issue.technical_debt_score
                }
                for issue in self.issues
            ],
            "refactoring_recommendations": [
                {
                    "file_path": rec.file_path,
                    "start_line": rec.start_line,
                    "end_line": rec.end_line,
                    "type": rec.recommendation_type,
                    "description": rec.description,
                    "benefits": rec.benefits,
                    "effort_estimate": rec.effort_estimate,
                    "priority": rec.priority
                }
                for rec in self.recommendations
            ],
            "technical_debt": [
                {
                    "id": debt.id,
                    "file_path": debt.file_path,
                    "description": debt.description,
                    "category": debt.category.value,
                    "severity": debt.severity.value,
                    "estimated_effort": debt.estimated_effort,
                    "business_impact": debt.business_impact,
                    "created_date": debt.created_date.isoformat(),
                    "priority_score": debt.priority_score,
                    "related_issues": debt.related_issues
                }
                for debt in self.technical_debt
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


class ComplexityAnalyzer:
    """Analyzes code complexity"""
    
    def analyze(self, file_path: str, tree: ast.AST, content: str) -> List[CodeIssue]:
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity=ReviewSeverity.HIGH if complexity > 15 else ReviewSeverity.MEDIUM,
                        category=IssueCategory.COMPLEXITY,
                        message=f"Function '{node.name}' has high complexity ({complexity})",
                        suggestion="Consider breaking this function into smaller, more focused functions",
                        rule_id="HIGH_COMPLEXITY",
                        technical_debt_score=complexity // 5
                    ))
        
        return issues
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity


class MaintainabilityAnalyzer:
    """Analyzes code maintainability"""
    
    def analyze(self, file_path: str, tree: ast.AST, content: str) -> List[CodeIssue]:
        issues = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function length
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 20
                if func_lines > 50:
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity=ReviewSeverity.MEDIUM,
                        category=IssueCategory.MAINTAINABILITY,
                        message=f"Function '{node.name}' is too long ({func_lines} lines)",
                        suggestion="Consider splitting this function into smaller functions",
                        rule_id="LONG_FUNCTION",
                        technical_debt_score=func_lines // 25
                    ))
                
                # Check for missing docstring
                if not ast.get_docstring(node):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity=ReviewSeverity.LOW,
                        category=IssueCategory.DOCUMENTATION,
                        message=f"Function '{node.name}' is missing a docstring",
                        suggestion="Add a docstring explaining the function's purpose, parameters, and return value",
                        rule_id="MISSING_DOCSTRING",
                        technical_debt_score=1
                    ))
        
        return issues


class PerformanceAnalyzer:
    """Analyzes potential performance issues"""
    
    def analyze(self, file_path: str, tree: ast.AST, content: str) -> List[CodeIssue]:
        issues = []
        
        for node in ast.walk(tree):
            # Check for nested loops
            if isinstance(node, (ast.For, ast.While)):
                nested_loops = self._count_nested_loops(node)
                if nested_loops > 2:
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity=ReviewSeverity.MEDIUM,
                        category=IssueCategory.PERFORMANCE,
                        message=f"Deeply nested loops detected ({nested_loops} levels)",
                        suggestion="Consider optimizing algorithm or using more efficient data structures",
                        rule_id="NESTED_LOOPS",
                        technical_debt_score=nested_loops
                    ))
        
        return issues
    
    def _count_nested_loops(self, node: ast.AST) -> int:
        """Count nested loop levels"""
        max_depth = 0
        
        def count_depth(n, current_depth=0):
            nonlocal max_depth
            if isinstance(n, (ast.For, ast.While)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(n):
                count_depth(child, current_depth)
        
        count_depth(node)
        return max_depth


class SecurityAnalyzer:
    """Analyzes potential security issues"""
    
    def analyze(self, file_path: str, tree: ast.AST, content: str) -> List[CodeIssue]:
        issues = []
        
        # Check for dangerous function calls
        dangerous_patterns = [
            ('eval', 'Use of eval() can execute arbitrary code'),
            ('exec', 'Use of exec() can execute arbitrary code'),
            ('subprocess.call', 'Direct subprocess calls may be vulnerable to injection'),
            ('os.system', 'Use of os.system() may be vulnerable to injection')
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                for pattern, message in dangerous_patterns:
                    if pattern in func_name:
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            severity=ReviewSeverity.HIGH,
                            category=IssueCategory.SECURITY,
                            message=f"Potentially dangerous function call: {message}",
                            suggestion="Use safer alternatives or implement proper input validation",
                            rule_id="DANGEROUS_FUNCTION",
                            technical_debt_score=5
                        ))
        
        return issues
    
    def _get_function_name(self, node: ast.AST) -> str:
        """Get function name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_function_name(node.value)}.{node.attr}"
        return ""