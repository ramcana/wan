"""
Type hint validation and enforcement.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
import logging

from ..models import QualityIssue, QualityIssueType, QualitySeverity, QualityConfig


logger = logging.getLogger(__name__)


class TypeHintValidator:
    """Validates type hints for functions and methods."""
    
    def __init__(self, config: QualityConfig):
        """Initialize validator with configuration."""
        self.config = config
    
    def validate_type_hints(self, file_path: Path, tree: ast.AST) -> tuple[List[QualityIssue], Dict[str, Any]]:
        """
        Validate type hints in the given AST.
        
        Returns:
            Tuple of (issues, metrics)
        """
        issues = []
        metrics = {
            'total_functions': 0,
            'annotated_functions': 0,
            'missing_return_annotations': 0,
            'missing_parameter_annotations': 0
        }
        
        # First, run mypy if available for comprehensive type checking
        mypy_issues = self._run_mypy_check(file_path)
        issues.extend(mypy_issues)
        
        # Walk through AST for manual type hint validation
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_issues, func_metrics = self._validate_function_type_hints(file_path, node)
                issues.extend(func_issues)
                
                metrics['total_functions'] += 1
                if func_metrics['has_annotations']:
                    metrics['annotated_functions'] += 1
                if func_metrics['missing_return_annotation']:
                    metrics['missing_return_annotations'] += 1
                metrics['missing_parameter_annotations'] += func_metrics['missing_parameter_annotations']
        
        return issues, metrics
    
    def _validate_function_type_hints(self, file_path: Path, node: ast.FunctionDef) -> tuple[List[QualityIssue], Dict[str, Any]]:
        """Validate type hints for a specific function."""
        issues = []
        metrics = {
            'has_annotations': False,
            'missing_return_annotation': False,
            'missing_parameter_annotations': 0
        }
        
        # Skip private functions and special methods if not in strict mode
        if not self.config.strict_mode and node.name.startswith('_'):
            return issues, metrics
        
        # Check return type annotation
        if self.config.require_return_types:
            if node.returns is None and not self._is_void_function(node):
                metrics['missing_return_annotation'] = True
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type=QualityIssueType.TYPE_HINTS,
                    severity=QualitySeverity.WARNING,
                    message=f"Function '{node.name}' missing return type annotation",
                    rule_code="MISSING_RETURN_TYPE",
                    suggestion="Add return type annotation (e.g., -> int, -> str, -> None)"
                )
                issues.append(issue)
        
        # Check parameter type annotations
        if self.config.require_parameter_types:
            for arg in node.args.args:
                # Skip 'self' and 'cls' parameters
                if arg.arg in ['self', 'cls']:
                    continue
                
                if arg.annotation is None:
                    metrics['missing_parameter_annotations'] += 1
                    issue = QualityIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type=QualityIssueType.TYPE_HINTS,
                        severity=QualitySeverity.WARNING,
                        message=f"Parameter '{arg.arg}' in function '{node.name}' missing type annotation",
                        rule_code="MISSING_PARAM_TYPE",
                        suggestion=f"Add type annotation for parameter '{arg.arg}' (e.g., {arg.arg}: int)"
                    )
                    issues.append(issue)
                else:
                    metrics['has_annotations'] = True
        
        # Check for inconsistent type annotations
        inconsistent_issues = self._check_inconsistent_annotations(file_path, node)
        issues.extend(inconsistent_issues)
        
        # Check for complex type annotations that could be simplified
        complex_issues = self._check_complex_annotations(file_path, node)
        issues.extend(complex_issues)
        
        return issues, metrics
    
    def _run_mypy_check(self, file_path: Path) -> List[QualityIssue]:
        """Run mypy type checker on the file."""
        issues = []
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'mypy',
                '--show-error-codes',
                '--no-error-summary',
                str(file_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0 and result.stdout:
                # Parse mypy output
                for line in result.stdout.strip().split('\n'):
                    if ':' in line and 'error:' in line:
                        mypy_issue = self._parse_mypy_output(file_path, line)
                        if mypy_issue:
                            issues.append(mypy_issue)
        
        except FileNotFoundError:
            logger.debug("mypy not found, skipping mypy type checking")
        except Exception as e:
            logger.error(f"Error running mypy on {file_path}: {e}")
        
        return issues
    
    def _parse_mypy_output(self, file_path: Path, line: str) -> QualityIssue:
        """Parse mypy output line into QualityIssue."""
        try:
            # Format: file.py:line:column: error: message [error-code]
            parts = line.split(':', 3)
            if len(parts) >= 4:
                line_num = int(parts[1])
                col_num = int(parts[2]) if parts[2].isdigit() else 1
                message_part = parts[3].strip()
                
                if message_part.startswith('error:'):
                    message = message_part[6:].strip()
                    
                    # Extract error code if present
                    rule_code = "MYPY_ERROR"
                    if '[' in message and ']' in message:
                        code_start = message.rfind('[')
                        code_end = message.rfind(']')
                        if code_start < code_end:
                            rule_code = message[code_start+1:code_end]
                            message = message[:code_start].strip()
                    
                    return QualityIssue(
                        file_path=file_path,
                        line_number=line_num,
                        column=col_num,
                        issue_type=QualityIssueType.TYPE_HINTS,
                        severity=QualitySeverity.ERROR,
                        message=message,
                        rule_code=rule_code,
                        suggestion="Fix type annotation based on mypy error"
                    )
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse mypy output: {line} - {e}")
        
        return None
    
    def _is_void_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is void (doesn't return a value)."""
        # Check for explicit return None or no return statements
        has_return_value = False
        
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                if child.value is not None:
                    # Check if it's not just returning None
                    if not (isinstance(child.value, ast.Constant) and child.value.value is None):
                        has_return_value = True
                        break
        
        return not has_return_value
    
    def _check_inconsistent_annotations(self, file_path: Path, node: ast.FunctionDef) -> List[QualityIssue]:
        """Check for inconsistent type annotations."""
        issues = []
        
        # Check if some parameters have annotations but others don't
        annotated_params = []
        unannotated_params = []
        
        for arg in node.args.args:
            if arg.arg in ['self', 'cls']:
                continue
            
            if arg.annotation is not None:
                annotated_params.append(arg.arg)
            else:
                unannotated_params.append(arg.arg)
        
        # If we have both annotated and unannotated parameters, flag as inconsistent
        if annotated_params and unannotated_params:
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.TYPE_HINTS,
                severity=QualitySeverity.INFO,
                message=f"Function '{node.name}' has inconsistent parameter annotations",
                rule_code="INCONSISTENT_ANNOTATIONS",
                suggestion="Either annotate all parameters or none for consistency"
            )
            issues.append(issue)
        
        return issues
    
    def _check_complex_annotations(self, file_path: Path, node: ast.FunctionDef) -> List[QualityIssue]:
        """Check for overly complex type annotations."""
        issues = []
        
        # Check return type complexity
        if node.returns and self._is_complex_annotation(node.returns):
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.TYPE_HINTS,
                severity=QualitySeverity.INFO,
                message=f"Function '{node.name}' has complex return type annotation",
                rule_code="COMPLEX_RETURN_TYPE",
                suggestion="Consider using type aliases for complex return types"
            )
            issues.append(issue)
        
        # Check parameter type complexity
        for arg in node.args.args:
            if arg.annotation and self._is_complex_annotation(arg.annotation):
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type=QualityIssueType.TYPE_HINTS,
                    severity=QualitySeverity.INFO,
                    message=f"Parameter '{arg.arg}' has complex type annotation",
                    rule_code="COMPLEX_PARAM_TYPE",
                    suggestion=f"Consider using type aliases for complex parameter types"
                )
                issues.append(issue)
        
        return issues
    
    def _is_complex_annotation(self, annotation: ast.AST) -> bool:
        """Check if type annotation is overly complex."""
        # Count nested levels and complexity
        complexity_score = self._calculate_annotation_complexity(annotation)
        return complexity_score > 3  # Threshold for complexity
    
    def _calculate_annotation_complexity(self, node: ast.AST) -> int:
        """Calculate complexity score for type annotation."""
        if isinstance(node, ast.Name):
            return 1
        elif isinstance(node, ast.Subscript):
            # Generic types like List[int], Dict[str, int]
            return 1 + self._calculate_annotation_complexity(node.slice)
        elif isinstance(node, ast.Tuple):
            # Union types or tuple types
            return sum(self._calculate_annotation_complexity(elt) for elt in node.elts)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union types with | operator (Python 3.10+)
            return self._calculate_annotation_complexity(node.left) + self._calculate_annotation_complexity(node.right)
        elif hasattr(node, 'elts'):
            # Handle other collection types
            return sum(self._calculate_annotation_complexity(elt) for elt in node.elts)
        else:
            return 1