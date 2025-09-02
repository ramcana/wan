"""
Documentation completeness validator.
"""

import ast
from pathlib import Path
from typing import List, Dict, Any
import logging

from ..models import QualityIssue, QualityIssueType, QualitySeverity, QualityConfig


logger = logging.getLogger(__name__)


class DocumentationValidator:
    """Validates documentation completeness for functions, classes, and modules."""
    
    def __init__(self, config: QualityConfig):
        """Initialize validator with configuration."""
        self.config = config
    
    def validate_documentation(self, file_path: Path, tree: ast.AST) -> tuple[List[QualityIssue], Dict[str, Any]]:
        """
        Validate documentation completeness in the given AST.
        
        Returns:
            Tuple of (issues, metrics)
        """
        issues = []
        metrics = {
            'functions_count': 0,
            'classes_count': 0,
            'documented_functions': 0,
            'documented_classes': 0,
            'has_module_docstring': False
        }
        
        # Check module docstring
        if self.config.require_module_docstrings:
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                metrics['has_module_docstring'] = True
                if len(module_docstring.strip()) < self.config.min_docstring_length:
                    issue = QualityIssue(
                        file_path=file_path,
                        line_number=1,
                        column=1,
                        issue_type=QualityIssueType.DOCUMENTATION,
                        severity=QualitySeverity.WARNING,
                        message=f"Module docstring too short (minimum {self.config.min_docstring_length} characters)",
                        rule_code="MODULE_DOCSTRING_SHORT",
                        suggestion="Expand module docstring with more detailed description"
                    )
                    issues.append(issue)
            else:
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=1,
                    column=1,
                    issue_type=QualityIssueType.DOCUMENTATION,
                    severity=QualitySeverity.WARNING,
                    message="Module is missing docstring",
                    rule_code="MISSING_MODULE_DOCSTRING",
                    suggestion="Add module docstring describing the module's purpose"
                )
                issues.append(issue)
        
        # Walk through AST nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_issues, func_documented = self._validate_function_documentation(file_path, node)
                issues.extend(func_issues)
                metrics['functions_count'] += 1
                if func_documented:
                    metrics['documented_functions'] += 1
            
            elif isinstance(node, ast.AsyncFunctionDef):
                func_issues, func_documented = self._validate_function_documentation(file_path, node)
                issues.extend(func_issues)
                metrics['functions_count'] += 1
                if func_documented:
                    metrics['documented_functions'] += 1
            
            elif isinstance(node, ast.ClassDef):
                class_issues, class_documented = self._validate_class_documentation(file_path, node)
                issues.extend(class_issues)
                metrics['classes_count'] += 1
                if class_documented:
                    metrics['documented_classes'] += 1
        
        return issues, metrics
    
    def _validate_function_documentation(self, file_path: Path, node: ast.FunctionDef) -> tuple[List[QualityIssue], bool]:
        """Validate function documentation."""
        issues = []
        documented = False
        
        # Skip private functions and special methods if configured
        if node.name.startswith('_') and not node.name.startswith('__'):
            return issues, True  # Consider private functions as documented by default
        
        if not self.config.require_function_docstrings:
            return issues, True
        
        docstring = ast.get_docstring(node)
        if docstring:
            documented = True
            
            # Check docstring length
            if len(docstring.strip()) < self.config.min_docstring_length:
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type=QualityIssueType.DOCUMENTATION,
                    severity=QualitySeverity.WARNING,
                    message=f"Function '{node.name}' docstring too short (minimum {self.config.min_docstring_length} characters)",
                    rule_code="FUNCTION_DOCSTRING_SHORT",
                    suggestion="Expand function docstring with parameters, returns, and examples"
                )
                issues.append(issue)
            
            # Check for parameter documentation
            if node.args.args:
                param_names = [arg.arg for arg in node.args.args if arg.arg != 'self']
                if param_names and not self._has_parameter_docs(docstring, param_names):
                    issue = QualityIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type=QualityIssueType.DOCUMENTATION,
                        severity=QualitySeverity.INFO,
                        message=f"Function '{node.name}' docstring missing parameter documentation",
                        rule_code="MISSING_PARAM_DOCS",
                        suggestion="Add parameter documentation in docstring"
                    )
                    issues.append(issue)
            
            # Check for return documentation
            if self._has_return_statement(node) and not self._has_return_docs(docstring):
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type=QualityIssueType.DOCUMENTATION,
                    severity=QualitySeverity.INFO,
                    message=f"Function '{node.name}' docstring missing return documentation",
                    rule_code="MISSING_RETURN_DOCS",
                    suggestion="Add return value documentation in docstring"
                )
                issues.append(issue)
        
        else:
            # Missing docstring
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.DOCUMENTATION,
                severity=QualitySeverity.WARNING,
                message=f"Function '{node.name}' is missing docstring",
                rule_code="MISSING_FUNCTION_DOCSTRING",
                suggestion="Add docstring describing function purpose, parameters, and return value"
            )
            issues.append(issue)
        
        return issues, documented
    
    def _validate_class_documentation(self, file_path: Path, node: ast.ClassDef) -> tuple[List[QualityIssue], bool]:
        """Validate class documentation."""
        issues = []
        documented = False
        
        # Skip private classes if configured
        if node.name.startswith('_'):
            return issues, True
        
        if not self.config.require_class_docstrings:
            return issues, True
        
        docstring = ast.get_docstring(node)
        if docstring:
            documented = True
            
            # Check docstring length
            if len(docstring.strip()) < self.config.min_docstring_length:
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type=QualityIssueType.DOCUMENTATION,
                    severity=QualitySeverity.WARNING,
                    message=f"Class '{node.name}' docstring too short (minimum {self.config.min_docstring_length} characters)",
                    rule_code="CLASS_DOCSTRING_SHORT",
                    suggestion="Expand class docstring with purpose, attributes, and usage examples"
                )
                issues.append(issue)
            
            # Check for attribute documentation if class has attributes
            if self._has_attributes(node) and not self._has_attribute_docs(docstring):
                issue = QualityIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type=QualityIssueType.DOCUMENTATION,
                    severity=QualitySeverity.INFO,
                    message=f"Class '{node.name}' docstring missing attribute documentation",
                    rule_code="MISSING_ATTRIBUTE_DOCS",
                    suggestion="Add attribute documentation in class docstring"
                )
                issues.append(issue)
        
        else:
            # Missing docstring
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.DOCUMENTATION,
                severity=QualitySeverity.WARNING,
                message=f"Class '{node.name}' is missing docstring",
                rule_code="MISSING_CLASS_DOCSTRING",
                suggestion="Add docstring describing class purpose, attributes, and usage"
            )
            issues.append(issue)
        
        return issues, documented
    
    def _has_parameter_docs(self, docstring: str, param_names: List[str]) -> bool:
        """Check if docstring contains parameter documentation."""
        docstring_lower = docstring.lower()
        
        # Look for common parameter documentation patterns
        param_indicators = ['args:', 'arguments:', 'parameters:', 'param ', 'parameter ']
        
        for indicator in param_indicators:
            if indicator in docstring_lower:
                return True
        
        # Check if any parameter names are mentioned
        for param in param_names:
            if param.lower() in docstring_lower:
                return True
        
        return False
    
    def _has_return_docs(self, docstring: str) -> bool:
        """Check if docstring contains return documentation."""
        docstring_lower = docstring.lower()
        return_indicators = ['returns:', 'return:', 'yields:', 'yield:']
        
        return any(indicator in docstring_lower for indicator in return_indicators)
    
    def _has_attribute_docs(self, docstring: str) -> bool:
        """Check if docstring contains attribute documentation."""
        docstring_lower = docstring.lower()
        attr_indicators = ['attributes:', 'attribute:', 'attrs:']
        
        return any(indicator in docstring_lower for indicator in attr_indicators)
    
    def _has_return_statement(self, node: ast.FunctionDef) -> bool:
        """Check if function has return statements."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
        return False
    
    def _has_attributes(self, node: ast.ClassDef) -> bool:
        """Check if class has attributes."""
        for child in node.body:
            if isinstance(child, ast.Assign):
                return True
            elif isinstance(child, ast.AnnAssign):
                return True
        return False