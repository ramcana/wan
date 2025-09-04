"""
Code complexity analysis and recommendations.
"""

import ast
import math
from pathlib import Path
from typing import List, Dict, Any, Set
import logging

from models import QualityIssue, QualityIssueType, QualitySeverity, QualityConfig


logger = logging.getLogger(__name__)


class ComplexityAnalyzer:
    """Analyzes code complexity and provides refactoring recommendations."""
    
    def __init__(self, config: QualityConfig):
        """Initialize analyzer with configuration."""
        self.config = config
    
    def analyze_complexity(self, file_path: Path, tree: ast.AST) -> tuple[List[QualityIssue], Dict[str, Any]]:
        """
        Analyze complexity in the given AST.
        
        Returns:
            Tuple of (issues, metrics)
        """
        issues = []
        metrics = {
            'total_complexity': 0,
            'function_complexities': [],
            'class_complexities': [],
            'average_complexity': 0.0,
            'maintainability_index': 0.0,
            'total_functions': 0,
            'complex_functions': 0
        }
        
        # Analyze module-level complexity
        module_issues, module_metrics = self._analyze_module_complexity(file_path, tree)
        issues.extend(module_issues)
        
        # Walk through AST for function and class analysis
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_issues, func_complexity = self._analyze_function_complexity(file_path, node)
                issues.extend(func_issues)
                
                metrics['total_functions'] += 1
                metrics['function_complexities'].append(func_complexity)
                metrics['total_complexity'] += func_complexity
                
                if func_complexity > self.config.max_cyclomatic_complexity:
                    metrics['complex_functions'] += 1
            
            elif isinstance(node, ast.ClassDef):
                class_issues, class_complexity = self._analyze_class_complexity(file_path, node)
                issues.extend(class_issues)
                metrics['class_complexities'].append(class_complexity)
        
        # Calculate average complexity
        if metrics['total_functions'] > 0:
            metrics['average_complexity'] = metrics['total_complexity'] / metrics['total_functions']
        
        # Calculate maintainability index
        metrics['maintainability_index'] = self._calculate_maintainability_index(
            file_path, tree, metrics['average_complexity']
        )
        
        return issues, metrics
    
    def _analyze_module_complexity(self, file_path: Path, tree: ast.AST) -> tuple[List[QualityIssue], Dict[str, Any]]:
        """Analyze module-level complexity."""
        issues = []
        metrics = {}
        
        # Count total lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            total_lines = len(lines)
        except Exception:
            total_lines = 0
        
        # Check module length
        if total_lines > self.config.max_module_length:
            issue = QualityIssue(
                file_path=file_path,
                line_number=1,
                column=1,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=QualitySeverity.WARNING,
                message=f"Module too long ({total_lines} > {self.config.max_module_length} lines)",
                rule_code="MODULE_TOO_LONG",
                suggestion="Consider splitting module into smaller, focused modules"
            )
            issues.append(issue)
        
        # Count imports and check for excessive imports
        import_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
        
        if import_count > 20:  # Threshold for excessive imports
            issue = QualityIssue(
                file_path=file_path,
                line_number=1,
                column=1,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=QualitySeverity.INFO,
                message=f"Module has many imports ({import_count})",
                rule_code="MANY_IMPORTS",
                suggestion="Consider reducing dependencies or splitting module"
            )
            issues.append(issue)
        
        return issues, metrics
    
    def _analyze_function_complexity(self, file_path: Path, node: ast.FunctionDef) -> tuple[List[QualityIssue], int]:
        """Analyze complexity of a specific function."""
        issues = []
        
        # Calculate cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(node)
        
        # Check function length
        function_length = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        
        if function_length > self.config.max_function_length:
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=QualitySeverity.WARNING,
                message=f"Function '{node.name}' too long ({function_length} > {self.config.max_function_length} lines)",
                rule_code="FUNCTION_TOO_LONG",
                suggestion="Consider breaking function into smaller functions"
            )
            issues.append(issue)
        
        # Check cyclomatic complexity
        if complexity > self.config.max_cyclomatic_complexity:
            severity = QualitySeverity.ERROR if complexity > self.config.max_cyclomatic_complexity * 1.5 else QualitySeverity.WARNING
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=severity,
                message=f"Function '{node.name}' too complex (complexity {complexity} > {self.config.max_cyclomatic_complexity})",
                rule_code="HIGH_COMPLEXITY",
                suggestion=self._get_complexity_reduction_suggestion(node, complexity)
            )
            issues.append(issue)
        
        # Check for deeply nested code
        max_nesting = self._calculate_max_nesting_depth(node)
        if max_nesting > 4:
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=QualitySeverity.WARNING,
                message=f"Function '{node.name}' has deep nesting (depth {max_nesting})",
                rule_code="DEEP_NESTING",
                suggestion="Consider using early returns or extracting nested logic into separate functions"
            )
            issues.append(issue)
        
        # Check parameter count
        param_count = len(node.args.args)
        if param_count > 5:
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=QualitySeverity.INFO,
                message=f"Function '{node.name}' has many parameters ({param_count})",
                rule_code="MANY_PARAMETERS",
                suggestion="Consider using a configuration object or reducing parameters"
            )
            issues.append(issue)
        
        return issues, complexity
    
    def _analyze_class_complexity(self, file_path: Path, node: ast.ClassDef) -> tuple[List[QualityIssue], int]:
        """Analyze complexity of a specific class."""
        issues = []
        
        # Count methods and attributes
        method_count = 0
        attribute_count = 0
        total_complexity = 0
        
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_count += 1
                method_complexity = self._calculate_cyclomatic_complexity(child)
                total_complexity += method_complexity
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                attribute_count += 1
        
        # Check class length
        class_length = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        
        if class_length > self.config.max_class_length:
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=QualitySeverity.WARNING,
                message=f"Class '{node.name}' too long ({class_length} > {self.config.max_class_length} lines)",
                rule_code="CLASS_TOO_LONG",
                suggestion="Consider splitting class into smaller, focused classes"
            )
            issues.append(issue)
        
        # Check method count
        if method_count > 20:
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=QualitySeverity.WARNING,
                message=f"Class '{node.name}' has many methods ({method_count})",
                rule_code="MANY_METHODS",
                suggestion="Consider splitting class responsibilities or using composition"
            )
            issues.append(issue)
        
        # Check for god class (many attributes and methods)
        if method_count > 15 and attribute_count > 10:
            issue = QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                issue_type=QualityIssueType.COMPLEXITY,
                severity=QualitySeverity.WARNING,
                message=f"Class '{node.name}' may be a 'god class' ({method_count} methods, {attribute_count} attributes)",
                rule_code="GOD_CLASS",
                suggestion="Consider applying Single Responsibility Principle and splitting the class"
            )
            issues.append(issue)
        
        return issues, total_complexity
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function or method."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points that increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # And/Or operations
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ListComp):
                # List comprehensions with conditions
                complexity += len(child.generators)
                for generator in child.generators:
                    complexity += len(generator.ifs)
            elif isinstance(child, ast.DictComp):
                # Dict comprehensions with conditions
                complexity += len(child.generators)
                for generator in child.generators:
                    complexity += len(generator.ifs)
            elif isinstance(child, ast.SetComp):
                # Set comprehensions with conditions
                complexity += len(child.generators)
                for generator in child.generators:
                    complexity += len(generator.ifs)
        
        return complexity
    
    def _calculate_max_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth in a function."""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(node)
    
    def _calculate_maintainability_index(self, file_path: Path, tree: ast.AST, avg_complexity: float) -> float:
        """Calculate maintainability index (0-100, higher is better)."""
        try:
            # Get file metrics
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            total_lines = len(lines)
            code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            
            # Halstead metrics (simplified)
            operators, operands = self._calculate_halstead_metrics(tree)
            
            if code_lines == 0:
                return 100.0
            
            # Maintainability Index formula (simplified version)
            # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * CM))
            # Where: HV = Halstead Volume, CC = Cyclomatic Complexity, LOC = Lines of Code, CM = Comment Ratio
            
            halstead_volume = (operators + operands) * math.log2(max(1, operators + operands))
            comment_ratio = comment_lines / max(1, total_lines) * 100
            
            mi = (171 
                  - 5.2 * math.log(max(1, halstead_volume))
                  - 0.23 * avg_complexity
                  - 16.2 * math.log(max(1, code_lines))
                  + 50 * math.sin(math.sqrt(2.4 * comment_ratio)))
            
            # Normalize to 0-100 range
            return max(0.0, min(100.0, mi))
        
        except Exception as e:
            logger.debug(f"Failed to calculate maintainability index for {file_path}: {e}")
            return 50.0  # Default neutral score
    
    def _calculate_halstead_metrics(self, tree: ast.AST) -> tuple[int, int]:
        """Calculate simplified Halstead operators and operands count."""
        operators = 0
        operands = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                               ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
                               ast.FloorDiv, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
                               ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                operators += 1
            elif isinstance(node, (ast.Name, ast.Constant)):
                operands += 1
        
        return operators, operands
    
    def _get_complexity_reduction_suggestion(self, node: ast.FunctionDef, complexity: int) -> str:
        """Get specific suggestion for reducing function complexity."""
        suggestions = []
        
        # Analyze what's contributing to complexity
        if_count = sum(1 for child in ast.walk(node) if isinstance(child, ast.If))
        loop_count = sum(1 for child in ast.walk(node) if isinstance(child, (ast.While, ast.For, ast.AsyncFor)))
        try_count = sum(1 for child in ast.walk(node) if isinstance(child, ast.Try))
        
        if if_count > 3:
            suggestions.append("Consider using polymorphism or strategy pattern to reduce conditional complexity")
        
        if loop_count > 2:
            suggestions.append("Consider extracting loop logic into separate functions")
        
        if try_count > 2:
            suggestions.append("Consider consolidating error handling or using context managers")
        
        if complexity > 15:
            suggestions.append("Consider breaking this function into multiple smaller functions")
        
        if not suggestions:
            suggestions.append("Consider refactoring to reduce cyclomatic complexity")
        
        return "; ".join(suggestions)