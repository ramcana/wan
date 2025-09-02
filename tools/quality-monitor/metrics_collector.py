"""
Quality metrics collection system.
"""

import os
import subprocess
import json
import ast
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

try:
    from tools.quality-monitor.models import QualityMetric, MetricType
except ImportError:
    from models import QualityMetric, MetricType


class MetricsCollector:
    """Collects various quality metrics from the codebase."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.python_files = self._find_python_files()
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip common directories that shouldn't be analyzed
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def collect_test_coverage(self) -> QualityMetric:
        """Collect test coverage metrics."""
        try:
            # Run coverage analysis
            result = subprocess.run(
                ['python', '-m', 'pytest', '--cov=.', '--cov-report=json', '--quiet'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Try to read coverage report
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0.0)
                
                return QualityMetric(
                    metric_type=MetricType.TEST_COVERAGE,
                    value=total_coverage,
                    timestamp=datetime.now(),
                    details={
                        'lines_covered': coverage_data.get('totals', {}).get('covered_lines', 0),
                        'total_lines': coverage_data.get('totals', {}).get('num_statements', 0),
                        'missing_lines': coverage_data.get('totals', {}).get('missing_lines', 0)
                    }
                )
            else:
                # Fallback: estimate coverage based on test files
                test_files = [f for f in self.python_files if 'test' in str(f)]
                source_files = [f for f in self.python_files if 'test' not in str(f)]
                
                if source_files:
                    coverage_estimate = min(100.0, (len(test_files) / len(source_files)) * 100)
                else:
                    coverage_estimate = 0.0
                
                return QualityMetric(
                    metric_type=MetricType.TEST_COVERAGE,
                    value=coverage_estimate,
                    timestamp=datetime.now(),
                    details={
                        'test_files': len(test_files),
                        'source_files': len(source_files),
                        'estimated': True
                    }
                )
        
        except Exception as e:
            return QualityMetric(
                metric_type=MetricType.TEST_COVERAGE,
                value=0.0,
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
    
    def collect_code_complexity(self) -> QualityMetric:
        """Collect code complexity metrics."""
        total_complexity = 0
        function_count = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        total_complexity += complexity
                        function_count += 1
            
            except Exception:
                continue
        
        average_complexity = total_complexity / function_count if function_count > 0 else 0
        
        return QualityMetric(
            metric_type=MetricType.CODE_COMPLEXITY,
            value=average_complexity,
            timestamp=datetime.now(),
            details={
                'total_complexity': total_complexity,
                'function_count': function_count,
                'files_analyzed': len(self.python_files)
            }
        )
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def collect_documentation_coverage(self) -> QualityMetric:
        """Collect documentation coverage metrics."""
        documented_functions = 0
        total_functions = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        
                        # Check if function/class has docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Constant) and 
                            isinstance(node.body[0].value.value, str)):
                            documented_functions += 1
            
            except Exception:
                continue
        
        coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        
        return QualityMetric(
            metric_type=MetricType.DOCUMENTATION_COVERAGE,
            value=coverage,
            timestamp=datetime.now(),
            details={
                'documented_functions': documented_functions,
                'total_functions': total_functions
            }
        )
    
    def collect_duplicate_code(self) -> QualityMetric:
        """Collect duplicate code metrics."""
        # Simple duplicate detection based on similar lines
        line_hashes = {}
        duplicate_lines = 0
        total_lines = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    # Normalize line (remove whitespace, comments)
                    normalized = re.sub(r'#.*$', '', line.strip())
                    if normalized and len(normalized) > 10:  # Skip short lines
                        line_hash = hash(normalized)
                        if line_hash in line_hashes:
                            duplicate_lines += 1
                        else:
                            line_hashes[line_hash] = 1
                        total_lines += 1
            
            except Exception:
                continue
        
        duplicate_percentage = (duplicate_lines / total_lines * 100) if total_lines > 0 else 0
        
        return QualityMetric(
            metric_type=MetricType.DUPLICATE_CODE,
            value=duplicate_percentage,
            timestamp=datetime.now(),
            details={
                'duplicate_lines': duplicate_lines,
                'total_lines': total_lines
            }
        )
    
    def collect_style_violations(self) -> QualityMetric:
        """Collect style violation metrics."""
        try:
            # Run flake8 for style checking
            result = subprocess.run(
                ['python', '-m', 'flake8', '--count', '--statistics', '.'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Parse flake8 output for violation count
            violations = 0
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip().isdigit():
                        violations += int(line.strip())
            
            # Calculate violations per 1000 lines of code
            total_lines = sum(1 for f in self.python_files 
                            for line in open(f, 'r', encoding='utf-8', errors='ignore'))
            
            violations_per_kloc = (violations / total_lines * 1000) if total_lines > 0 else 0
            
            return QualityMetric(
                metric_type=MetricType.STYLE_VIOLATIONS,
                value=violations_per_kloc,
                timestamp=datetime.now(),
                details={
                    'total_violations': violations,
                    'total_lines': total_lines
                }
            )
        
        except Exception as e:
            return QualityMetric(
                metric_type=MetricType.STYLE_VIOLATIONS,
                value=0.0,
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
    
    def collect_type_hint_coverage(self) -> QualityMetric:
        """Collect type hint coverage metrics."""
        functions_with_hints = 0
        total_functions = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Check if function has type hints
                        has_hints = (
                            node.returns is not None or
                            any(arg.annotation is not None for arg in node.args.args)
                        )
                        
                        if has_hints:
                            functions_with_hints += 1
            
            except Exception:
                continue
        
        coverage = (functions_with_hints / total_functions * 100) if total_functions > 0 else 0
        
        return QualityMetric(
            metric_type=MetricType.TYPE_HINT_COVERAGE,
            value=coverage,
            timestamp=datetime.now(),
            details={
                'functions_with_hints': functions_with_hints,
                'total_functions': total_functions
            }
        )
    
    def collect_all_metrics(self) -> List[QualityMetric]:
        """Collect all available quality metrics."""
        metrics = []
        
        collectors = [
            self.collect_test_coverage,
            self.collect_code_complexity,
            self.collect_documentation_coverage,
            self.collect_duplicate_code,
            self.collect_style_violations,
            self.collect_type_hint_coverage
        ]
        
        for collector in collectors:
            try:
                metric = collector()
                metrics.append(metric)
            except Exception as e:
                print(f"Error collecting metric with {collector.__name__}: {e}")
        
        return metrics