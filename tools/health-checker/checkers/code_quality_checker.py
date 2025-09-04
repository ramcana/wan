import pytest
"""
Code quality health checker
"""

import subprocess
import ast
from pathlib import Path
from typing import List, Dict, Any
import logging

from tools...health_models import ComponentHealth, HealthIssue, HealthCategory, Severity, HealthConfig


class CodeQualityChecker:
    """Checks code quality metrics and issues"""
    
    def __init__(self, config: HealthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def check_health(self) -> ComponentHealth:
        """Check code quality health"""
        issues = []
        metrics = {}
        
        # Discover Python files
        python_files = self._discover_python_files()
        metrics["total_python_files"] = len(python_files)
        
        if len(python_files) == 0:
            issues.append(HealthIssue(
                severity=Severity.LOW,
                category=HealthCategory.CODE_QUALITY,
                title="No Python Files Found",
                description="No Python files found for code quality analysis",
                affected_components=["code_quality"],
                remediation_steps=["Ensure Python files exist in the project"]
            ))
            return ComponentHealth(
                component_name="code_quality",
                category=HealthCategory.CODE_QUALITY,
                score=50.0,
                status="warning",
                issues=issues,
                metrics=metrics
            )
        
        # Check syntax errors
        syntax_errors = self._check_syntax_errors(python_files)
        metrics["syntax_errors"] = len(syntax_errors)
        
        if syntax_errors:
            issues.append(HealthIssue(
                severity=Severity.CRITICAL,
                category=HealthCategory.CODE_QUALITY,
                title="Syntax Errors Found",
                description=f"Found {len(syntax_errors)} files with syntax errors",
                affected_components=["code_quality"],
                remediation_steps=[
                    "Fix syntax errors in Python files",
                    "Use IDE with syntax checking",
                    "Add pre-commit hooks for syntax validation"
                ],
                metadata={"syntax_errors": syntax_errors[:5]}
            ))
        
        # Check code complexity
        complexity_issues = self._check_code_complexity(python_files)
        metrics["high_complexity_functions"] = len(complexity_issues)
        
        if complexity_issues:
            issues.append(HealthIssue(
                severity=Severity.MEDIUM,
                category=HealthCategory.CODE_QUALITY,
                title="High Code Complexity",
                description=f"Found {len(complexity_issues)} functions with high complexity",
                affected_components=["code_quality"],
                remediation_steps=[
                    "Refactor complex functions",
                    "Break down large functions",
                    "Use helper functions to reduce complexity"
                ],
                metadata={"complex_functions": complexity_issues[:5]}
            ))
        
        # Check for code smells
        code_smells = self._check_code_smells(python_files)
        metrics["code_smells"] = len(code_smells)
        
        if code_smells:
            issues.append(HealthIssue(
                severity=Severity.LOW,
                category=HealthCategory.CODE_QUALITY,
                title="Code Smells Detected",
                description=f"Found {len(code_smells)} potential code quality issues",
                affected_components=["code_quality"],
                remediation_steps=[
                    "Review and refactor flagged code",
                    "Follow Python best practices",
                    "Use code quality tools like pylint or flake8"
                ],
                metadata={"code_smells": code_smells[:5]}
            ))
        
        # Check import organization
        import_issues = self._check_import_organization(python_files)
        metrics["import_issues"] = len(import_issues)
        
        if import_issues:
            issues.append(HealthIssue(
                severity=Severity.LOW,
                category=HealthCategory.CODE_QUALITY,
                title="Import Organization Issues",
                description=f"Found {len(import_issues)} files with import organization issues",
                affected_components=["code_quality"],
                remediation_steps=[
                    "Organize imports according to PEP 8",
                    "Use tools like isort to automatically organize imports",
                    "Remove unused imports"
                ],
                metadata={"import_issues": import_issues[:5]}
            ))
        
        # Check for TODO/FIXME comments
        todo_comments = self._check_todo_comments(python_files)
        metrics["todo_comments"] = len(todo_comments)
        
        if len(todo_comments) > 20:
            issues.append(HealthIssue(
                severity=Severity.LOW,
                category=HealthCategory.CODE_QUALITY,
                title="Many TODO Comments",
                description=f"Found {len(todo_comments)} TODO/FIXME comments",
                affected_components=["code_quality"],
                remediation_steps=[
                    "Review and address TODO comments",
                    "Create issues for important TODOs",
                    "Remove outdated TODO comments"
                ]
            ))
        
        # Run external code quality tools if available
        external_metrics = self._run_external_quality_tools()
        metrics.update(external_metrics)
        
        # Calculate score
        score = self._calculate_code_quality_score(metrics, issues)
        status = self._determine_status(score)
        
        return ComponentHealth(
            component_name="code_quality",
            category=HealthCategory.CODE_QUALITY,
            score=score,
            status=status,
            issues=issues,
            metrics=metrics
        )
    
    def _discover_python_files(self) -> List[Path]:
        """Discover Python files in the project"""
        python_files = []
        
        try:
            # Find all Python files, excluding common directories
            exclude_dirs = {'.git', '__pycache__', 'venv', 'node_modules', '.pytest_cache'}
            
            for py_file in self.config.project_root.rglob("*.py"):
                # Skip files in excluded directories
                if not any(excluded in py_file.parts for excluded in exclude_dirs):
                    python_files.append(py_file)
        except Exception as e:
            self.logger.warning(f"Failed to discover Python files: {e}")
        
        return python_files
    
    def _check_syntax_errors(self, python_files: List[Path]) -> List[Dict[str, str]]:
        """Check for syntax errors in Python files"""
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    syntax_errors.append({
                        "file": str(py_file.relative_to(self.config.project_root)),
                        "line": e.lineno or 0,
                        "error": str(e.msg)
                    })
            except Exception as e:
                self.logger.warning(f"Failed to check syntax for {py_file}: {e}")
        
        return syntax_errors
    
    def _check_code_complexity(self, python_files: List[Path]) -> List[Dict[str, Any]]:
        """Check for high complexity functions"""
        complex_functions = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            complexity = self._calculate_cyclomatic_complexity(node)
                            
                            if complexity > 10:  # Threshold for high complexity
                                complex_functions.append({
                                    "file": str(py_file.relative_to(self.config.project_root)),
                                    "function": node.name,
                                    "line": node.lineno,
                                    "complexity": complexity
                                })
                except SyntaxError:
                    # Skip files with syntax errors (already caught above)
                    continue
            except Exception as e:
                self.logger.warning(f"Failed to check complexity for {py_file}: {e}")
        
        return complex_functions
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Count decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _check_code_smells(self, python_files: List[Path]) -> List[Dict[str, str]]:
        """Check for common code smells"""
        code_smells = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    # Check for long lines
                    if len(line.rstrip()) > 120:
                        code_smells.append({
                            "file": str(py_file.relative_to(self.config.project_root)),
                            "line": i,
                            "issue": "Line too long (>120 characters)",
                            "type": "line_length"
                        })
                    
                    # Check for multiple statements on one line
                    if ';' in line and not line.strip().startswith('#'):
                        code_smells.append({
                            "file": str(py_file.relative_to(self.config.project_root)),
                            "line": i,
                            "issue": "Multiple statements on one line",
                            "type": "multiple_statements"
                        })
                    
                    # Check for bare except clauses
                    if 'except:' in line:
                        code_smells.append({
                            "file": str(py_file.relative_to(self.config.project_root)),
                            "line": i,
                            "issue": "Bare except clause",
                            "type": "bare_except"
                        })
            except Exception as e:
                self.logger.warning(f"Failed to check code smells for {py_file}: {e}")
        
        return code_smells
    
    def _check_import_organization(self, python_files: List[Path]) -> List[Dict[str, str]]:
        """Check import organization issues"""
        import_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            imports.append((node.lineno, type(node).__name__))
                    
                    # Check if imports are at the top
                    if imports:
                        first_import_line = min(imp[0] for imp in imports)
                        
                        # Look for non-import, non-comment lines before first import
                        lines = content.split('\n')
                        for i, line in enumerate(lines[:first_import_line-1], 1):
                            stripped = line.strip()
                            if (stripped and 
                                not stripped.startswith('#') and 
                                not stripped.startswith('"""') and
                                not stripped.startswith("'''") and
                                'coding:' not in stripped and
                                '__future__' not in stripped):
                                
                                import_issues.append({
                                    "file": str(py_file.relative_to(self.config.project_root)),
                                    "issue": "Imports not at top of file",
                                    "line": first_import_line
                                })
                                break
                except SyntaxError:
                    continue
            except Exception as e:
                self.logger.warning(f"Failed to check imports for {py_file}: {e}")
        
        return import_issues
    
    def _check_todo_comments(self, python_files: List[Path]) -> List[Dict[str, str]]:
        """Check for TODO/FIXME comments"""
        todo_comments = []
        
        import re
        todo_pattern = re.compile(r'#.*\b(TODO|FIXME|XXX|HACK)\b', re.IGNORECASE)
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    match = todo_pattern.search(line)
                    if match:
                        todo_comments.append({
                            "file": str(py_file.relative_to(self.config.project_root)),
                            "line": i,
                            "comment": line.strip(),
                            "type": match.group(1).upper()
                        })
            except Exception as e:
                self.logger.warning(f"Failed to check TODO comments for {py_file}: {e}")
        
        return todo_comments
    
    def _run_external_quality_tools(self) -> Dict[str, Any]:
        """Run external code quality tools if available"""
        metrics = {}
        
        # Try to run flake8
        try:
            result = subprocess.run([
                "python", "-m", "flake8", ".", "--count", "--statistics"
            ], capture_output=True, text=True, timeout=60, cwd=self.config.project_root)
            
            if result.returncode == 0:
                metrics["flake8_issues"] = 0
            else:
                # Parse flake8 output for issue count
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if line.strip().isdigit():
                        metrics["flake8_issues"] = int(line.strip())
                        break
        except:
            pass
        
        # Try to run pylint (basic check)
        try:
            result = subprocess.run([
                "python", "-m", "pylint", ".", "--reports=n", "--score=y"
            ], capture_output=True, text=True, timeout=120, cwd=self.config.project_root)
            
            # Parse pylint score
            import re
            score_match = re.search(r'Your code has been rated at ([\d.]+)/10', result.stdout)
            if score_match:
                metrics["pylint_score"] = float(score_match.group(1))
        except:
            pass
        
        return metrics
    
    def _calculate_code_quality_score(self, metrics: Dict[str, Any], issues: List[HealthIssue]) -> float:
        """Calculate code quality score"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                base_score -= 30
            elif issue.severity == Severity.HIGH:
                base_score -= 20
            elif issue.severity == Severity.MEDIUM:
                base_score -= 15
            elif issue.severity == Severity.LOW:
                base_score -= 5
        
        # Bonus points for external tool scores
        pylint_score = metrics.get("pylint_score", 0)
        if pylint_score >= 8.0:
            base_score += 10
        elif pylint_score >= 6.0:
            base_score += 5
        
        flake8_issues = metrics.get("flake8_issues", 0)
        if flake8_issues == 0:
            base_score += 5
        elif flake8_issues < 10:
            base_score += 2
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_status(self, score: float) -> str:
        """Determine health status from score"""
        if score >= self.config.warning_threshold:
            return "healthy"
        elif score >= self.config.critical_threshold:
            return "warning"
        else:
            return "critical"