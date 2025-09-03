#!/usr/bin/env python3
"""
Comprehensive Test Suite Auditor

This module provides comprehensive analysis of test files to identify:
- Broken tests (import errors, syntax issues)
- Incomplete tests (missing assertions, empty test bodies)
- Flaky tests (intermittent failures)
- Performance issues (slow or hanging tests)
- Missing dependencies and fixtures
"""

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import traceback


@dataclass
class TestIssue:
    """Represents a specific issue found in a test"""
    test_file: str
    test_name: str
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class TestFileAnalysis:
    """Analysis results for a single test file"""
    file_path: str
    total_tests: int
    passing_tests: int
    failing_tests: int
    skipped_tests: int
    issues: List[TestIssue]
    imports: List[str]
    missing_imports: List[str]
    fixtures_used: List[str]
    missing_fixtures: List[str]
    execution_time: float
    has_syntax_errors: bool
    has_import_errors: bool


@dataclass
class TestSuiteAuditReport:
    """Complete audit report for the entire test suite"""
    total_files: int
    total_tests: int
    passing_tests: int
    failing_tests: int
    skipped_tests: int
    broken_files: List[str]
    flaky_tests: List[str]
    slow_tests: List[str]
    file_analyses: List[TestFileAnalysis]
    critical_issues: List[TestIssue]
    recommendations: List[str]
    execution_summary: Dict[str, Any]


class TestDiscoveryEngine:
    """Discovers and categorizes test files across the project"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_patterns = [
            r'test_.*\.py$',
            r'.*_test\.py$',
            r'tests\.py$'
        ]
        
    def discover_test_files(self) -> List[Path]:
        """Discover all test files in the project"""
        test_files = []
        
        # Common test directories
        test_dirs = [
            'tests',
            'backend/tests',
            'frontend/src/tests',
            'local_installation/tests'
        ]
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                test_files.extend(self._scan_directory(test_path))
        
        # Also scan for test files in other directories
        for root, dirs, files in os.walk(self.project_root):
            # Skip common non-test directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
            
            for file in files:
                if any(re.match(pattern, file) for pattern in self.test_patterns):
                    file_path = Path(root) / file
                    if file_path not in test_files:
                        test_files.append(file_path)
        
        return sorted(test_files)
    
    def _scan_directory(self, directory: Path) -> List[Path]:
        """Recursively scan directory for test files"""
        test_files = []
        
        for item in directory.rglob('*.py'):
            if any(re.match(pattern, item.name) for pattern in self.test_patterns):
                test_files.append(item)
        
        return test_files


class TestDependencyAnalyzer:
    """Analyzes test dependencies, imports, and fixtures"""
    
    def __init__(self):
        self.common_test_imports = {
            'pytest', 'unittest', 'mock', 'asyncio', 'json', 'os', 'sys',
            'pathlib', 'tempfile', 'shutil', 'subprocess', 'time', 'datetime'
        }
    
    def analyze_dependencies(self, test_file: Path) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Analyze test file dependencies
        Returns: (imports, missing_imports, fixtures_used, missing_fixtures)
        """
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            imports = self._extract_imports(tree)
            fixtures_used = self._extract_fixtures(tree)
            
            missing_imports = self._check_missing_imports(imports, test_file)
            missing_fixtures = self._check_missing_fixtures(fixtures_used, test_file)
            
            return imports, missing_imports, fixtures_used, missing_fixtures
            
        except Exception as e:
            return [], [f"Error analyzing dependencies: {str(e)}"], [], []
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all import statements from AST"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        return imports
    
    def _extract_fixtures(self, tree: ast.AST) -> List[str]:
        """Extract pytest fixtures used in test functions"""
        fixtures = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function parameters for fixtures
                for arg in node.args.args:
                    if arg.arg not in ['self', 'cls']:
                        fixtures.append(arg.arg)
                
                # Check for fixture decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'fixture':
                        fixtures.append(node.name)
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == 'fixture':
                        fixtures.append(node.name)
        
        return fixtures
    
    def _check_missing_imports(self, imports: List[str], test_file: Path) -> List[str]:
        """Check for imports that cannot be resolved"""
        missing = []
        
        for imp in imports:
            if imp in self.common_test_imports:
                continue
                
            try:
                # Try to import the module
                if '.' in imp:
                    module_name = imp.split('.')[0]
                else:
                    module_name = imp
                
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    missing.append(imp)
            except (ImportError, ModuleNotFoundError, ValueError):
                missing.append(imp)
        
        return missing
    
    def _check_missing_fixtures(self, fixtures: List[str], test_file: Path) -> List[str]:
        """Check for fixtures that are not defined"""
        missing = []
        
        # Look for conftest.py files that might define fixtures
        conftest_files = []
        current_dir = test_file.parent
        
        while current_dir != current_dir.parent:
            conftest_path = current_dir / 'conftest.py'
            if conftest_path.exists():
                conftest_files.append(conftest_path)
            current_dir = current_dir.parent
        
        # Parse conftest files to find defined fixtures
        defined_fixtures = set()
        for conftest_file in conftest_files:
            try:
                with open(conftest_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for decorator in node.decorator_list:
                            if (isinstance(decorator, ast.Name) and decorator.id == 'fixture') or \
                               (isinstance(decorator, ast.Attribute) and decorator.attr == 'fixture'):
                                defined_fixtures.add(node.name)
            except Exception:
                continue
        
        # Check which fixtures are missing
        for fixture in fixtures:
            if fixture not in defined_fixtures and fixture not in ['request', 'monkeypatch', 'tmp_path', 'capsys']:
                missing.append(fixture)
        
        return missing


class TestPerformanceProfiler:
    """Profiles test execution performance to identify slow tests"""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.slow_test_threshold = 5.0  # seconds
    
    def profile_test_file(self, test_file: Path) -> Tuple[float, List[str], bool]:
        """
        Profile a single test file
        Returns: (execution_time, slow_tests, timed_out)
        """
        try:
            start_time = time.time()
            
            # Run pytest with timing information
            cmd = [
                sys.executable, '-m', 'pytest', 
                str(test_file), 
                '--tb=no', 
                '-v',
                '--durations=0'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=test_file.parent
            )
            
            execution_time = time.time() - start_time
            slow_tests = self._parse_slow_tests(result.stdout)
            
            return execution_time, slow_tests, False
            
        except subprocess.TimeoutExpired:
            return self.timeout_seconds, [], True
        except Exception as e:
            return 0.0, [], False
    
    def _parse_slow_tests(self, pytest_output: str) -> List[str]:
        """Parse pytest output to identify slow tests"""
        slow_tests = []
        
        # Look for duration information in pytest output
        duration_pattern = r'(\d+\.\d+)s\s+.*?::(.*?)(?:\s|$)'
        matches = re.findall(duration_pattern, pytest_output)
        
        for duration_str, test_name in matches:
            duration = float(duration_str)
            if duration > self.slow_test_threshold:
                slow_tests.append(f"{test_name} ({duration:.2f}s)")
        
        return slow_tests


class TestAuditor:
    """Main test auditor that orchestrates all analysis components"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.discovery_engine = TestDiscoveryEngine(project_root)
        self.dependency_analyzer = TestDependencyAnalyzer()
        self.performance_profiler = TestPerformanceProfiler()
    
    def audit_test_suite(self) -> TestSuiteAuditReport:
        """Perform comprehensive audit of the entire test suite"""
        print("Starting comprehensive test suite audit...")
        
        # Discover all test files
        test_files = self.discovery_engine.discover_test_files()
        print(f"Discovered {len(test_files)} test files")
        
        file_analyses = []
        total_tests = 0
        passing_tests = 0
        failing_tests = 0
        skipped_tests = 0
        broken_files = []
        flaky_tests = []
        slow_tests = []
        critical_issues = []
        
        for i, test_file in enumerate(test_files, 1):
            print(f"Analyzing {i}/{len(test_files)}: {test_file.relative_to(self.project_root)}")
            
            analysis = self._analyze_test_file(test_file)
            file_analyses.append(analysis)
            
            total_tests += analysis.total_tests
            passing_tests += analysis.passing_tests
            failing_tests += analysis.failing_tests
            skipped_tests += analysis.skipped_tests
            
            if analysis.has_syntax_errors or analysis.has_import_errors:
                broken_files.append(str(test_file.relative_to(self.project_root)))
            
            # Collect critical issues
            for issue in analysis.issues:
                if issue.severity == 'critical':
                    critical_issues.append(issue)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(file_analyses)
        
        # Create execution summary
        execution_summary = {
            'total_execution_time': sum(fa.execution_time for fa in file_analyses),
            'average_execution_time': sum(fa.execution_time for fa in file_analyses) / len(file_analyses) if file_analyses else 0,
            'files_with_issues': len([fa for fa in file_analyses if fa.issues]),
            'import_error_files': len([fa for fa in file_analyses if fa.has_import_errors]),
            'syntax_error_files': len([fa for fa in file_analyses if fa.has_syntax_errors])
        }
        
        return TestSuiteAuditReport(
            total_files=len(test_files),
            total_tests=total_tests,
            passing_tests=passing_tests,
            failing_tests=failing_tests,
            skipped_tests=skipped_tests,
            broken_files=broken_files,
            flaky_tests=flaky_tests,
            slow_tests=slow_tests,
            file_analyses=file_analyses,
            critical_issues=critical_issues,
            recommendations=recommendations,
            execution_summary=execution_summary
        )
    
    def _analyze_test_file(self, test_file: Path) -> TestFileAnalysis:
        """Analyze a single test file comprehensively"""
        issues = []
        
        # Check if file exists and is readable
        if not test_file.exists():
            issues.append(TestIssue(
                test_file=str(test_file),
                test_name="",
                issue_type="file_missing",
                severity="critical",
                description="Test file does not exist"
            ))
            return TestFileAnalysis(
                file_path=str(test_file),
                total_tests=0,
                passing_tests=0,
                failing_tests=0,
                skipped_tests=0,
                issues=issues,
                imports=[],
                missing_imports=[],
                fixtures_used=[],
                missing_fixtures=[],
                execution_time=0.0,
                has_syntax_errors=True,
                has_import_errors=True
            )
        
        # Analyze syntax and structure
        has_syntax_errors, syntax_issues = self._check_syntax(test_file)
        issues.extend(syntax_issues)
        
        # Analyze dependencies
        imports, missing_imports, fixtures_used, missing_fixtures = \
            self.dependency_analyzer.analyze_dependencies(test_file)
        
        has_import_errors = len(missing_imports) > 0
        
        # Add import issues
        for missing_import in missing_imports:
            issues.append(TestIssue(
                test_file=str(test_file),
                test_name="",
                issue_type="missing_import",
                severity="high",
                description=f"Missing import: {missing_import}",
                suggestion=f"Install or fix import for {missing_import}"
            ))
        
        # Add fixture issues
        for missing_fixture in missing_fixtures:
            issues.append(TestIssue(
                test_file=str(test_file),
                test_name="",
                issue_type="missing_fixture",
                severity="medium",
                description=f"Missing fixture: {missing_fixture}",
                suggestion=f"Define fixture {missing_fixture} in conftest.py"
            ))
        
        # Analyze test structure
        test_count, structure_issues = self._analyze_test_structure(test_file)
        issues.extend(structure_issues)
        
        # Profile performance
        execution_time, slow_test_names, timed_out = \
            self.performance_profiler.profile_test_file(test_file)
        
        if timed_out:
            issues.append(TestIssue(
                test_file=str(test_file),
                test_name="",
                issue_type="timeout",
                severity="critical",
                description="Test file execution timed out",
                suggestion="Investigate hanging tests or reduce test complexity"
            ))
        
        # Run actual tests to get pass/fail counts
        passing, failing, skipped = self._run_tests(test_file)
        
        return TestFileAnalysis(
            file_path=str(test_file),
            total_tests=test_count,
            passing_tests=passing,
            failing_tests=failing,
            skipped_tests=skipped,
            issues=issues,
            imports=imports,
            missing_imports=missing_imports,
            fixtures_used=fixtures_used,
            missing_fixtures=missing_fixtures,
            execution_time=execution_time,
            has_syntax_errors=has_syntax_errors,
            has_import_errors=has_import_errors
        )
    
    def _check_syntax(self, test_file: Path) -> Tuple[bool, List[TestIssue]]:
        """Check file for syntax errors"""
        issues = []
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            return False, issues
            
        except SyntaxError as e:
            issues.append(TestIssue(
                test_file=str(test_file),
                test_name="",
                issue_type="syntax_error",
                severity="critical",
                description=f"Syntax error: {str(e)}",
                line_number=e.lineno,
                suggestion="Fix syntax error in the file"
            ))
            return True, issues
        except Exception as e:
            issues.append(TestIssue(
                test_file=str(test_file),
                test_name="",
                issue_type="parse_error",
                severity="critical",
                description=f"Parse error: {str(e)}",
                suggestion="Check file encoding and structure"
            ))
            return True, issues
    
    def _analyze_test_structure(self, test_file: Path) -> Tuple[int, List[TestIssue]]:
        """Analyze test file structure and identify issues"""
        issues = []
        test_count = 0
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_count += 1
                    
                    # Check for empty test functions
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        issues.append(TestIssue(
                            test_file=str(test_file),
                            test_name=node.name,
                            issue_type="empty_test",
                            severity="medium",
                            description="Test function is empty (only contains pass)",
                            line_number=node.lineno,
                            suggestion="Implement test logic or remove empty test"
                        ))
                    
                    # Check for missing assertions
                    has_assertions = self._has_assertions(node)
                    if not has_assertions:
                        issues.append(TestIssue(
                            test_file=str(test_file),
                            test_name=node.name,
                            issue_type="no_assertions",
                            severity="high",
                            description="Test function has no assertions",
                            line_number=node.lineno,
                            suggestion="Add assertions to verify test behavior"
                        ))
            
            return test_count, issues
            
        except Exception as e:
            issues.append(TestIssue(
                test_file=str(test_file),
                test_name="",
                issue_type="structure_analysis_error",
                severity="medium",
                description=f"Error analyzing test structure: {str(e)}"
            ))
            return 0, issues
    
    def _has_assertions(self, func_node: ast.FunctionDef) -> bool:
        """Check if function contains assertions"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                return True
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr.startswith('assert'):
                        return True
                elif isinstance(node.func, ast.Name):
                    if node.func.id.startswith('assert'):
                        return True
        return False
    
    def _run_tests(self, test_file: Path) -> Tuple[int, int, int]:
        """Run tests and return pass/fail/skip counts"""
        try:
            cmd = [
                sys.executable, '-m', 'pytest', 
                str(test_file), 
                '--tb=no', 
                '-q'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=test_file.parent
            )
            
            # Parse pytest output for counts
            output = result.stdout + result.stderr
            
            # Look for patterns like "1 passed, 2 failed, 1 skipped"
            passed_match = re.search(r'(\d+) passed', output)
            failed_match = re.search(r'(\d+) failed', output)
            skipped_match = re.search(r'(\d+) skipped', output)
            
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            skipped = int(skipped_match.group(1)) if skipped_match else 0
            
            return passed, failed, skipped
            
        except Exception:
            return 0, 0, 0
    
    def _generate_recommendations(self, file_analyses: List[TestFileAnalysis]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Count issue types
        issue_counts = {}
        for analysis in file_analyses:
            for issue in analysis.issues:
                issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        # Generate recommendations based on common issues
        if issue_counts.get('missing_import', 0) > 0:
            recommendations.append(
                f"Fix {issue_counts['missing_import']} missing import issues by installing "
                "required packages or updating import statements"
            )
        
        if issue_counts.get('syntax_error', 0) > 0:
            recommendations.append(
                f"Fix {issue_counts['syntax_error']} syntax errors in test files"
            )
        
        if issue_counts.get('empty_test', 0) > 0:
            recommendations.append(
                f"Implement or remove {issue_counts['empty_test']} empty test functions"
            )
        
        if issue_counts.get('no_assertions', 0) > 0:
            recommendations.append(
                f"Add assertions to {issue_counts['no_assertions']} test functions that lack them"
            )
        
        # Performance recommendations
        slow_files = [fa for fa in file_analyses if fa.execution_time > 10.0]
        if slow_files:
            recommendations.append(
                f"Optimize {len(slow_files)} slow test files that take over 10 seconds to run"
            )
        
        # Coverage recommendations
        total_tests = sum(fa.total_tests for fa in file_analyses)
        if total_tests == 0:
            recommendations.append("No test functions found - consider adding tests to the project")
        
        return recommendations


def main():
    """Main entry point for test auditor"""
    project_root = Path.cwd()
    auditor = TestAuditor(project_root)
    
    print("Starting comprehensive test suite audit...")
    report = auditor.audit_test_suite()
    
    # Save report to file
    report_file = project_root / 'test_audit_report.json'
    with open(report_file, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    print(f"\nAudit complete! Report saved to {report_file}")
    print(f"Total files: {report.total_files}")
    print(f"Total tests: {report.total_tests}")
    print(f"Passing: {report.passing_tests}")
    print(f"Failing: {report.failing_tests}")
    print(f"Critical issues: {len(report.critical_issues)}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")


if __name__ == '__main__':
    main()