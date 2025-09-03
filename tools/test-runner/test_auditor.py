"""
Test Auditor - Identifies broken, incomplete, and outdated tests for fixing and categorization
"""

import ast
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re
import importlib.util

from orchestrator import TestCategory, TestConfig

logger = logging.getLogger(__name__)


class TestIssueType(Enum):
    """Types of test issues that can be detected"""
    BROKEN = "broken"  # Test fails to run
    INCOMPLETE = "incomplete"  # Test is not fully implemented
    OUTDATED = "outdated"  # Test references deprecated code
    MISSING_IMPORTS = "missing_imports"  # Import errors
    SYNTAX_ERROR = "syntax_error"  # Python syntax errors
    MISSING_ASSERTIONS = "missing_assertions"  # No assertions in test
    DEPRECATED_API = "deprecated_api"  # Uses deprecated APIs
    UNCATEGORIZED = "uncategorized"  # Not properly categorized


@dataclass
class TestIssue:
    """Represents an issue found in a test"""
    issue_type: TestIssueType
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None


@dataclass
class TestFileAudit:
    """Audit results for a single test file"""
    file_path: Path
    current_category: Optional[TestCategory]
    suggested_category: Optional[TestCategory]
    issues: List[TestIssue] = field(default_factory=list)
    is_runnable: bool = True
    has_tests: bool = True
    test_functions: List[str] = field(default_factory=list)
    
    @property
    def is_broken(self) -> bool:
        """Check if test file has critical issues"""
        return any(issue.severity == 'critical' for issue in self.issues)
    
    @property
    def needs_categorization(self) -> bool:
        """Check if test needs to be moved to correct category"""
        return (self.current_category != self.suggested_category and 
                self.suggested_category is not None)


@dataclass
class AuditReport:
    """Complete audit report for all tests"""
    total_files: int
    broken_files: List[TestFileAudit]
    incomplete_files: List[TestFileAudit]
    miscategorized_files: List[TestFileAudit]
    healthy_files: List[TestFileAudit]
    summary: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate summary statistics"""
        self.summary = {
            'total_files': self.total_files,
            'broken_files': len(self.broken_files),
            'incomplete_files': len(self.incomplete_files),
            'miscategorized_files': len(self.miscategorized_files),
            'healthy_files': len(self.healthy_files),
            'issues_found': sum(len(f.issues) for f in 
                              self.broken_files + self.incomplete_files + self.miscategorized_files)
        }


class TestCodeAnalyzer:
    """Analyzes test code for issues and categorization"""
    
    def __init__(self):
        self.deprecated_patterns = [
            r'unittest\.TestCase',  # Prefer pytest
            r'setUp\(',  # Old unittest style
            r'tearDown\(',  # Old unittest style
            r'assert_\w+\(',  # Old unittest assertions
        ]
        
        self.integration_indicators = [
            'integration', 'api', 'database', 'service', 'endpoint',
            'request', 'response', 'client', 'server'
        ]
        
        self.performance_indicators = [
            'benchmark', 'performance', 'speed', 'timing', 'load',
            'stress', 'memory', 'cpu', 'throughput'
        ]
        
        self.e2e_indicators = [
            'e2e', 'end_to_end', 'workflow', 'scenario', 'journey',
            'full_stack', 'complete'
        ]
    
    def analyze_file(self, file_path: Path) -> TestFileAudit:
        """Analyze a single test file for issues and categorization"""
        logger.debug(f"Analyzing test file: {file_path}")
        
        audit = TestFileAudit(
            file_path=file_path,
            current_category=self._determine_current_category(file_path),
            suggested_category=None
        )
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for syntax errors
            try:
                tree = ast.parse(content)
                audit.issues.extend(self._analyze_ast(tree, content))
            except SyntaxError as e:
                audit.issues.append(TestIssue(
                    issue_type=TestIssueType.SYNTAX_ERROR,
                    severity='critical',
                    description=f"Syntax error: {e.msg}",
                    line_number=e.lineno,
                    suggested_fix="Fix Python syntax error"
                ))
                audit.is_runnable = False
                return audit
            
            # Analyze content
            audit.issues.extend(self._analyze_content(content, file_path))
            audit.test_functions = self._extract_test_functions(tree)
            audit.has_tests = len(audit.test_functions) > 0
            
            # Suggest category
            audit.suggested_category = self._suggest_category(content, file_path)
            
            # Check if file is runnable
            audit.is_runnable = not audit.is_broken
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            audit.issues.append(TestIssue(
                issue_type=TestIssueType.BROKEN,
                severity='critical',
                description=f"Failed to analyze file: {e}",
                suggested_fix="Manual investigation required"
            ))
            audit.is_runnable = False
        
        return audit
    
    def _determine_current_category(self, file_path: Path) -> Optional[TestCategory]:
        """Determine current category based on file location"""
        path_str = str(file_path).lower()
        
        if '/unit/' in path_str or '\\unit\\' in path_str:
            return TestCategory.UNIT
        elif '/integration/' in path_str or '\\integration\\' in path_str:
            return TestCategory.INTEGRATION
        elif '/performance/' in path_str or '\\performance\\' in path_str:
            return TestCategory.PERFORMANCE
        elif '/e2e/' in path_str or '\\e2e\\' in path_str:
            return TestCategory.E2E
        
        return None
    
    def _analyze_ast(self, tree: ast.AST, content: str) -> List[TestIssue]:
        """Analyze AST for structural issues"""
        issues = []
        
        # Check for missing imports
        imports = []
test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
else:
                    imports.append(node.module)

            elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_functions.append(node)
        
        # Check for missing test framework imports
        has_pytest = any('pytest' in imp for imp in imports if imp)
        has_unittest = any('unittest' in imp for imp in imports if imp)
        
        if test_functions and not (has_pytest or has_unittest):
            issues.append(TestIssue(
                issue_type=TestIssueType.MISSING_IMPORTS,
                severity='high',
                description="No test framework imported (pytest or unittest)",
                suggested_fix="Add 'import pytest' or 'import unittest'"
            ))
        
        # Check each test function
        for func in test_functions:
            func_issues = self._analyze_test_function(func)
            issues.extend(func_issues)
        
        return issues
    
    def _analyze_test_function(self, func_node: ast.FunctionDef) -> List[TestIssue]:
        """Analyze individual test function"""
        issues = []
        
        # Check for assertions
        has_assertions = False
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                has_assertions = True
                break
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr.startswith('assert')):
                    has_assertions = True
                    break
        
        if not has_assertions:
            issues.append(TestIssue(
                issue_type=TestIssueType.MISSING_ASSERTIONS,
                severity='medium',
                description=f"Test function '{func_node.name}' has no assertions",
                line_number=func_node.lineno,
                suggested_fix="Add assertions to verify test behavior"
            ))
        
        # Check for incomplete implementation
        for node in ast.walk(func_node):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if node.value.value in ['TODO', 'FIXME', 'NotImplemented']:
                    issues.append(TestIssue(
                        issue_type=TestIssueType.INCOMPLETE,
                        severity='medium',
                        description=f"Test function '{func_node.name}' is incomplete",
                        line_number=node.lineno,
                        suggested_fix="Complete test implementation"
                    ))
            elif isinstance(node, ast.Raise):
                if (isinstance(node.exc, ast.Name) and 
                    node.exc.id == 'NotImplementedError'):
                    issues.append(TestIssue(
                        issue_type=TestIssueType.INCOMPLETE,
                        severity='high',
                        description=f"Test function '{func_node.name}' raises NotImplementedError",
                        line_number=node.lineno,
                        suggested_fix="Implement test logic"
                    ))
        
        return issues
    
    def _analyze_content(self, content: str, file_path: Path) -> List[TestIssue]:
        """Analyze file content for various issues"""
        issues = []
        lines = content.split('\n')
        
        # Check for deprecated patterns
        for i, line in enumerate(lines, 1):
            for pattern in self.deprecated_patterns:
                if re.search(pattern, line):
                    issues.append(TestIssue(
                        issue_type=TestIssueType.DEPRECATED_API,
                        severity='low',
                        description=f"Deprecated pattern found: {pattern}",
                        line_number=i,
                        suggested_fix="Update to modern pytest style"
                    ))
        
        # Check for common issues
        if 'def test_' not in content and 'class Test' not in content:
            issues.append(TestIssue(
                issue_type=TestIssueType.INCOMPLETE,
                severity='high',
                description="No test functions or classes found",
                suggested_fix="Add test functions starting with 'test_'"
            ))
        
        return issues
    
    def _extract_test_functions(self, tree: ast.AST) -> List[str]:
        """Extract names of test functions from AST"""
        test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_functions.append(node.name)
            elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                # Extract methods from test classes
                for item in node.body:
                    if (isinstance(item, ast.FunctionDef) and 
                        item.name.startswith('test_')):
                        test_functions.append(f"{node.name}.{item.name}")
        
        return test_functions
    
    def _suggest_category(self, content: str, file_path: Path) -> Optional[TestCategory]:
        """Suggest appropriate category based on content analysis"""
        content_lower = content.lower()
        path_lower = str(file_path).lower()
        
        # Count indicators for each category
        integration_score = sum(1 for indicator in self.integration_indicators 
                              if indicator in content_lower or indicator in path_lower)
        
        performance_score = sum(1 for indicator in self.performance_indicators 
                              if indicator in content_lower or indicator in path_lower)
        
        e2e_score = sum(1 for indicator in self.e2e_indicators 
                       if indicator in content_lower or indicator in path_lower)
        
        # Determine category based on highest score
        scores = {
            TestCategory.INTEGRATION: integration_score,
            TestCategory.PERFORMANCE: performance_score,
            TestCategory.E2E: e2e_score,
            TestCategory.UNIT: 0  # Default fallback
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        return TestCategory.UNIT  # Default to unit tests


class TestAuditor:
    """
    Main test auditor that identifies broken, incomplete, and miscategorized tests
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.analyzer = TestCodeAnalyzer()
        self.test_root = Path.cwd() / "tests"
    
    def audit_all_tests(self) -> AuditReport:
        """Audit all test files in the project"""
        logger.info("Starting comprehensive test audit")
        
        # Discover all test files
        test_files = self._discover_all_test_files()
        
        # Analyze each file
        file_audits = []
        for test_file in test_files:
            audit = self.analyzer.analyze_file(test_file)
            file_audits.append(audit)
        
        # Categorize results
        broken_files = [audit for audit in file_audits if audit.is_broken]
        incomplete_files = [audit for audit in file_audits 
                          if not audit.is_broken and 
                          any(issue.issue_type == TestIssueType.INCOMPLETE 
                              for issue in audit.issues)]
        miscategorized_files = [audit for audit in file_audits 
                              if audit.needs_categorization]
        healthy_files = [audit for audit in file_audits 
                        if not audit.is_broken and 
                        not any(issue.issue_type == TestIssueType.INCOMPLETE 
                               for issue in audit.issues) and
                        not audit.needs_categorization]
        
        report = AuditReport(
            total_files=len(file_audits),
            broken_files=broken_files,
            incomplete_files=incomplete_files,
            miscategorized_files=miscategorized_files,
            healthy_files=healthy_files
        )
        
        logger.info(f"Audit complete: {report.summary}")
        return report
    
    def _discover_all_test_files(self) -> List[Path]:
        """Discover all Python test files"""
        test_files = []
        
        # Make sure we're looking from the project root
        project_root = Path.cwd()
        test_root = project_root / "tests"
        
        if not test_root.exists():
            logger.warning(f"Test directory not found: {test_root}")
            return []
        
        # Search in all test directories
        for pattern in ['**/test_*.py', '**/test*.py']:
            test_files.extend(test_root.glob(pattern))
        
        # Filter out __pycache__ and other non-test files
        valid_files = []
        for file_path in test_files:
            if ('__pycache__' not in str(file_path) and 
                file_path.suffix == '.py' and
                file_path.is_file() and
                file_path.name.startswith('test_')):
                valid_files.append(file_path)
        
        logger.info(f"Discovered {len(valid_files)} test files in {test_root}")
        return valid_files
    
    def fix_broken_tests(self, report: AuditReport, auto_fix: bool = False) -> Dict[str, Any]:
        """
        Fix broken tests based on audit report
        
        Args:
            report: Audit report with identified issues
            auto_fix: Whether to automatically apply fixes
            
        Returns:
            Dictionary with fix results
        """
        logger.info(f"Fixing {len(report.broken_files)} broken test files")
        
        fix_results = {
            'files_fixed': 0,
            'files_removed': 0,
            'manual_fixes_needed': 0,
            'fixes_applied': []
        }
        
        for audit in report.broken_files:
            try:
                if auto_fix:
                    fixed = self._auto_fix_file(audit)
                    if fixed:
                        fix_results['files_fixed'] += 1
                        fix_results['fixes_applied'].append(str(audit.file_path))
                    else:
                        fix_results['manual_fixes_needed'] += 1
                else:
                    # Generate fix suggestions
                    self._generate_fix_suggestions(audit)
                    fix_results['manual_fixes_needed'] += 1
                    
            except Exception as e:
                logger.error(f"Error fixing {audit.file_path}: {e}")
        
        return fix_results
    
    def _auto_fix_file(self, audit: TestFileAudit) -> bool:
        """Attempt to automatically fix a test file"""
        try:
            with open(audit.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply automatic fixes
            for issue in audit.issues:
                if issue.issue_type == TestIssueType.MISSING_IMPORTS:
                    if 'import pytest' not in content:
                        content = 'import pytest\n' + content
                
                elif issue.issue_type == TestIssueType.SYNTAX_ERROR:
                    # Can't auto-fix syntax errors
                    return False
            
            # Write back if changes were made
            if content != original_content:
                with open(audit.file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Auto-fixed {audit.file_path}")
                return True
            
        except Exception as e:
            logger.error(f"Auto-fix failed for {audit.file_path}: {e}")
        
        return False
    
    def _generate_fix_suggestions(self, audit: TestFileAudit):
        """Generate fix suggestions for a test file"""
        suggestions_file = audit.file_path.with_suffix('.fix_suggestions.md')
        
        content = f"# Fix Suggestions for {audit.file_path}\n\n"
        content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        content += "## Issues Found\n\n"
        for i, issue in enumerate(audit.issues, 1):
            content += f"### Issue {i}: {issue.issue_type.value.title()}\n"
            content += f"- **Severity**: {issue.severity}\n"
            content += f"- **Description**: {issue.description}\n"
            if issue.line_number:
                content += f"- **Line**: {issue.line_number}\n"
            if issue.suggested_fix:
                content += f"- **Suggested Fix**: {issue.suggested_fix}\n"
            content += "\n"
        
        with open(suggestions_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Fix suggestions written to {suggestions_file}")
    
    def categorize_tests(self, report: AuditReport, apply_moves: bool = False) -> Dict[str, Any]:
        """
        Categorize tests into proper directories
        
        Args:
            report: Audit report with categorization suggestions
            apply_moves: Whether to actually move files
            
        Returns:
            Dictionary with categorization results
        """
        logger.info(f"Categorizing {len(report.miscategorized_files)} test files")
        
        categorization_results = {
            'files_moved': 0,
            'moves_planned': 0,
            'move_plan': []
        }
        
        for audit in report.miscategorized_files:
            if audit.suggested_category:
                target_dir = self.test_root / audit.suggested_category.value
                target_path = target_dir / audit.file_path.name
                
                move_info = {
                    'source': str(audit.file_path),
                    'target': str(target_path),
                    'category': audit.suggested_category.value,
                    'reason': f"Content analysis suggests {audit.suggested_category.value} category"
                }
                
                categorization_results['move_plan'].append(move_info)
                categorization_results['moves_planned'] += 1
                
                if apply_moves:
                    try:
                        # Create target directory if it doesn't exist
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Move the file
                        audit.file_path.rename(target_path)
                        
                        categorization_results['files_moved'] += 1
                        logger.info(f"Moved {audit.file_path} to {target_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to move {audit.file_path}: {e}")
        
        return categorization_results
    
    def generate_audit_report(self, report: AuditReport, output_path: Path):
        """Generate comprehensive audit report"""
        content = f"""# Test Suite Audit Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Files**: {report.summary['total_files']}
- **Broken Files**: {report.summary['broken_files']}
- **Incomplete Files**: {report.summary['incomplete_files']}
- **Miscategorized Files**: {report.summary['miscategorized_files']}
- **Healthy Files**: {report.summary['healthy_files']}
- **Total Issues**: {report.summary['issues_found']}

## Broken Files

"""
        
        for audit in report.broken_files:
            content += f"### {audit.file_path}\n"
            content += f"- **Runnable**: {audit.is_runnable}\n"
            content += f"- **Has Tests**: {audit.has_tests}\n"
            content += "- **Issues**:\n"
            for issue in audit.issues:
                content += f"  - {issue.severity.upper()}: {issue.description}\n"
            content += "\n"
        
        content += "## Incomplete Files\n\n"
        for audit in report.incomplete_files:
            content += f"### {audit.file_path}\n"
            incomplete_issues = [i for i in audit.issues 
                               if i.issue_type == TestIssueType.INCOMPLETE]
            for issue in incomplete_issues:
                content += f"- {issue.description}\n"
            content += "\n"
        
        content += "## Miscategorized Files\n\n"
        for audit in report.miscategorized_files:
            content += f"### {audit.file_path}\n"
            content += f"- **Current Category**: {audit.current_category.value if audit.current_category else 'None'}\n"
            content += f"- **Suggested Category**: {audit.suggested_category.value}\n\n"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Audit report generated: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    from datetime import datetime
    
    # Load config
    config = TestConfig.load_from_file(Path("tests/config/test-config.yaml"))
    
    # Create auditor
    auditor = TestAuditor(config)
    
    # Run audit
    report = auditor.audit_all_tests()
    
    # Generate report
    auditor.generate_audit_report(report, Path("test_results/audit_report.md"))
    
    # Fix broken tests (dry run)
    fix_results = auditor.fix_broken_tests(report, auto_fix=False)
    
    # Categorize tests (dry run)
    categorization_results = auditor.categorize_tests(report, apply_moves=False)
    
    print(f"Audit complete: {report.summary}")
    print(f"Fix results: {fix_results}")
    print(f"Categorization results: {categorization_results}")