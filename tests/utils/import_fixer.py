import pytest
"""
Test Import Fixer - Automatically fixes common import issues in test files.

This module scans test files for import errors and provides automatic fixes
for common patterns, ensuring consistent and working imports across the test suite.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import importlib.util
import re
from dataclasses import dataclass


@dataclass
class ImportIssue:
    """Represents an import issue found in a test file."""
    file_path: Path
    line_number: int
    import_statement: str
    issue_type: str
    suggested_fix: str
    severity: str  # 'error', 'warning', 'info'


@dataclass
class ImportFixResult:
    """Result of import fixing operation."""
    file_path: Path
    issues_found: List[ImportIssue]
    fixes_applied: List[str]
    success: bool
    error_message: Optional[str] = None


class TestImportFixer:
    """Fixes common import issues in test files."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.test_root = self.project_root / "tests"
        
        # Common import mappings for fixing broken imports
        self.import_mappings = {
            # Startup manager imports
            'startup_manager': 'scripts.startup_manager',
            'startup_manager.config': 'scripts.startup_manager.config',
            'startup_manager.environment_validator': 'scripts.startup_manager.environment_validator',
            'startup_manager.port_manager': 'scripts.startup_manager.port_manager',
            'startup_manager.process_manager': 'scripts.startup_manager.process_manager',
            'startup_manager.recovery_engine': 'scripts.startup_manager.recovery_engine',
            'startup_manager.analytics': 'scripts.startup_manager.analytics',
            'startup_manager.performance_monitor': 'scripts.startup_manager.performance_monitor',
            'startup_manager.windows_utils': 'scripts.startup_manager.windows_utils',
            
            # Test runner imports
            'tools.test_runner': 'tools.test-runner',
            'tools.test_runner.orchestrator': 'tools.test-runner.orchestrator',
            'tools.test_runner.test_auditor': 'tools.test-runner.test_auditor',
            'tools.test_runner.coverage_analyzer': 'tools.test-runner.coverage_analyzer',
            'tools.test_runner.runner_engine': 'tools.test-runner.runner_engine',
            
            # Mock imports
            'tests.performance.mock_startup_manager': 'tests.utils.mock_startup_manager',
            '.mock_startup_manager': 'tests.utils.mock_startup_manager',
        }
        
        # Standard test imports that should be available
        self.standard_imports = {
            'pytest',
            'unittest',
            'unittest.mock',
            'pathlib',
            'tempfile',
            'json',
            'os',
            'sys',
            'time',
            'threading',
            'subprocess',
            'shutil',
            'typing',
        }
    
    def scan_test_files(self) -> List[Path]:
        """Scan for all test files in the test directory."""
        test_files = []
        
        for pattern in ['test_*.py', '*_test.py']:
            test_files.extend(self.test_root.rglob(pattern))
        
        return test_files
    
    def analyze_imports(self, file_path: Path) -> List[ImportIssue]:
        """Analyze imports in a test file and identify issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to find import statements
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        issue = self._check_import(file_path, node.lineno, alias.name)
                        if issue:
                            issues.append(issue)
                
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ''
                    issue = self._check_from_import(file_path, node.lineno, module_name, node.names)
                    if issue:
                        issues.append(issue)
        
        except SyntaxError as e:
            issues.append(ImportIssue(
                file_path=file_path,
                line_number=e.lineno or 0,
                import_statement="",
                issue_type="syntax_error",
                suggested_fix=f"Fix syntax error: {e.msg}",
                severity="error"
            ))
        except Exception as e:
            issues.append(ImportIssue(
                file_path=file_path,
                line_number=0,
                import_statement="",
                issue_type="parse_error",
                suggested_fix=f"Fix parse error: {str(e)}",
                severity="error"
            ))
        
        return issues
    
    def _check_import(self, file_path: Path, line_number: int, module_name: str) -> Optional[ImportIssue]:
        """Check a simple import statement for issues."""
        if not self._can_import_module(module_name):
            # Check if we have a mapping for this import
            if module_name in self.import_mappings:
                return ImportIssue(
                    file_path=file_path,
                    line_number=line_number,
                    import_statement=f"import {module_name}",
                    issue_type="missing_module",
                    suggested_fix=f"import {self.import_mappings[module_name]}",
                    severity="error"
                )
            else:
                return ImportIssue(
                    file_path=file_path,
                    line_number=line_number,
                    import_statement=f"import {module_name}",
                    issue_type="missing_module",
                    suggested_fix=f"Install or fix module: {module_name}",
                    severity="error"
                )
        
        return None
    
    def _check_from_import(self, file_path: Path, line_number: int, module_name: str, names: List[ast.alias]) -> Optional[ImportIssue]:
        """Check a from...import statement for issues."""
        if not self._can_import_module(module_name):
            # Check if we have a mapping for this import
            if module_name in self.import_mappings:
                names_str = ', '.join(alias.name for alias in names)
                return ImportIssue(
                    file_path=file_path,
                    line_number=line_number,
                    import_statement=f"from {module_name} import {names_str}",
                    issue_type="missing_module",
                    suggested_fix=f"from {self.import_mappings[module_name]} import {names_str}",
                    severity="error"
                )
            else:
                names_str = ', '.join(alias.name for alias in names)
                return ImportIssue(
                    file_path=file_path,
                    line_number=line_number,
                    import_statement=f"from {module_name} import {names_str}",
                    issue_type="missing_module",
                    suggested_fix=f"Install or fix module: {module_name}",
                    severity="error"
                )
        
        return None
    
    def _can_import_module(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        if not module_name:
            return True
        
        # Handle relative imports
        if module_name.startswith('.'):
            return True  # Skip relative imports for now
        
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    
    def fix_imports(self, file_path: Path, issues: List[ImportIssue]) -> ImportFixResult:
        """Fix import issues in a test file."""
        fixes_applied = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Sort issues by line number in reverse order to avoid line number shifts
            sorted_issues = sorted(issues, key=lambda x: x.line_number, reverse=True)
            
            for issue in sorted_issues:
                if issue.issue_type == "missing_module" and issue.line_number > 0:
                    # Replace the import line
                    old_line = lines[issue.line_number - 1]
                    new_line = self._generate_fixed_import_line(issue)
                    
                    if new_line and new_line != old_line:
                        lines[issue.line_number - 1] = new_line
                        fixes_applied.append(f"Line {issue.line_number}: {old_line.strip()} -> {new_line.strip()}")
            
            # Write the fixed content back
            if fixes_applied:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            
            return ImportFixResult(
                file_path=file_path,
                issues_found=issues,
                fixes_applied=fixes_applied,
                success=True
            )
        
        except Exception as e:
            return ImportFixResult(
                file_path=file_path,
                issues_found=issues,
                fixes_applied=[],
                success=False,
                error_message=str(e)
            )
    
    def _generate_fixed_import_line(self, issue: ImportIssue) -> Optional[str]:
        """Generate a fixed import line based on the issue."""
        if "import" in issue.suggested_fix:
            # Extract indentation from original line
            original_line = issue.import_statement
            indentation = ""
            
            # Try to preserve indentation
            if original_line:
                indentation = original_line[:len(original_line) - len(original_line.lstrip())]
            
            return f"{indentation}{issue.suggested_fix}\n"
        
        return None
    
    def fix_all_test_files(self) -> Dict[Path, ImportFixResult]:
        """Fix imports in all test files."""
        results = {}
        test_files = self.scan_test_files()
        
        for file_path in test_files:
            issues = self.analyze_imports(file_path)
            if issues:
                result = self.fix_imports(file_path, issues)
                results[file_path] = result
        
        return results
    
    def generate_report(self, results: Dict[Path, ImportFixResult]) -> str:
        """Generate a report of import fixing results."""
        report_lines = [
            "Test Import Fixing Report",
            "=" * 50,
            ""
        ]
        
        total_files = len(results)
        successful_fixes = sum(1 for r in results.values() if r.success)
        total_issues = sum(len(r.issues_found) for r in results.values())
        total_fixes = sum(len(r.fixes_applied) for r in results.values())
        
        report_lines.extend([
            f"Files processed: {total_files}",
            f"Successful fixes: {successful_fixes}",
            f"Total issues found: {total_issues}",
            f"Total fixes applied: {total_fixes}",
            ""
        ])
        
        for file_path, result in results.items():
            relative_path = file_path.relative_to(self.project_root)
            report_lines.append(f"File: {relative_path}")
            
            if result.success:
                if result.fixes_applied:
                    report_lines.append(f"  ✓ Applied {len(result.fixes_applied)} fixes:")
                    for fix in result.fixes_applied:
                        report_lines.append(f"    - {fix}")
                else:
                    report_lines.append("  ✓ No fixes needed")
            else:
                report_lines.append(f"  ✗ Error: {result.error_message}")
            
            if result.issues_found:
                unfixed_issues = [i for i in result.issues_found 
                                if not any(f"Line {i.line_number}" in fix for fix in result.fixes_applied)]
                if unfixed_issues:
                    report_lines.append(f"  ! {len(unfixed_issues)} issues remain:")
                    for issue in unfixed_issues:
                        report_lines.append(f"    - Line {issue.line_number}: {issue.suggested_fix}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def create_mock_startup_manager():
    """Create a mock startup manager for tests that need it."""
    mock_content = '''"""
Mock StartupManager for testing purposes.
"""

from unittest.mock import Mock, MagicMock
from pathlib import Path
from typing import Optional, Dict, Any


class MockStartupManager:
    """Mock implementation of StartupManager for testing."""
    
    def __init__(self, config=None):
        self.config = config or Mock()
        self.environment_validator = Mock()
        self.port_manager = Mock()
        self.process_manager = Mock()
        self.recovery_engine = Mock()
        
        # Set up default mock behaviors
        self._setup_default_mocks()
    
    def _setup_default_mocks(self):
        """Set up default mock behaviors."""
        # Environment validator
        self.environment_validator.validate_all.return_value = Mock(
            is_valid=True,
            issues=[],
            warnings=[],
            system_info={"platform": "test", "python_version": "3.9.0"}
        )
        
        # Port manager
        from unittest.mock import Mock as MockClass
        port_allocation = MockClass()
        port_allocation.backend = 8000
        port_allocation.frontend = 3000
        port_allocation.conflicts_resolved = []
        port_allocation.alternative_ports_used = False
        
        self.port_manager.allocate_ports.return_value = port_allocation
        
        # Process manager
        process_info = MockClass()
        process_info.name = "test"
        process_info.port = 8000
        process_info.pid = 1234
        
        process_result = MockClass()
        process_result.success = True
        process_result.process_info = process_info
        
        self.process_manager.start_backend.return_value = process_result
        self.process_manager.start_frontend.return_value = process_result
        
        # Recovery engine
        recovery_result = MockClass()
        recovery_result.success = True
        recovery_result.action_taken = "mock_recovery"
        recovery_result.message = "Mock recovery successful"
        recovery_result.retry_recommended = False
        
        self.recovery_engine.attempt_recovery.return_value = recovery_result
    
    def start_servers(self):
        """Mock start_servers method."""
        result = Mock()
        result.success = True
        result.backend_info = Mock(port=8000, pid=1234)
        result.frontend_info = Mock(port=3000, pid=5678)
        return result
    
    def stop_servers(self):
        """Mock stop_servers method."""
        result = Mock()
        result.success = True
        return result


# For backward compatibility
StartupManager = MockStartupManager
'''
    
    mock_file_path = Path("tests/utils/mock_startup_manager.py")
    mock_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(mock_file_path, 'w', encoding='utf-8') as f:
        f.write(mock_content)
    
    return mock_file_path


def fix_test_imports():
    """Main function to fix test imports."""
    fixer = TestImportFixer()
    
    # Create mock startup manager if it doesn't exist
    create_mock_startup_manager()
    
    # Fix all imports
    results = fixer.fix_all_test_files()
    
    # Generate and print report
    report = fixer.generate_report(results)
    print(report)
    
    return results


if __name__ == "__main__":
    fix_test_imports()
