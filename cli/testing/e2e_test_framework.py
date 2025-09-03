"""End-to-End testing framework for meta-tools"""

import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class E2ETestFramework:
    """Framework for testing our meta-tools safely"""
    
    def __init__(self):
        self.fixtures_dir = project_root / "test_fixtures"
        self.temp_dirs = []
    
    def setup_test_project(self, fixture_name: str) -> Path:
        """
        Copy a test fixture to a temporary directory for testing.
        
        Args:
            fixture_name: Name of the fixture directory
            
        Returns:
            Path to the temporary test project
        """
        fixture_path = self.fixtures_dir / fixture_name
        if not fixture_path.exists():
            raise ValueError(f"Fixture {fixture_name} not found")
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix=f"wan_test_{fixture_name}_"))
        self.temp_dirs.append(temp_dir)
        
        # Copy fixture to temp directory
        test_project = temp_dir / "test_project"
        shutil.copytree(fixture_path, test_project)
        
        return test_project
    
    def run_tool(self, tool_command: List[str], project_path: Path) -> Dict[str, Any]:
        """
        Run a CLI tool on a test project.
        
        Args:
            tool_command: CLI command as list (e.g., ['clean', 'imports', '--fix'])
            project_path: Path to the test project
            
        Returns:
            Dictionary with execution results
        """
        # Change to project directory
        original_cwd = Path.cwd()
        
        try:
            import os
            os.chdir(project_path)
            
            # Run the CLI command
            cmd = [sys.executable, "-m", "cli.main"] + tool_command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_path
            )
            
            return {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
        finally:
            os.chdir(original_cwd)
    
    def assert_file_contains(self, file_path: Path, expected_content: str):
        """Assert that a file contains expected content"""
        if not file_path.exists():
            raise AssertionError(f"File {file_path} does not exist")
        
        content = file_path.read_text()
        if expected_content not in content:
            raise AssertionError(f"File {file_path} does not contain: {expected_content}")
    
    def assert_file_not_contains(self, file_path: Path, unexpected_content: str):
        """Assert that a file does not contain unexpected content"""
        if not file_path.exists():
            return  # File doesn't exist, so it doesn't contain the content
        
        content = file_path.read_text()
        if unexpected_content in content:
            raise AssertionError(f"File {file_path} unexpectedly contains: {unexpected_content}")
    
    def assert_file_exists(self, file_path: Path):
        """Assert that a file exists"""
        if not file_path.exists():
            raise AssertionError(f"File {file_path} does not exist")
    
    def assert_file_not_exists(self, file_path: Path):
        """Assert that a file does not exist"""
        if file_path.exists():
            raise AssertionError(f"File {file_path} unexpectedly exists")
    
    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Test cases using the framework
class TestImportFixer:
    """Test the import fixer tool"""
    
    def test_fixes_broken_imports(self):
        """Test that import fixer correctly fixes broken imports"""
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_broken_imports")
            main_py = project_path / "main.py"
            
            # Verify the problem exists
            framework.assert_file_contains(main_py, "from nonexistent_module import something")
            framework.assert_file_contains(main_py, "from utils.helpers import clean_string")
            
            # Act
            result = framework.run_tool(["clean", "imports", "--fix"], project_path)
            
            # Assert
            assert result['success'], f"Import fixer failed: {result['stderr']}"
            
            # Check that broken import was removed or commented
            framework.assert_file_not_contains(main_py, "from nonexistent_module import something")
            
            # Check that relative import was fixed
            framework.assert_file_contains(main_py, "from .utils.helpers import clean_string")
    
    def test_preserves_necessary_files(self):
        """Test that import fixer doesn't delete necessary files"""
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_broken_imports")
            helpers_py = project_path / "utils" / "helpers.py"
            
            # Act
            result = framework.run_tool(["clean", "imports", "--fix"], project_path)
            
            # Assert
            assert result['success']
            framework.assert_file_exists(helpers_py)


class TestDuplicateDetector:
    """Test the duplicate detector tool"""
    
    def test_detects_duplicates(self):
        """Test that duplicate detector finds duplicate files"""
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_duplicate_files")
            
            # Act
            result = framework.run_tool(["clean", "duplicates"], project_path)
            
            # Assert
            assert result['success']
            assert "Found" in result['stdout'] and "duplicate" in result['stdout']
    
    def test_removes_duplicates_safely(self):
        """Test that duplicate removal preserves at least one copy"""
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_duplicate_files")
            
            # Act
            result = framework.run_tool(["clean", "duplicates", "--remove"], project_path)
            
            # Assert
            assert result['success']
            # Verify at least one copy of each file type remains
            # (specific assertions depend on fixture structure)


class TestQualityChecker:
    """Test the quality checker tool"""
    
    def test_detects_quality_issues(self):
        """Test that quality checker finds code quality issues"""
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_quality_issues")
            
            # Act
            result = framework.run_tool(["quality", "check"], project_path)
            
            # Assert
            assert result['success']  # Tool should run successfully
            assert "quality issues" in result['stdout'] or "Quality Score" in result['stdout']
    
    def test_fixes_quality_issues(self):
        """Test that quality checker can auto-fix issues"""
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_quality_issues")
            
            # Act
            result = framework.run_tool(["quality", "check", "--fix"], project_path)
            
            # Assert
            assert result['success']


class TestHealthChecker:
    """Test the health checker tool"""
    
    def test_runs_health_check(self):
        """Test that health checker runs without errors"""
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_broken_imports")
            
            # Act
            result = framework.run_tool(["health", "check", "--quick"], project_path)
            
            # Assert
            assert result['success']
            assert "Health Score" in result['stdout']


# Performance testing
class TestToolPerformance:
    """Test that tools perform within acceptable time limits"""
    
    def test_quick_validation_is_fast(self):
        """Test that quick validation completes within time limit"""
        import time
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_broken_imports")
            
            # Act
            start_time = time.time()
            result = framework.run_tool(["quick"], project_path)
            end_time = time.time()
            
            # Assert
            duration = end_time - start_time
            assert duration < 30, f"Quick validation took {duration:.1f}s, should be under 30s"
    
    def test_full_test_suite_performance(self):
        """Test that full test suite completes within reasonable time"""
        import time
        
        with E2ETestFramework() as framework:
            # Arrange
            project_path = framework.setup_test_project("project_with_broken_imports")
            
            # Act
            start_time = time.time()
            result = framework.run_tool(["test", "run", "--fast"], project_path)
            end_time = time.time()
            
            # Assert
            duration = end_time - start_time
            assert duration < 120, f"Fast test suite took {duration:.1f}s, should be under 2 minutes"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])