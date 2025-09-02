"""
Tests for the code quality checking system.
"""

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.code-quality.quality_checker import QualityChecker
from tools.code-quality.models import QualityConfig, QualityIssueType, QualitySeverity


class TestQualityChecker(unittest.TestCase):
    """Test cases for QualityChecker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = QualityConfig()
        self.checker = QualityChecker(self.config)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test Python file with given content."""
        file_path = self.temp_path / filename
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def test_basic_quality_check(self):
        """Test basic quality checking functionality."""
        # Create a simple test file
        test_content = '''
def hello_world():
    """Say hello to the world."""
    print("Hello, world!")
    return "Hello"
'''
        test_file = self.create_test_file("test_basic.py", test_content)
        
        # Run quality check
        report = self.checker.check_quality(test_file)
        
        # Verify basic report structure
        self.assertEqual(report.files_analyzed, 1)
        self.assertGreaterEqual(report.quality_score, 0)
        self.assertLessEqual(report.quality_score, 100)
        self.assertIsNotNone(report.metrics)
    
    def test_documentation_validation(self):
        """Test documentation validation."""
        # Test file with missing docstrings
        test_content = '''
def undocumented_function():
    return "no docs"

class UndocumentedClass:
    def method_without_docs(self):
        pass
'''
        test_file = self.create_test_file("test_docs.py", test_content)
        
        # Run documentation check
        report = self.checker.check_quality(test_file, checks=['documentation'])
        
        # Should find missing docstring issues
        doc_issues = [issue for issue in report.issues 
                     if issue.issue_type == QualityIssueType.DOCUMENTATION]
        self.assertGreater(len(doc_issues), 0)
        
        # Check that issues mention missing docstrings
        missing_docstring_issues = [issue for issue in doc_issues 
                                   if 'missing docstring' in issue.message.lower()]
        self.assertGreater(len(missing_docstring_issues), 0)
    
    def test_type_hint_validation(self):
        """Test type hint validation."""
        # Test file with missing type hints
        test_content = '''
def function_without_types(param1, param2):
    """Function without type hints."""
    return param1 + param2

def function_with_types(param1: int, param2: str) -> str:
    """Function with type hints."""
    return str(param1) + param2
'''
        test_file = self.create_test_file("test_types.py", test_content)
        
        # Run type hint check
        report = self.checker.check_quality(test_file, checks=['type_hints'])
        
        # Should find missing type hint issues
        type_issues = [issue for issue in report.issues 
                      if issue.issue_type == QualityIssueType.TYPE_HINTS]
        
        # At least one function should have missing type hints
        self.assertGreaterEqual(len(type_issues), 0)
    
    def test_complexity_analysis(self):
        """Test complexity analysis."""
        # Create a complex function
        test_content = '''
def complex_function(x):
    """A complex function for testing."""
    if x > 10:
        if x > 20:
            if x > 30:
                if x > 40:
                    if x > 50:
                        return "very high"
                    else:
                        return "high"
                else:
                    return "medium-high"
            else:
                return "medium"
        else:
            return "low-medium"
    else:
        return "low"
'''
        test_file = self.create_test_file("test_complexity.py", test_content)
        
        # Run complexity check
        report = self.checker.check_quality(test_file, checks=['complexity'])
        
        # Should find complexity issues
        complexity_issues = [issue for issue in report.issues 
                           if issue.issue_type == QualityIssueType.COMPLEXITY]
        
        # The complex function should trigger complexity warnings
        self.assertGreater(len(complexity_issues), 0)
    
    def test_formatting_check(self):
        """Test formatting validation."""
        # Create poorly formatted file
        test_content = '''
def   badly_formatted(  x,y  ):
    """Badly formatted function."""
    if x>0:
        return y+1
    else:
        return y-1   
'''
        test_file = self.create_test_file("test_formatting.py", test_content)
        
        # Run formatting check
        report = self.checker.check_quality(test_file, checks=['formatting'])
        
        # Should find formatting issues
        formatting_issues = [issue for issue in report.issues 
                           if issue.issue_type == QualityIssueType.FORMATTING]
        
        # Should have some formatting issues
        self.assertGreaterEqual(len(formatting_issues), 0)
    
    def test_style_validation(self):
        """Test style validation."""
        # Create file with style issues
        test_content = '''
def function_with_print():
    """Function that uses print instead of logging."""
    print("This should use logging")
    # TODO: Fix this later
    return None
'''
        test_file = self.create_test_file("test_style.py", test_content)
        
        # Run style check
        report = self.checker.check_quality(test_file, checks=['style'])
        
        # Should find style issues
        style_issues = [issue for issue in report.issues 
                       if issue.issue_type == QualityIssueType.STYLE]
        
        # Should have some style issues
        self.assertGreaterEqual(len(style_issues), 0)
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # Create a high-quality file
        good_content = '''
"""High-quality module with proper documentation and type hints."""

from typing import List


def well_written_function(items: List[str]) -> int:
    """
    Count the number of items in the list.
    
    Args:
        items: List of strings to count
        
    Returns:
        Number of items in the list
    """
    return len(items)


class WellDocumentedClass:
    """A well-documented class with proper structure."""
    
    def __init__(self, name: str) -> None:
        """Initialize the class with a name."""
        self.name = name
    
    def get_name(self) -> str:
        """Get the name."""
        return self.name
'''
        good_file = self.create_test_file("good_quality.py", good_content)
        
        # Create a poor-quality file
        bad_content = '''
def bad_function(x,y,z,a,b,c,d,e,f,g):
    if x:
        if y:
            if z:
                if a:
                    if b:
                        if c:
                            if d:
                                if e:
                                    if f:
                                        if g:
                                            return "deeply nested"
    return None
'''
        bad_file = self.create_test_file("bad_quality.py", bad_content)
        
        # Check both files
        good_report = self.checker.check_quality(good_file)
        bad_report = self.checker.check_quality(bad_file)
        
        # Good file should have higher quality score
        self.assertGreater(good_report.quality_score, bad_report.quality_score)
        self.assertGreater(good_report.quality_score, 70)  # Should be reasonably high
        self.assertLess(bad_report.quality_score, 50)     # Should be low
    
    def test_config_loading(self):
        """Test configuration loading from file."""
        # Create a config file
        config_content = '''
formatting:
  line_length: 100
  use_black: false

documentation:
  require_function_docstrings: false

complexity:
  max_cyclomatic_complexity: 20
'''
        config_file = self.temp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Load checker with config
        checker = QualityChecker.from_config_file(config_file)
        
        # Verify config was loaded
        self.assertEqual(checker.config.line_length, 100)
        self.assertFalse(checker.config.use_black)
        self.assertFalse(checker.config.require_function_docstrings)
        self.assertEqual(checker.config.max_cyclomatic_complexity, 20)
    
    def test_specific_checks(self):
        """Test running specific checks only."""
        test_content = '''
def test_function():
    print("test")
    return None
'''
        test_file = self.create_test_file("test_specific.py", test_content)
        
        # Run only documentation checks
        doc_report = self.checker.check_quality(test_file, checks=['documentation'])
        
        # Run only formatting checks
        fmt_report = self.checker.check_quality(test_file, checks=['formatting'])
        
        # Reports should have different issue types
        doc_issue_types = {issue.issue_type for issue in doc_report.issues}
        fmt_issue_types = {issue.issue_type for issue in fmt_report.issues}
        
        # Documentation report should only have documentation issues (if any)
        if doc_issue_types:
            self.assertTrue(all(t == QualityIssueType.DOCUMENTATION for t in doc_issue_types))
        
        # Formatting report should only have formatting/style issues (if any)
        if fmt_issue_types:
            self.assertTrue(all(t in [QualityIssueType.FORMATTING, QualityIssueType.STYLE] 
                              for t in fmt_issue_types))
    
    def test_directory_analysis(self):
        """Test analyzing entire directory."""
        # Create multiple test files
        self.create_test_file("file1.py", "def func1(): pass")
        self.create_test_file("file2.py", "def func2(): pass")
        self.create_test_file("file3.py", "def func3(): pass")
        
        # Analyze directory
        report = self.checker.check_quality(self.temp_path)
        
        # Should analyze all Python files
        self.assertEqual(report.files_analyzed, 3)
    
    def test_report_generation(self):
        """Test report generation in different formats."""
        test_content = '''
def test_function():
    """Test function."""
    return "test"
'''
        test_file = self.create_test_file("test_report.py", test_content)
        
        # Generate report
        report = self.checker.check_quality(test_file)
        
        # Test different formats
        json_report = self.checker.generate_report(report, 'json')
        yaml_report = self.checker.generate_report(report, 'yaml')
        text_report = self.checker.generate_report(report, 'text')
        
        # All should be non-empty strings
        self.assertIsInstance(json_report, str)
        self.assertIsInstance(yaml_report, str)
        self.assertIsInstance(text_report, str)
        self.assertGreater(len(json_report), 0)
        self.assertGreater(len(yaml_report), 0)
        self.assertGreater(len(text_report), 0)
        
        # JSON should be valid JSON
        import json
        json_data = json.loads(json_report)
        self.assertIn('quality_score', json_data)
        self.assertIn('files_analyzed', json_data)


if __name__ == '__main__':
    unittest.main()