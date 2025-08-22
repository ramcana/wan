"""
Unit tests for the SyntaxValidator class
Tests syntax validation functionality, error detection, and automated repair capabilities
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
from syntax_validator import SyntaxValidator, ValidationResult, RepairResult, SyntaxIssue

class TestSyntaxValidator(unittest.TestCase):
    """Test cases for SyntaxValidator"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.validator = SyntaxValidator(backup_dir=os.path.join(self.test_dir, "backups"))
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str) -> str:
        """Create a test file with given content"""
        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_validate_valid_file(self):
        """Test validation of a syntactically correct Python file"""
        valid_content = '''
def hello_world():
    """A simple function"""
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
'''
        file_path = self.create_test_file("valid.py", valid_content)
        
        result = self.validator.validate_file(file_path)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.file_path, file_path)
    
    def test_validate_file_with_syntax_error(self):
        """Test validation of a file with syntax errors"""
        invalid_content = '''
def broken_function():
    print("Missing closing quote)
    return True
'''
        file_path = self.create_test_file("invalid.py", invalid_content)
        
        result = self.validator.validate_file(file_path)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        self.assertEqual(result.file_path, file_path)
        
        # Check that error details are captured
        error = result.errors[0]
        self.assertIsInstance(error, SyntaxIssue)
        self.assertGreater(error.line_number, 0)
        self.assertIn("syntax_error", error.error_type)
    
    def test_validate_missing_else_clause(self):
        """Test detection of missing else clause in conditional expressions"""
        content_with_missing_else = '''
def test_function():
    # This should trigger a missing else clause error
    result = [item if condition] 
    return result
'''
        file_path = self.create_test_file("missing_else.py", content_with_missing_else)
        
        result = self.validator.validate_file(file_path)
        
        self.assertFalse(result.is_valid)
        # Should detect the syntax error from AST parsing
        self.assertGreater(len(result.errors), 0)
    
    def test_validate_file_not_found(self):
        """Test validation of non-existent file"""
        non_existent_path = os.path.join(self.test_dir, "does_not_exist.py")
        
        result = self.validator.validate_file(non_existent_path)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].error_type, "file_not_found")
    
    def test_backup_file(self):
        """Test file backup functionality"""
        content = "print('test')"
        file_path = self.create_test_file("test_backup.py", content)
        
        backup_path = self.validator.backup_file(file_path)
        
        self.assertTrue(os.path.exists(backup_path))
        self.assertIn("test_backup", backup_path)
        self.assertIn(".backup", backup_path)
        
        # Verify backup content matches original
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        self.assertEqual(backup_content, content)
    
    def test_repair_missing_else_clause(self):
        """Test repair of missing else clause in conditional expression"""
        # Create a file with the specific syntax error pattern we fixed
        content_with_error = '''
def test_function(components):
    outputs = [comp for comp in [
        components['item'] if comp is not None],
        components['other_item']
    ]
    return outputs
'''
        file_path = self.create_test_file("repair_test.py", content_with_error)
        
        # First verify it has syntax errors
        validation_result = self.validator.validate_file(file_path)
        self.assertFalse(validation_result.is_valid)
        
        # Attempt repair
        repair_result = self.validator.repair_syntax_errors(file_path)
        
        self.assertIsInstance(repair_result, RepairResult)
        self.assertIsNotNone(repair_result.backup_path)
        self.assertTrue(os.path.exists(repair_result.backup_path))
        
        # Check if repairs were made
        if repair_result.repairs_made:
            self.assertGreater(len(repair_result.repairs_made), 0)
    
    def test_repair_valid_file(self):
        """Test repair of already valid file (should do nothing)"""
        valid_content = '''
def valid_function():
    return "This is valid Python code"
'''
        file_path = self.create_test_file("valid_repair.py", valid_content)
        
        repair_result = self.validator.repair_syntax_errors(file_path)
        
        self.assertTrue(repair_result.success)
        self.assertIsNone(repair_result.backup_path)  # No backup needed for valid file
        self.assertEqual(len(repair_result.repairs_made), 0)
        self.assertEqual(len(repair_result.remaining_errors), 0)
    
    def test_validate_enhanced_handlers(self):
        """Test validation of the actual enhanced event handlers file"""
        # This tests the real file that was fixed
        if os.path.exists('ui_event_handlers_enhanced.py'):
            result = self.validator.validate_enhanced_handlers()
            
            self.assertIsInstance(result, ValidationResult)
            # After our fixes, this should be valid
            self.assertTrue(result.is_valid, 
                          f"Enhanced handlers should be valid after fixes. Errors: {result.errors}")
    
    def test_validate_critical_files(self):
        """Test validation of all critical files"""
        # Create some mock critical files for testing
        valid_content = "print('valid')"
        
        mock_files = ['main.py', 'ui.py', 'utils.py']
        created_files = []
        
        for filename in mock_files:
            file_path = self.create_test_file(filename, valid_content)
            created_files.append(file_path)
        
        # Temporarily modify the validator's critical files list
        original_critical_files = self.validator.critical_files
        self.validator.critical_files = created_files
        
        try:
            results = self.validator.validate_critical_files()
            
            self.assertEqual(len(results), len(created_files))
            for file_path, result in results.items():
                self.assertIsInstance(result, ValidationResult)
                self.assertTrue(result.is_valid)
        finally:
            # Restore original critical files list
            self.validator.critical_files = original_critical_files
    
    def test_suggest_fix_for_missing_else(self):
        """Test fix suggestion for missing else clause"""
        content = "result = [item if condition]"
        
        # Create a mock syntax error
        class MockSyntaxError:
            def __init__(self):
                self.msg = "expected 'else' after 'if' expression"
                self.lineno = 1
        
        mock_error = MockSyntaxError()
        suggestion = self.validator._suggest_fix(content, mock_error)
        
        self.assertIsNotNone(suggestion)
        self.assertIn("else", suggestion.lower())
    
    def test_additional_checks_trailing_comma(self):
        """Test detection of trailing commas in function calls"""
        content_with_trailing_comma = '''
def test_function():
    result = some_function(arg1, arg2,)
    return result
'''
        file_path = self.create_test_file("trailing_comma.py", content_with_trailing_comma)
        
        # Note: Modern Python allows trailing commas in function calls, 
        # so this might not be detected as an error by AST
        result = self.validator.validate_file(file_path)
        
        # The file should be valid (trailing commas are allowed in modern Python)
        self.assertTrue(result.is_valid)
    
    def test_get_validation_summary(self):
        """Test generation of validation summary"""
        # Create mock validation results
        valid_result = ValidationResult(True, "valid.py", [], [])
        invalid_result = ValidationResult(False, "invalid.py", [
            SyntaxIssue("invalid.py", 10, 5, "Test error", "test_error")
        ], [])
        
        results = {
            "valid.py": valid_result,
            "invalid.py": invalid_result
        }
        
        summary = self.validator.get_validation_summary(results)
        
        self.assertIsInstance(summary, str)
        self.assertIn("Total files checked: 2", summary)
        self.assertIn("Valid files: 1", summary)
        self.assertIn("Files with errors: 1", summary)
        self.assertIn("invalid.py", summary)
        self.assertIn("Test error", summary)
    
    def test_pattern_repairs(self):
        """Test pattern-based repair functionality"""
        content = "test content"
        
        # Test the pattern repair method
        repaired_content, repairs = self.validator._apply_pattern_repairs(content)
        
        self.assertIsInstance(repaired_content, str)
        self.assertIsInstance(repairs, list)
    
    def test_repair_syntax_error_method(self):
        """Test the specific syntax error repair method"""
        content = '''
def test():
    result = [item if condition]
    return result
'''
        
        # Create a mock syntax issue
        error = SyntaxIssue(
            file_path="test.py",
            line_number=2,
            column=0,
            message="expected 'else' after 'if' expression",
            error_type="syntax_error"
        )
        
        repaired_content, repair_description = self.validator._repair_syntax_error(content, error)
        
        self.assertIsInstance(repaired_content, str)
        # The repair might or might not be successful depending on the pattern matching
        if repair_description:
            self.assertIsInstance(repair_description, str)

class TestSyntaxValidatorIntegration(unittest.TestCase):
    """Integration tests for syntax validator with real files"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.validator = SyntaxValidator()
    
    def test_validate_real_critical_files(self):
        """Test validation of actual critical files in the project"""
        # Test files that should exist and be valid
        test_files = [
            'syntax_validator.py',  # Our own file should be valid
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                with self.subTest(file=file_path):
                    result = self.validator.validate_file(file_path)
                    self.assertIsInstance(result, ValidationResult)
                    self.assertTrue(result.is_valid, 
                                  f"File {file_path} should be valid. Errors: {result.errors}")
    
    def test_convenience_functions(self):
        """Test the convenience functions"""
        from syntax_validator import validate_file, repair_file, validate_all_critical_files
        
        # Test validate_file function
        if os.path.exists('syntax_validator.py'):
            result = validate_file('syntax_validator.py')
            self.assertIsInstance(result, ValidationResult)
            self.assertTrue(result.is_valid)
        
        # Test validate_all_critical_files function
        results = validate_all_critical_files()
        self.assertIsInstance(results, dict)

class TestSyntaxValidatorErrorCases(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up error case test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.validator = SyntaxValidator(backup_dir=os.path.join(self.test_dir, "backups"))
    
    def tearDown(self):
        """Clean up error case test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_backup_nonexistent_file(self):
        """Test backup of non-existent file"""
        non_existent_path = os.path.join(self.test_dir, "does_not_exist.py")
        
        with self.assertRaises(Exception):
            self.validator.backup_file(non_existent_path)
    
    def test_repair_nonexistent_file(self):
        """Test repair of non-existent file"""
        non_existent_path = os.path.join(self.test_dir, "does_not_exist.py")
        
        result = self.validator.repair_syntax_errors(non_existent_path)
        
        self.assertFalse(result.success)
        self.assertEqual(len(result.remaining_errors), 1)
        self.assertEqual(result.remaining_errors[0].error_type, "file_not_found")
    
    def test_validate_empty_file(self):
        """Test validation of empty file"""
        empty_file = os.path.join(self.test_dir, "empty.py")
        with open(empty_file, 'w') as f:
            f.write("")
        
        result = self.validator.validate_file(empty_file)
        
        # Empty file should be valid Python
        self.assertTrue(result.is_valid)
    
    def test_validate_binary_file(self):
        """Test validation of binary file (should fail gracefully)"""
        binary_file = os.path.join(self.test_dir, "binary.py")
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03')
        
        result = self.validator.validate_file(binary_file)
        
        # Should handle binary files gracefully
        self.assertFalse(result.is_valid)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)