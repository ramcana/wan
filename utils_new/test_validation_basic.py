"""
Basic tests for the validation framework components that are working
"""

import unittest
from validation_framework import (
    ValidationResult, ValidationIssue, ValidationSeverity,
    PromptValidator
)

class TestValidationResult(unittest.TestCase):
    """Test ValidationResult class"""
    
    def setUp(self):
        self.result = ValidationResult(is_valid=True)
    
    def test_add_error(self):
        """Test adding error issues"""
        self.result.add_error("Test error", "test_field", "Test suggestion")
        
        self.assertFalse(self.result.is_valid)
        self.assertTrue(self.result.has_errors())
        self.assertEqual(len(self.result.get_errors()), 1)
        
        error = self.result.get_errors()[0]
        self.assertEqual(error.severity, ValidationSeverity.ERROR)
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.field, "test_field")
        self.assertEqual(error.suggestion, "Test suggestion")

        assert True  # TODO: Add proper assertion
    
    def test_add_warning(self):
        """Test adding warning issues"""
        self.result.add_warning("Test warning", "test_field")
        
        self.assertTrue(self.result.is_valid)  # Warnings don't invalidate
        self.assertTrue(self.result.has_warnings())
        self.assertEqual(len(self.result.get_warnings()), 1)

        assert True  # TODO: Add proper assertion
    
    def test_add_info(self):
        """Test adding info issues"""
        self.result.add_info("Test info", "test_field")
        
        self.assertTrue(self.result.is_valid)
        self.assertEqual(len(self.result.get_info()), 1)

        assert True  # TODO: Add proper assertion
    
    def test_to_dict(self):
        """Test serialization to dictionary"""
        self.result.add_error("Error", "field1")
        self.result.add_warning("Warning", "field2")
        
        result_dict = self.result.to_dict()
        
        self.assertFalse(result_dict["is_valid"])
        self.assertEqual(len(result_dict["issues"]), 2)
        self.assertEqual(result_dict["issues"][0]["severity"], "error")
        self.assertEqual(result_dict["issues"][1]["severity"], "warning")

        assert True  # TODO: Add proper assertion

class TestPromptValidator(unittest.TestCase):
    """Test PromptValidator class"""
    
    def setUp(self):
        self.validator = PromptValidator()
    
    def test_valid_prompt(self):
        """Test validation of valid prompts"""
        prompt = "A beautiful landscape with mountains and flowing water"
        result = self.validator.validate_prompt(prompt, "t2v-A14B")
        
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors())

        assert True  # TODO: Add proper assertion
    
    def test_empty_prompt(self):
        """Test validation of empty prompt"""
        result = self.validator.validate_prompt("", "t2v-A14B")
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors())
        
        errors = result.get_errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("empty", errors[0].message.lower())

        assert True  # TODO: Add proper assertion
    
    def test_non_string_prompt(self):
        """Test validation of non-string prompt"""
        result = self.validator.validate_prompt(123, "t2v-A14B")
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors())

        assert True  # TODO: Add proper assertion
    
    def test_prompt_too_short(self):
        """Test validation of too short prompt"""
        result = self.validator.validate_prompt("Hi", "t2v-A14B")
        
        self.assertFalse(result.is_valid)
        errors = result.get_errors()
        self.assertTrue(any("too short" in error.message.lower() for error in errors))

        assert True  # TODO: Add proper assertion
    
    def test_prompt_too_long(self):
        """Test validation of too long prompt"""
        long_prompt = "A" * 600  # Exceeds default max length
        result = self.validator.validate_prompt(long_prompt, "t2v-A14B")
        
        self.assertFalse(result.is_valid)
        errors = result.get_errors()
        self.assertTrue(any("too long" in error.message.lower() for error in errors))

        assert True  # TODO: Add proper assertion
    
    def test_problematic_content_detection(self):
        """Test detection of problematic content"""
        problematic_prompt = "A nude person walking in the park"
        result = self.validator.validate_prompt(problematic_prompt, "t2v-A14B")
        
        # Should still be valid but have warnings
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warnings())
        
        warnings = result.get_warnings()
        self.assertTrue(any("problematic" in warning.message.lower() for warning in warnings))

        assert True  # TODO: Add proper assertion
    
    def test_special_characters(self):
        """Test detection of special characters"""
        prompt_with_special = "A scene with <special> characters and {brackets}"
        result = self.validator.validate_prompt(prompt_with_special, "t2v-A14B")
        
        self.assertTrue(result.has_warnings())

        assert True  # TODO: Add proper assertion
    
    def test_model_specific_validation(self):
        """Test model-specific validation rules"""
        prompt = "Create a static still image of a house"
        result = self.validator.validate_prompt(prompt, "t2v-A14B")
        
        # Should warn about "static" for video generation
        warnings = result.get_warnings()
        self.assertTrue(any("static" in warning.message.lower() for warning in warnings))

        assert True  # TODO: Add proper assertion
    
    def test_encoding_validation(self):
        """Test prompt encoding validation"""
        # Test with valid UTF-8
        valid_prompt = "A beautiful café with naïve art"
        result = self.validator.validate_prompt(valid_prompt, "t2v-A14B")
        self.assertTrue(result.is_valid)
        
        # Test with unusual Unicode
        unusual_prompt = "A scene with 𝕌𝕟𝕚𝕔𝕠𝕕𝕖 characters"
        result = self.validator.validate_prompt(unusual_prompt, "t2v-A14B")
        self.assertTrue(result.has_warnings())

        assert True  # TODO: Add proper assertion
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions"""
        # Short prompt without video terms
        short_prompt = "A house"
        result = self.validator.validate_prompt(short_prompt, "t2v-A14B")
        
        info_messages = result.get_info()
        self.assertTrue(len(info_messages) > 0)
        self.assertTrue(any("motion" in info.message.lower() or "short" in info.message.lower() 
                          for info in info_messages))

        assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)