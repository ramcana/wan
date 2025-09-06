"""
Unit tests for the input validation framework
Tests all validation components: PromptValidator, ImageValidator, ConfigValidator
"""

import unittest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

from validation_framework import (
    ValidationResult, ValidationIssue, ValidationSeverity,
    PromptValidator, ImageValidator, ConfigValidator,
    validate_generation_request
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

class TestImageValidator(unittest.TestCase):
    """Test ImageValidator class"""
    
    def setUp(self):
        self.validator = ImageValidator()
        
        # Create temporary test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Valid test image
        self.valid_image = Image.new('RGB', (1280, 720), color='red')
        self.valid_image_path = os.path.join(self.temp_dir, 'valid.jpg')
        self.valid_image.save(self.valid_image_path, 'JPEG')
        
        # Small test image
        self.small_image = Image.new('RGB', (100, 100), color='blue')
        self.small_image_path = os.path.join(self.temp_dir, 'small.jpg')
        self.small_image.save(self.small_image_path, 'JPEG')
        
        # Large test image
        self.large_image = Image.new('RGB', (3000, 3000), color='green')
        self.large_image_path = os.path.join(self.temp_dir, 'large.jpg')
        self.large_image.save(self.large_image_path, 'JPEG')
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_valid_image_object(self):
        """Test validation of valid PIL Image object"""
        result = self.validator.validate_image(self.valid_image, "i2v-A14B")
        
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors())

        assert True  # TODO: Add proper assertion
    
    def test_valid_image_path(self):
        """Test validation of valid image file path"""
        result = self.validator.validate_image(self.valid_image_path, "i2v-A14B")
        
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors())

        assert True  # TODO: Add proper assertion
    
    def test_nonexistent_image_path(self):
        """Test validation of non-existent image path"""
        result = self.validator.validate_image("/nonexistent/path.jpg", "i2v-A14B")
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors())
        
        errors = result.get_errors()
        self.assertTrue(any("not found" in error.message.lower() for error in errors))

        assert True  # TODO: Add proper assertion
    
    def test_invalid_image_object(self):
        """Test validation of invalid image object"""
        result = self.validator.validate_image("not an image", "i2v-A14B")
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors())

        assert True  # TODO: Add proper assertion
    
    def test_image_too_small(self):
        """Test validation of too small image"""
        result = self.validator.validate_image(self.small_image, "i2v-A14B")
        
        self.assertFalse(result.is_valid)
        errors = result.get_errors()
        self.assertTrue(any("too low" in error.message.lower() for error in errors))

        assert True  # TODO: Add proper assertion
    
    def test_image_too_large(self):
        """Test validation of too large image"""
        result = self.validator.validate_image(self.large_image, "i2v-A14B")
        
        self.assertFalse(result.is_valid)
        errors = result.get_errors()
        self.assertTrue(any("too high" in error.message.lower() for error in errors))

        assert True  # TODO: Add proper assertion
    
    def test_unusual_aspect_ratio(self):
        """Test validation of unusual aspect ratio"""
        narrow_image = Image.new('RGB', (100, 500), color='red')
        result = self.validator.validate_image(narrow_image, "i2v-A14B")
        
        # Should be valid but have warnings
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warnings())
        
        warnings = result.get_warnings()
        self.assertTrue(any("aspect ratio" in warning.message.lower() for warning in warnings))

        assert True  # TODO: Add proper assertion
    
    def test_grayscale_image(self):
        """Test validation of grayscale image"""
        gray_image = Image.new('L', (1280, 720), color=128)
        result = self.validator.validate_image(gray_image, "i2v-A14B")
        
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warnings())
        
        warnings = result.get_warnings()
        self.assertTrue(any("grayscale" in warning.message.lower() for warning in warnings))

        assert True  # TODO: Add proper assertion
    
    def test_palette_mode_image(self):
        """Test validation of palette mode image"""
        palette_image = Image.new('P', (1280, 720))
        result = self.validator.validate_image(palette_image, "i2v-A14B")
        
        self.assertTrue(result.is_valid)
        # Should have info about automatic conversion
        info_messages = result.get_info()
        self.assertTrue(any("palette" in info.message.lower() for info in info_messages))

        assert True  # TODO: Add proper assertion
    
    def test_model_specific_suggestions(self):
        """Test model-specific validation suggestions"""
        # Test I2V specific suggestions
        result_i2v = self.validator.validate_image(self.valid_image, "i2v-A14B")
        info_i2v = result_i2v.get_info()
        self.assertTrue(any("i2v" in info.message.lower() for info in info_i2v))
        
        # Test TI2V specific suggestions
        result_ti2v = self.validator.validate_image(self.valid_image, "ti2v-5B")
        info_ti2v = result_ti2v.get_info()
        self.assertTrue(any("ti2v" in info.message.lower() for info in info_ti2v))

        assert True  # TODO: Add proper assertion

class TestConfigValidator(unittest.TestCase):
    """Test ConfigValidator class"""
    
    def setUp(self):
        self.validator = ConfigValidator()
    
    def test_valid_config(self):
        """Test validation of valid configuration"""
        config = {
            "steps": 50,
            "guidance_scale": 7.5,
            "resolution": "720p",
            "seed": 12345
        }
        
        result = self.validator.validate_generation_params(config, "t2v-A14B")
        
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors())

        assert True  # TODO: Add proper assertion
    
    def test_invalid_steps(self):
        """Test validation of invalid steps parameter"""
        # Test too low
        config_low = {"steps": 0}
        result_low = self.validator.validate_generation_params(config_low, "t2v-A14B")
        self.assertFalse(result_low.is_valid)
        
        # Test too high
        config_high = {"steps": 150}
        result_high = self.validator.validate_generation_params(config_high, "t2v-A14B")
        self.assertFalse(result_high.is_valid)
        
        # Test non-numeric
        config_invalid = {"steps": "fifty"}
        result_invalid = self.validator.validate_generation_params(config_invalid, "t2v-A14B")
        self.assertFalse(result_invalid.is_valid)

        assert True  # TODO: Add proper assertion
    
    def test_invalid_guidance_scale(self):
        """Test validation of invalid guidance scale"""
        config = {"guidance_scale": 25.0}  # Too high
        result = self.validator.validate_generation_params(config, "t2v-A14B")
        
        self.assertFalse(result.is_valid)
        errors = result.get_errors()
        self.assertTrue(any("guidance_scale" in error.field for error in errors))

        assert True  # TODO: Add proper assertion
    
    def test_invalid_resolution(self):
        """Test validation of invalid resolution"""
        # Test unsupported resolution
        config_unsupported = {"resolution": "4K"}
        result_unsupported = self.validator.validate_generation_params(config_unsupported, "t2v-A14B")
        self.assertFalse(result_unsupported.is_valid)
        
        # Test invalid format
        config_invalid = {"resolution": "not_a_resolution"}
        result_invalid = self.validator.validate_generation_params(config_invalid, "t2v-A14B")
        self.assertFalse(result_invalid.is_valid)
        
        # Test custom resolution (should be valid)
        config_custom = {"resolution": "1280x720"}
        result_custom = self.validator.validate_generation_params(config_custom, "t2v-A14B")
        self.assertTrue(result_custom.is_valid)

        assert True  # TODO: Add proper assertion
    
    def test_lora_config_validation(self):
        """Test LoRA configuration validation"""
        # Valid LoRA config
        config_valid = {
            "lora_config": {
                "style_lora": 1.0,
                "character_lora": 0.8
            }
        }
        result_valid = self.validator.validate_generation_params(config_valid, "t2v-A14B")
        self.assertTrue(result_valid.is_valid)
        
        # Invalid LoRA config (not dict)
        config_invalid = {"lora_config": "not_a_dict"}
        result_invalid = self.validator.validate_generation_params(config_invalid, "t2v-A14B")
        self.assertFalse(result_invalid.is_valid)
        
        # Invalid LoRA strength
        config_bad_strength = {
            "lora_config": {
                "test_lora": 5.0  # Too high
            }
        }
        result_bad_strength = self.validator.validate_generation_params(config_bad_strength, "t2v-A14B")
        self.assertTrue(result_bad_strength.has_warnings())

        assert True  # TODO: Add proper assertion
    
    def test_model_type_validation(self):
        """Test model type validation"""
        # Valid model type
        config_valid = {"model_type": "t2v-A14B"}
        result_valid = self.validator.validate_generation_params(config_valid, "t2v-A14B")
        self.assertTrue(result_valid.is_valid)
        
        # Invalid model type
        config_invalid = {"model_type": "unknown_model"}
        result_invalid = self.validator.validate_generation_params(config_invalid, "t2v-A14B")
        self.assertFalse(result_invalid.is_valid)
        
        # Non-string model type
        config_non_string = {"model_type": 123}
        result_non_string = self.validator.validate_generation_params(config_non_string, "t2v-A14B")
        self.assertFalse(result_non_string.is_valid)

        assert True  # TODO: Add proper assertion
    
    def test_required_params_check(self):
        """Test checking for required parameters"""
        # Missing resolution
        config_missing = {"steps": 50}
        result = self.validator.validate_generation_params(config_missing, "t2v-A14B")
        
        self.assertFalse(result.is_valid)
        errors = result.get_errors()
        self.assertTrue(any("missing" in error.message.lower() and "resolution" in error.message.lower() 
                          for error in errors))

        assert True  # TODO: Add proper assertion
    
    def test_parameter_combinations(self):
        """Test validation of parameter combinations"""
        # High steps + high resolution
        config = {
            "steps": 80,
            "resolution": "1080p"
        }
        result = self.validator.validate_generation_params(config, "t2v-A14B")
        
        self.assertTrue(result.is_valid)  # Valid but should have warnings
        self.assertTrue(result.has_warnings())
        
        warnings = result.get_warnings()
        self.assertTrue(any("generation time" in warning.message.lower() for warning in warnings))

        assert True  # TODO: Add proper assertion
    
    def test_model_specific_constraints(self):
        """Test model-specific parameter constraints"""
        # Test steps exceeding model-specific maximum
        config = {"steps": 90}  # Exceeds t2v-A14B max of 80
        result = self.validator.validate_generation_params(config, "t2v-A14B")
        
        self.assertTrue(result.is_valid)  # Valid but should have warnings
        self.assertTrue(result.has_warnings())
        
        warnings = result.get_warnings()
        self.assertTrue(any("exceed" in warning.message.lower() for warning in warnings))

        assert True  # TODO: Add proper assertion

class TestIntegratedValidation(unittest.TestCase):
    """Test the integrated validation function"""
    
    def setUp(self):
        # Create a test image
        self.test_image = Image.new('RGB', (1280, 720), color='red')
    
    def test_complete_valid_request(self):
        """Test validation of complete valid generation request"""
        prompt = "A beautiful landscape with flowing water and mountains"
        params = {
            "steps": 50,
            "guidance_scale": 7.5,
            "resolution": "720p",
            "seed": 12345
        }
        
        result = validate_generation_request(prompt, None, params, "t2v-A14B")
        
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors())

        assert True  # TODO: Add proper assertion
    
    def test_complete_invalid_request(self):
        """Test validation of complete invalid generation request"""
        prompt = ""  # Invalid empty prompt
        params = {
            "steps": 200,  # Invalid high steps
            "resolution": "invalid_res"  # Invalid resolution
        }
        
        result = validate_generation_request(prompt, None, params, "t2v-A14B")
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors())
        
        # Should have errors from both prompt and config validation
        errors = result.get_errors()
        error_fields = [error.field for error in errors]
        self.assertIn("prompt", error_fields)
        self.assertIn("steps", error_fields)
        self.assertIn("resolution", error_fields)

        assert True  # TODO: Add proper assertion
    
    def test_request_with_image(self):
        """Test validation of request with image"""
        prompt = "Transform this image into a dynamic video"
        params = {
            "steps": 50,
            "resolution": "720p",
            "strength": 0.8
        }
        
        result = validate_generation_request(prompt, self.test_image, params, "i2v-A14B")
        
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors())

        assert True  # TODO: Add proper assertion
    
    def test_request_with_invalid_image(self):
        """Test validation of request with invalid image"""
        prompt = "A valid prompt"
        invalid_image = Image.new('RGB', (100, 100), color='red')  # Too small
        params = {"steps": 50, "resolution": "720p"}
        
        result = validate_generation_request(prompt, invalid_image, params, "i2v-A14B")
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors())
        
        # Should have image validation errors
        errors = result.get_errors()
        self.assertTrue(any("image" in error.field for error in errors))

        assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)