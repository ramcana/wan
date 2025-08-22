"""
Unit tests for the input validation framework
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import io

# Import the validation components
from input_validation import (
    ValidationResult, ValidationSeverity, ValidationIssue,
    PromptValidator, ImageValidator, ConfigValidator,
    validate_generation_request
)


class TestValidationResult:
    """Test ValidationResult data model"""
    
    def test_init_valid(self):
        """Test initialization of valid result"""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert len(result.issues) == 0
    
    def test_add_error(self):
        """Test adding error makes result invalid"""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error", "field", "suggestion", "CODE")
        
        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.issues[0].severity == ValidationSeverity.ERROR
        assert result.issues[0].message == "Test error"
        assert result.issues[0].field == "field"
        assert result.issues[0].suggestion == "suggestion"
        assert result.issues[0].code == "CODE"
    
    def test_add_warning_keeps_valid(self):
        """Test adding warning doesn't make result invalid"""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning", "field")
        
        assert result.is_valid is True
        assert len(result.issues) == 1
        assert result.issues[0].severity == ValidationSeverity.WARNING
    
    def test_has_errors(self):
        """Test has_errors method"""
        result = ValidationResult(is_valid=True)
        assert not result.has_errors()
        
        result.add_warning("Warning", "field")
        assert not result.has_errors()
        
        result.add_error("Error", "field")
        assert result.has_errors()
    
    def test_get_errors(self):
        """Test get_errors method"""
        result = ValidationResult(is_valid=True)
        result.add_warning("Warning", "field")
        result.add_error("Error", "field")
        result.add_info("Info", "field")
        
        errors = result.get_errors()
        assert len(errors) == 1
        assert errors[0].severity == ValidationSeverity.ERROR
    
    def test_get_warnings(self):
        """Test get_warnings method"""
        result = ValidationResult(is_valid=True)
        result.add_warning("Warning", "field")
        result.add_error("Error", "field")
        result.add_info("Info", "field")
        
        warnings = result.get_warnings()
        assert len(warnings) == 1
        assert warnings[0].severity == ValidationSeverity.WARNING
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = ValidationResult(is_valid=True)
        result.add_error("Error", "field", "suggestion", "CODE")
        
        result_dict = result.to_dict()
        assert result_dict["is_valid"] is False
        assert len(result_dict["issues"]) == 1
        assert result_dict["issues"][0]["severity"] == "error"
        assert result_dict["issues"][0]["message"] == "Error"


class TestPromptValidator:
    """Test PromptValidator class"""
    
    def test_init(self):
        """Test validator initialization"""
        validator = PromptValidator()
        assert validator.max_length == 512
        assert validator.min_length == 3
    
    def test_init_with_config(self):
        """Test validator initialization with custom config"""
        config = {"max_prompt_length": 256, "min_prompt_length": 5}
        validator = PromptValidator(config)
        assert validator.max_length == 256
        assert validator.min_length == 5
    
    def test_validate_empty_prompt(self):
        """Test validation of empty prompt"""
        validator = PromptValidator()
        result = validator.validate_prompt("")
        
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("empty" in error.message.lower() for error in errors)
    
    def test_validate_none_prompt(self):
        """Test validation of None prompt"""
        validator = PromptValidator()
        result = validator.validate_prompt(None)
        
        assert not result.is_valid
        assert result.has_errors()
    
    def test_validate_non_string_prompt(self):
        """Test validation of non-string prompt"""
        validator = PromptValidator()
        result = validator.validate_prompt(123)
        
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("string" in error.message.lower() for error in errors)
    
    def test_validate_short_prompt(self):
        """Test validation of too short prompt"""
        validator = PromptValidator()
        result = validator.validate_prompt("hi")
        
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("short" in error.message.lower() for error in errors)
    
    def test_validate_long_prompt(self):
        """Test validation of too long prompt"""
        validator = PromptValidator()
        long_prompt = "a" * 600  # Exceeds default max of 512
        result = validator.validate_prompt(long_prompt)
        
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("long" in error.message.lower() for error in errors)
    
    def test_validate_valid_prompt(self):
        """Test validation of valid prompt"""
        validator = PromptValidator()
        result = validator.validate_prompt("A beautiful sunset over the ocean with gentle waves")
        
        assert result.is_valid
        # May have warnings or info, but no errors
        assert not result.has_errors()
    
    def test_validate_problematic_content(self):
        """Test detection of problematic content"""
        validator = PromptValidator()
        result = validator.validate_prompt("A video with nude content and explicit material")
        
        # Should be valid but have warnings
        assert result.is_valid
        assert result.has_warnings()
        warnings = result.get_warnings()
        assert any("nsfw" in warning.code.lower() if warning.code else False for warning in warnings)
    
    def test_validate_model_specific_t2v(self):
        """Test model-specific validation for T2V"""
        validator = PromptValidator()
        result = validator.validate_prompt("A static image of a house", "t2v-A14B")
        
        # Should have warnings about static content
        warnings = result.get_warnings()
        assert any("static" in warning.message.lower() for warning in warnings)
    
    def test_validate_model_specific_i2v(self):
        """Test model-specific validation for I2V"""
        validator = PromptValidator()
        result = validator.validate_prompt("Generate an image of a cat", "i2v-A14B")
        
        # Should have warnings about image generation terms
        warnings = result.get_warnings()
        assert any("generate image" in warning.message.lower() for warning in warnings)
    
    def test_validate_encoding_issues(self):
        """Test validation of encoding issues"""
        validator = PromptValidator()
        # Test with unusual Unicode characters
        result = validator.validate_prompt("A video with 𝕌𝕟𝕚𝕔𝕠𝕕𝕖 characters")
        
        # Should be valid but may have warnings
        assert result.is_valid
        # Check for Unicode warnings
        warnings = result.get_warnings()
        # May or may not have warnings depending on the specific characters


class TestImageValidator:
    """Test ImageValidator class"""
    
    def test_init(self):
        """Test validator initialization"""
        validator = ImageValidator()
        assert validator.supported_formats == ["JPEG", "PNG", "WEBP", "BMP"]
        assert validator.max_file_size_mb == 50
    
    def test_validate_none_image(self):
        """Test validation of None image"""
        validator = ImageValidator()
        result = validator.validate_image(None)
        
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("none" in error.message.lower() for error in errors)
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file"""
        validator = ImageValidator()
        result = validator.validate_image("nonexistent_file.jpg")
        
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("not found" in error.message.lower() for error in errors)
    
    @patch('input_validation.PILImage')
    def test_validate_mock_image_valid(self, mock_pil):
        """Test validation of valid mock image"""
        # Mock PIL Image
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.mode = "RGB"
        mock_pil.open.return_value = mock_image
        
        validator = ImageValidator()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = tmp.name
        
        try:
            result = validator.validate_image(tmp_path)
            # Should be valid
            assert result.is_valid
            assert not result.has_errors()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    @patch('input_validation.PILImage')
    def test_validate_mock_image_too_small(self, mock_pil):
        """Test validation of too small image"""
        # Mock PIL Image with small dimensions
        mock_image = Mock()
        mock_image.size = (100, 100)  # Below minimum 256x256
        mock_image.mode = "RGB"
        mock_pil.open.return_value = mock_image
        
        validator = ImageValidator()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = tmp.name
        
        try:
            result = validator.validate_image(tmp_path)
            assert not result.is_valid
            assert result.has_errors()
            errors = result.get_errors()
            assert any("too low" in error.message.lower() for error in errors)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    @patch('input_validation.PILImage')
    def test_validate_mock_image_grayscale(self, mock_pil):
        """Test validation of grayscale image"""
        # Mock PIL Image with grayscale mode
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.mode = "L"  # Grayscale
        mock_pil.open.return_value = mock_image
        
        validator = ImageValidator()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = tmp.name
        
        try:
            result = validator.validate_image(tmp_path)
            # Should be valid but have warnings
            assert result.is_valid
            assert result.has_warnings()
            warnings = result.get_warnings()
            assert any("grayscale" in warning.message.lower() for warning in warnings)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestConfigValidator:
    """Test ConfigValidator class"""
    
    def test_init(self):
        """Test validator initialization"""
        validator = ConfigValidator()
        assert "steps" in validator.constraints
        assert "guidance_scale" in validator.constraints
        assert "720p" in validator.supported_resolutions
    
    def test_validate_empty_params(self):
        """Test validation of empty parameters"""
        validator = ConfigValidator()
        result = validator.validate_generation_params({})
        
        # Should have errors for missing required params
        assert not result.is_valid
        assert result.has_errors()
    
    def test_validate_valid_params(self):
        """Test validation of valid parameters"""
        validator = ConfigValidator()
        params = {
            "steps": 50,
            "guidance_scale": 7.5,
            "resolution": "720p",
            "seed": 12345
        }
        result = validator.validate_generation_params(params)
        
        # Should be valid
        assert result.is_valid
        assert not result.has_errors()
    
    def test_validate_invalid_steps(self):
        """Test validation of invalid steps"""
        validator = ConfigValidator()
        params = {"steps": 150}  # Above max of 100
        result = validator.validate_generation_params(params)
        
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("too high" in error.message.lower() for error in errors)
    
    def test_validate_invalid_resolution(self):
        """Test validation of invalid resolution"""
        validator = ConfigValidator()
        params = {"resolution": "invalid_resolution"}
        result = validator.validate_generation_params(params)
        
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("unsupported" in error.message.lower() for error in errors)
    
    def test_validate_custom_resolution(self):
        """Test validation of custom resolution format"""
        validator = ConfigValidator()
        params = {"resolution": "1280x720"}
        result = validator.validate_generation_params(params)
        
        # Should be valid with info message
        assert result.is_valid
        info_messages = result.get_info()
        assert any("custom resolution" in info.message.lower() for info in info_messages)
    
    def test_validate_lora_config(self):
        """Test validation of LoRA configuration"""
        validator = ConfigValidator()
        params = {
            "lora_config": {
                "style_lora": 1.0,
                "character_lora": 0.8
            }
        }
        result = validator.validate_generation_params(params)
        
        # Should be valid
        assert result.is_valid
        assert not result.has_errors()
    
    def test_validate_invalid_lora_config(self):
        """Test validation of invalid LoRA configuration"""
        validator = ConfigValidator()
        params = {
            "lora_config": {
                "style_lora": 3.0  # Above max of 2.0
            }
        }
        result = validator.validate_generation_params(params)
        
        # Should be valid but have warnings
        assert result.is_valid
        assert result.has_warnings()
        warnings = result.get_warnings()
        assert any("outside typical range" in warning.message.lower() for warning in warnings)
    
    def test_validate_model_specific_constraints(self):
        """Test model-specific constraint validation"""
        validator = ConfigValidator()
        params = {
            "steps": 90,  # Above recommended max for i2v-A14B (60)
            "resolution": "720p"
        }
        result = validator.validate_generation_params(params, "i2v-A14B")
        
        # Should be valid but have warnings
        assert result.is_valid
        warnings = result.get_warnings()
        assert any("exceed recommended maximum" in warning.message.lower() for warning in warnings)


class TestValidateGenerationRequest:
    """Test the comprehensive validation function"""
    
    def test_validate_complete_request(self):
        """Test validation of complete generation request"""
        prompt = "A beautiful sunset over the ocean"
        config = {
            "steps": 50,
            "guidance_scale": 7.5,
            "resolution": "720p"
        }
        
        result = validate_generation_request(prompt=prompt, config=config)
        
        # Should be valid
        assert result.is_valid
        assert not result.has_errors()
    
    def test_validate_request_with_errors(self):
        """Test validation of request with errors"""
        prompt = ""  # Empty prompt
        config = {
            "steps": 150,  # Too high
            "resolution": "invalid"  # Invalid resolution
        }
        
        result = validate_generation_request(prompt=prompt, config=config)
        
        # Should have errors
        assert not result.is_valid
        assert result.has_errors()
        
        # Should have errors from both prompt and config validation
        errors = result.get_errors()
        assert len(errors) >= 2  # At least one from prompt, one from config
    
    def test_validate_request_none_inputs(self):
        """Test validation with None inputs"""
        result = validate_generation_request()
        
        # Should be valid since all inputs are optional
        assert result.is_valid
        assert not result.has_errors()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])