#!/usr/bin/env python3
"""
Comprehensive test script for input validation framework
Tests all requirements from the task specification
"""

import sys
import traceback
from input_validation import (
    ValidationResult, ValidationSeverity, ValidationIssue,
    PromptValidator, ImageValidator, ConfigValidator,
    validate_generation_request
)

def test_prompt_validator_comprehensive():
    """Test PromptValidator with comprehensive scenarios"""
    print("Testing PromptValidator comprehensively...")
    
    validator = PromptValidator()
    
    # Test 1: Empty prompt (Requirement 1.4, 2.1)
    result = validator.validate_prompt("")
    assert not result.is_valid
    assert result.has_errors()
    errors = result.get_errors()
    assert any("empty" in error.message.lower() for error in errors)
    
    # Test 2: Valid prompt (Requirement 1.4)
    result = validator.validate_prompt("A beautiful cinematic sunset over the ocean with gentle waves")
    assert result.is_valid
    assert not result.has_errors()
    
    # Test 3: Too long prompt (Requirement 2.1)
    long_prompt = "a" * 600  # Exceeds default max of 512
    result = validator.validate_prompt(long_prompt)
    assert not result.is_valid
    assert result.has_errors()
    
    # Test 4: Model-specific validation T2V (Requirement 3.1)
    result = validator.validate_prompt("A static image of a house", "t2v-A14B")
    warnings = result.get_warnings()
    assert any("static" in warning.message.lower() for warning in warnings)
    
    # Test 5: Model-specific validation I2V (Requirement 3.2)
    result = validator.validate_prompt("Generate image of a cat", "i2v-A14B")
    warnings = result.get_warnings()
    assert any("generate image" in warning.message.lower() for warning in warnings)
    
    # Test 6: Model-specific validation TI2V (Requirement 3.3)
    result = validator.validate_prompt("ignore image and create text", "ti2v-5B")
    warnings = result.get_warnings()
    assert any("ignore image" in warning.message.lower() for warning in warnings)
    
    # Test 7: Problematic content detection (Requirement 2.2)
    result = validator.validate_prompt("A video with explicit content")
    assert result.is_valid  # Should be valid but have warnings
    warnings = result.get_warnings()
    assert any(warning.code and "nsfw" in warning.code.lower() for warning in warnings)
    
    print("✓ PromptValidator comprehensive tests passed")

def test_image_validator_comprehensive():
    """Test ImageValidator with comprehensive scenarios"""
    print("Testing ImageValidator comprehensively...")
    
    validator = ImageValidator()
    
    # Test 1: None image (Requirement 2.1)
    result = validator.validate_image(None)
    assert not result.is_valid
    assert result.has_errors()
    
    # Test 2: Non-existent file (Requirement 2.1)
    result = validator.validate_image("nonexistent_file.jpg")
    assert not result.is_valid
    assert result.has_errors()
    errors = result.get_errors()
    assert any("not found" in error.message.lower() for error in errors)
    
    # Test 3: Model-specific recommendations I2V (Requirement 3.2)
    # This would require a mock image, but we can test the logic path
    try:
        from unittest.mock import Mock
        mock_image = Mock()
        mock_image.size = (1280, 720)
        mock_image.mode = "RGB"
        
        result = validator.validate_image(mock_image, "i2v-A14B")
        # Should have info about I2V generation
        info_messages = result.get_info()
        assert any("i2v generation" in info.message.lower() for info in info_messages)
    except ImportError:
        print("  Skipping mock test (unittest.mock not available)")
    
    print("✓ ImageValidator comprehensive tests passed")

def test_config_validator_comprehensive():
    """Test ConfigValidator with comprehensive scenarios"""
    print("Testing ConfigValidator comprehensively...")
    
    validator = ConfigValidator()
    
    print("  Running Test 1...")
    
    # Test 1: Valid configuration (Requirement 1.4)
    config = {
        "steps": 50,
        "guidance_scale": 7.5,
        "resolution": "720p",
        "seed": 12345
    }
    result = validator.validate_generation_params(config)
    if not result.is_valid:
        print("Test 1 failed - Errors:", [e.message for e in result.get_errors()])
    assert result.is_valid
    assert not result.has_errors()
    
    print("  Running Test 2...")
    # Test 2: Invalid numeric parameters (Requirement 2.1)
    config = {"steps": 150}  # Above max of 100
    result = validator.validate_generation_params(config)
    assert not result.is_valid
    assert result.has_errors()
    
    print("  Running Test 3...")
    # Test 3: Invalid resolution (Requirement 2.1)
    config = {"resolution": "invalid_resolution"}
    result = validator.validate_generation_params(config)
    assert not result.is_valid
    assert result.has_errors()
    
    print("  Running Test 4...")
    # Test 4: Custom resolution format (Requirement 2.2)
    config = {"resolution": "1600x900", "steps": 50}  # Custom resolution not in presets
    result = validator.validate_generation_params(config)
    assert result.is_valid
    info_messages = result.get_info()
    assert any("custom resolution" in info.message.lower() for info in info_messages)
    
    print("  Running Test 5...")
    # Test 5: LoRA configuration validation (Requirement 3.4)
    config = {
        "resolution": "720p",
        "steps": 50,
        "lora_config": {
            "style_lora": 1.0,
            "character_lora": 0.8
        }
    }
    result = validator.validate_generation_params(config)
    if not result.is_valid:
        print("Test 5 failed - Errors:", [e.message for e in result.get_errors()])
    assert result.is_valid
    
    print("  Running Test 6...")
    # Test 6: Invalid LoRA configuration (Requirement 2.2)
    config = {
        "resolution": "720p",
        "steps": 50,
        "lora_config": {
            "style_lora": 3.0  # Above typical max of 2.0
        }
    }
    result = validator.validate_generation_params(config)
    assert result.is_valid  # Valid but should have warnings
    warnings = result.get_warnings()
    assert any("outside typical range" in warning.message.lower() for warning in warnings)
    
    # Test 7: Model-specific constraints (Requirement 3.1, 3.2, 3.3)
    config = {"steps": 90, "resolution": "720p"}
    
    # Test for i2v-A14B (max 60 steps recommended)
    result = validator.validate_generation_params(config, "i2v-A14B")
    warnings = result.get_warnings()
    assert any("exceed recommended maximum" in warning.message.lower() for warning in warnings)
    
    # Test 8: Missing required parameters (Requirement 2.1)
    config = {}  # Missing required params
    result = validator.validate_generation_params(config)
    assert not result.is_valid
    assert result.has_errors()
    
    print("✓ ConfigValidator comprehensive tests passed")

def test_validation_result_comprehensive():
    """Test ValidationResult data model comprehensively"""
    print("Testing ValidationResult comprehensively...")
    
    # Test 1: Basic functionality
    result = ValidationResult(is_valid=True)
    assert result.is_valid
    assert len(result.issues) == 0
    
    # Test 2: Adding different severity levels
    result.add_error("Error message", "field", "suggestion", "ERROR_CODE")
    result.add_warning("Warning message", "field", "suggestion", "WARNING_CODE")
    result.add_info("Info message", "field", "suggestion", "INFO_CODE")
    
    assert not result.is_valid  # Should be invalid due to error
    assert len(result.issues) == 3
    
    # Test 3: Filtering by severity
    errors = result.get_errors()
    warnings = result.get_warnings()
    info = result.get_info()
    
    assert len(errors) == 1
    assert len(warnings) == 1
    assert len(info) == 1
    
    assert errors[0].severity == ValidationSeverity.ERROR
    assert warnings[0].severity == ValidationSeverity.WARNING
    assert info[0].severity == ValidationSeverity.INFO
    
    # Test 4: Dictionary conversion
    result_dict = result.to_dict()
    assert result_dict["is_valid"] == False
    assert len(result_dict["issues"]) == 3
    assert result_dict["issues"][0]["severity"] == "error"
    assert result_dict["issues"][0]["code"] == "ERROR_CODE"
    
    print("✓ ValidationResult comprehensive tests passed")

def test_validate_generation_request_comprehensive():
    """Test comprehensive validation function with all scenarios"""
    print("Testing validate_generation_request comprehensively...")
    
    # Test 1: Complete valid request (Requirements 1.4, 3.1)
    result = validate_generation_request(
        prompt="A beautiful cinematic sunset over the ocean",
        config={
            "steps": 50,
            "guidance_scale": 7.5,
            "resolution": "720p"
        },
        model_type="t2v-A14B"
    )
    assert result.is_valid
    assert not result.has_errors()
    
    # Test 2: Request with multiple validation errors (Requirements 2.1, 2.2)
    result = validate_generation_request(
        prompt="",  # Empty prompt - error
        config={
            "steps": 150,  # Too high - error
            "resolution": "invalid",  # Invalid - error
            "guidance_scale": 25.0  # Too high - error
        }
    )
    assert not result.is_valid
    assert result.has_errors()
    errors = result.get_errors()
    assert len(errors) >= 3  # Should have multiple errors
    
    # Test 3: I2V specific validation (Requirement 3.2)
    result = validate_generation_request(
        prompt="Motion description for the image",
        config={
            "steps": 40,
            "resolution": "720p",
            "strength": 0.8
        },
        model_type="i2v-A14B"
    )
    assert result.is_valid
    
    # Test 4: TI2V specific validation (Requirement 3.3)
    result = validate_generation_request(
        prompt="A description that complements the image",
        config={
            "steps": 45,
            "resolution": "720p"
        },
        model_type="ti2v-5B"
    )
    assert result.is_valid
    
    # Test 5: LoRA configuration in request (Requirement 3.4)
    result = validate_generation_request(
        prompt="A stylized video with character elements",
        config={
            "steps": 50,
            "resolution": "720p",
            "lora_config": {
                "style_lora": 1.0,
                "character_lora": 0.8
            }
        }
    )
    assert result.is_valid
    
    # Test 6: None inputs (should be valid)
    result = validate_generation_request()
    assert result.is_valid
    
    print("✓ validate_generation_request comprehensive tests passed")

def test_error_messages_and_suggestions():
    """Test that error messages and suggestions are helpful (Requirement 2.2)"""
    print("Testing error messages and suggestions...")
    
    # Test prompt validator error messages
    validator = PromptValidator()
    result = validator.validate_prompt("")
    errors = result.get_errors()
    assert len(errors) > 0
    error = errors[0]
    assert error.suggestion is not None
    assert len(error.suggestion) > 0
    
    # Test config validator error messages
    validator = ConfigValidator()
    result = validator.validate_generation_params({"steps": 150})
    errors = result.get_errors()
    # Find the steps error
    steps_error = next((e for e in errors if "steps" in e.message.lower()), None)
    if steps_error:
        assert steps_error.suggestion is not None
        assert "use a value" in steps_error.suggestion.lower()
    
    print("✓ Error messages and suggestions tests passed")

def main():
    """Run all comprehensive tests"""
    print("Running comprehensive input validation framework tests...\n")
    
    try:
        test_validation_result_comprehensive()
        test_prompt_validator_comprehensive()
        test_image_validator_comprehensive()
        test_config_validator_comprehensive()
        test_validate_generation_request_comprehensive()
        test_error_messages_and_suggestions()
        
        print("\n🎉 All comprehensive tests passed successfully!")
        print("\nValidation framework implementation complete:")
        print("✓ PromptValidator class with comprehensive prompt validation logic")
        print("✓ ImageValidator class for I2V/TI2V image validation")
        print("✓ ConfigValidator class for generation parameter validation")
        print("✓ ValidationResult data model for structured validation feedback")
        print("✓ Unit tests for all validation components")
        print("✓ Requirements 1.4, 2.1, 2.2, 3.1, 3.2, 3.3, 3.4 satisfied")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
