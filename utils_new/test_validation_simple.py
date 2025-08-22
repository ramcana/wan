#!/usr/bin/env python3
"""
Simple test script for input validation framework
"""

import sys
import traceback
from input_validation import (
    ValidationResult, ValidationSeverity, ValidationIssue,
    PromptValidator, ImageValidator, ConfigValidator,
    validate_generation_request
)

def test_validation_result():
    """Test ValidationResult basic functionality"""
    print("Testing ValidationResult...")
    
    # Test valid result
    result = ValidationResult(is_valid=True)
    assert result.is_valid == True
    assert len(result.issues) == 0
    
    # Test adding error
    result.add_error("Test error", "field", "suggestion")
    assert result.is_valid == False
    assert len(result.issues) == 1
    assert result.has_errors() == True
    
    # Test adding warning
    result.add_warning("Test warning", "field")
    assert len(result.issues) == 2
    assert result.has_warnings() == True
    
    print("✓ ValidationResult tests passed")

def test_prompt_validator():
    """Test PromptValidator functionality"""
    print("Testing PromptValidator...")
    
    validator = PromptValidator()
    
    # Test empty prompt
    result = validator.validate_prompt("")
    assert not result.is_valid
    assert result.has_errors()
    
    # Test valid prompt
    result = validator.validate_prompt("A beautiful sunset over the ocean")
    assert result.is_valid
    assert not result.has_errors()
    
    # Test too long prompt
    long_prompt = "a" * 600
    result = validator.validate_prompt(long_prompt)
    assert not result.is_valid
    assert result.has_errors()
    
    print("✓ PromptValidator tests passed")

def test_image_validator():
    """Test ImageValidator functionality"""
    print("Testing ImageValidator...")
    
    validator = ImageValidator()
    
    # Test None image
    result = validator.validate_image(None)
    assert not result.is_valid
    assert result.has_errors()
    
    # Test non-existent file
    result = validator.validate_image("nonexistent.jpg")
    assert not result.is_valid
    assert result.has_errors()
    
    print("✓ ImageValidator tests passed")

def test_config_validator():
    """Test ConfigValidator functionality"""
    print("Testing ConfigValidator...")
    
    validator = ConfigValidator()
    
    # Test valid config
    config = {
        "steps": 50,
        "guidance_scale": 7.5,
        "resolution": "720p"
    }
    result = validator.validate_generation_params(config)
    assert result.is_valid
    
    # Test invalid steps
    config = {"steps": 150}  # Too high
    result = validator.validate_generation_params(config)
    assert not result.is_valid
    assert result.has_errors()
    
    print("✓ ConfigValidator tests passed")

def test_validate_generation_request():
    """Test comprehensive validation function"""
    print("Testing validate_generation_request...")
    
    # Test valid request
    result = validate_generation_request(
        prompt="A beautiful sunset",
        config={"steps": 50, "resolution": "720p"}
    )
    assert result.is_valid
    
    # Test invalid request
    result = validate_generation_request(
        prompt="",  # Empty
        config={"steps": 150}  # Too high
    )
    assert not result.is_valid
    assert result.has_errors()
    
    print("✓ validate_generation_request tests passed")

def main():
    """Run all tests"""
    print("Running input validation framework tests...\n")
    
    try:
        test_validation_result()
        test_prompt_validator()
        test_image_validator()
        test_config_validator()
        test_validate_generation_request()
        
        print("\n🎉 All tests passed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())