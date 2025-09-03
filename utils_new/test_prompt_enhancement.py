#!/usr/bin/env python3
"""
Test script for the prompt enhancement system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    enhance_prompt, 
    validate_prompt, 
    get_enhancement_preview, 
    detect_vace_aesthetics
)

def test_basic_enhancement():
    """Test basic prompt enhancement"""
    print("=== Testing Basic Enhancement ===")
    
    test_prompts = [
        "A beautiful sunset over mountains",
        "Person walking in the city",
        "Dragon flying through clouds",
        "Robot in futuristic laboratory"
    ]
    
    for prompt in test_prompts:
        print(f"\nOriginal: {prompt}")
        enhanced = enhance_prompt(prompt)
        print(f"Enhanced: {enhanced}")

def test_vace_detection():
    """Test VACE aesthetic detection"""
    print("\n=== Testing VACE Detection ===")
    
    test_prompts = [
        "A beautiful sunset with VACE aesthetics",
        "Artistic rendering of a forest scene",
        "Experimental visual composition",
        "Regular video of a car driving"
    ]
    
    for prompt in test_prompts:
        is_vace = detect_vace_aesthetics(prompt)
        print(f"'{prompt}' -> VACE detected: {is_vace}")

def test_validation():
    """Test prompt validation"""
    print("\n=== Testing Prompt Validation ===")
    
    test_prompts = [
        "",  # Empty
        "Hi",  # Too short
        "A" * 600,  # Too long
        "Valid prompt for testing",  # Valid
        "Prompt with <invalid> characters",  # Invalid chars
    ]
    
    for prompt in test_prompts:
        is_valid, message = validate_prompt(prompt)
        print(f"'{prompt[:50]}...' -> Valid: {is_valid}, Message: {message}")

def test_enhancement_preview():
    """Test enhancement preview functionality"""
    print("\n=== Testing Enhancement Preview ===")
    
    test_prompt = "A person walking through a magical forest with VACE aesthetics"
    preview = get_enhancement_preview(test_prompt)
    
    print(f"Original prompt: {preview['original_prompt']}")
    print(f"Original length: {preview['original_length']}")
    print(f"Is valid: {preview['is_valid']}")
    print(f"Detected VACE: {preview['detected_vace']}")
    print(f"Detected style: {preview['detected_style']}")
    print(f"Suggested enhancements: {preview['suggested_enhancements']}")
    print(f"Estimated final length: {preview['estimated_final_length']}")

def main():
    """Run all tests"""
    try:
        test_basic_enhancement()
        test_vace_detection()
        test_validation()
        test_enhancement_preview()
        print("\n=== All tests completed successfully! ===")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
traceback.print_exc()

if __name__ == "__main__":
    main()