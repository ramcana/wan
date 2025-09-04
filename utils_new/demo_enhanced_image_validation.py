from unittest.mock import Mock, patch
"""
Demo script for Enhanced Image Validation System
Demonstrates comprehensive validation, feedback generation, and thumbnail creation
"""

import os
import tempfile
from PIL import Image, ImageDraw
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enhanced_image_validation import (
    EnhancedImageValidator,
    validate_start_image,
    validate_end_image,
    validate_image_pair,
    get_image_validator
)

def create_test_image(width, height, color=(255, 255, 255), format='PNG'):
    """Create a test image with specified dimensions and color"""
    image = Image.new('RGB', (width, height), color)
    
    # Add some content to make it more realistic
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, width-10, height-10], outline=(0, 0, 0), width=2)
    draw.text((width//2-20, height//2), "TEST", fill=(0, 0, 0))
    
    return image

def demo_basic_validation():
    """Demonstrate basic image validation functionality"""
    print("\n" + "="*60)
    print("DEMO: Basic Image Validation")
    print("="*60)
    
    validator = EnhancedImageValidator()
    
    # Test 1: Valid image
    print("\n1. Testing valid image (1280x720)...")
    valid_image = create_test_image(1280, 720)
    result = validator.validate_image_upload(valid_image, "start", "i2v-A14B")
    print(f"Valid: {result.is_valid}, Severity: {result.severity}")
    print(f"Title: {result.title}")
    print(f"Message: {result.message}")
    if result.metadata:
        print(f"Dimensions: {result.metadata.dimensions}")
        print(f"Aspect Ratio: {result.metadata.aspect_ratio_string}")
    
    # Test 2: Too small image
    print("\n2. Testing too small image (100x100)...")
    small_image = create_test_image(100, 100)
    result = validator.validate_image_upload(small_image, "start", "i2v-A14B")
    print(f"Valid: {result.is_valid}, Severity: {result.severity}")
    print(f"Title: {result.title}")
    print(f"Details: {result.details}")
    print(f"Suggestions: {result.suggestions}")
    
    # Test 3: None image
    print("\n3. Testing None image...")
    result = validator.validate_image_upload(None, "start", "i2v-A14B")
    print(f"Valid: {result.is_valid}, Severity: {result.severity}")
    print(f"Title: {result.title}")
    print(f"Message: {result.message}")

def demo_model_specific_validation():
    """Demonstrate model-specific validation"""
    print("\n" + "="*60)
    print("DEMO: Model-Specific Validation")
    print("="*60)
    
    validator = EnhancedImageValidator()
    test_image = create_test_image(400, 400)  # Below recommended size
    
    models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
    
    for model in models:
        print(f"\n--- Testing with {model} ---")
        result = validator.validate_image_upload(test_image, "start", model)
        print(f"Valid: {result.is_valid}, Severity: {result.severity}")
        print(f"Title: {result.title}")
        if result.details:
            print(f"Details: {result.details}")
        if result.suggestions:
            print(f"Suggestions: {result.suggestions[:2]}")  # Show first 2 suggestions

def demo_image_compatibility():
    """Demonstrate image compatibility validation"""
    print("\n" + "="*60)
    print("DEMO: Image Compatibility Validation")
    print("="*60)
    
    validator = EnhancedImageValidator()
    
    # Test 1: Compatible images
    print("\n1. Testing compatible images...")
    start_image = create_test_image(1280, 720, (255, 0, 0))  # Red
    end_image = create_test_image(1280, 720, (0, 255, 0))    # Green
    result = validator.validate_image_compatibility(start_image, end_image)
    print(f"Valid: {result.is_valid}, Severity: {result.severity}")
    print(f"Title: {result.title}")
    print(f"Message: {result.message}")
    
    # Test 2: Incompatible images (different sizes)
    print("\n2. Testing incompatible images (different sizes)...")
    start_image = create_test_image(1280, 720, (255, 0, 0))
    end_image = create_test_image(1920, 1080, (0, 255, 0))
    result = validator.validate_image_compatibility(start_image, end_image)
    print(f"Valid: {result.is_valid}, Severity: {result.severity}")
    print(f"Title: {result.title}")
    print(f"Details: {result.details}")
    print(f"Suggestions: {result.suggestions}")

def demo_html_feedback():
    """Demonstrate HTML feedback generation"""
    print("\n" + "="*60)
    print("DEMO: HTML Feedback Generation")
    print("="*60)
    
    validator = EnhancedImageValidator()
    
    # Test different feedback types
    test_cases = [
        ("Valid image", create_test_image(1280, 720)),
        ("Small image", create_test_image(100, 100)),
        ("None image", None)
    ]
    
    for case_name, image in test_cases:
        print(f"\n--- {case_name} ---")
        result = validator.validate_image_upload(image, "start", "i2v-A14B")
        html = result.to_html()
        
        # Extract key parts of HTML for display
        if "✅" in html:
            print("HTML contains success icon ✅")
        elif "⚠️" in html:
            print("HTML contains warning icon ⚠️")
        elif "❌" in html:
            print("HTML contains error icon ❌")
        
        if result.metadata and result.metadata.thumbnail_data:
            print("HTML includes thumbnail data")
        
        print(f"HTML length: {len(html)} characters")

def demo_convenience_functions():
    """Demonstrate convenience functions"""
    print("\n" + "="*60)
    print("DEMO: Convenience Functions")
    print("="*60)
    
    test_image = create_test_image(1280, 720)
    
    # Test convenience functions
    print("\n1. validate_start_image()...")
    result = validate_start_image(test_image, "i2v-A14B")
    print(f"Result: {result.severity} - {result.title}")
    
    print("\n2. validate_end_image()...")
    result = validate_end_image(test_image, "ti2v-5B")
    print(f"Result: {result.severity} - {result.title}")
    
    print("\n3. validate_image_pair()...")
    start_img = create_test_image(1280, 720, (255, 0, 0))
    end_img = create_test_image(1280, 720, (0, 255, 0))
    result = validate_image_pair(start_img, end_img)
    print(f"Result: {result.severity} - {result.title}")
    
    print("\n4. get_image_validator() with custom config...")
    config = {"max_file_size_mb": 25, "min_dimensions": (512, 512)}
    validator = get_image_validator(config)
    print(f"Custom validator max file size: {validator.max_file_size_mb} MB")
    print(f"Custom validator min dimensions: {validator.min_dimensions}")

def demo_thumbnail_generation():
    """Demonstrate thumbnail generation"""
    print("\n" + "="*60)
    print("DEMO: Thumbnail Generation")
    print("="*60)
    
    validator = EnhancedImageValidator()
    
    # Create a larger test image
    large_image = create_test_image(2048, 1536, (100, 150, 200))
    
    print(f"Original image size: {large_image.size}")
    
    result = validator.validate_image_upload(large_image, "start", "i2v-A14B")
    
    if result.metadata and result.metadata.thumbnail_data:
        print("✅ Thumbnail generated successfully")
        print(f"Thumbnail data length: {len(result.metadata.thumbnail_data)} characters")
        print("Thumbnail format: Base64-encoded PNG")
        
        # The thumbnail data would be used in HTML like:
        # <img src="{result.metadata.thumbnail_data}" alt="Thumbnail" />
        print("Thumbnail ready for HTML display")
    else:
        print("❌ Thumbnail generation failed")

def demo_error_handling():
    """Demonstrate error handling"""
    print("\n" + "="*60)
    print("DEMO: Error Handling")
    print("="*60)
    
    validator = EnhancedImageValidator()
    
    # Test with invalid input
    print("\n1. Testing with invalid input (string instead of image)...")
    try:
        result = validator.validate_image_upload("not_an_image", "start", "i2v-A14B")
        print(f"Result: {result.severity} - {result.title}")
        print(f"Message: {result.message}")
    except Exception as e:
        print(f"Exception handled: {e}")
    
    # Test with mock corrupted image
    print("\n2. Testing error recovery...")
    class MockCorruptedImage:
        @property
        def size(self):
            raise Exception("Corrupted image data")
    
    corrupted = MockCorruptedImage()
    result = validator.validate_image_upload(corrupted, "start", "i2v-A14B")
    print(f"Result: {result.severity} - {result.title}")
    print(f"Error handled gracefully: {result.is_valid}")

def main():
    """Run all demos"""
    print("Enhanced Image Validation System Demo")
    print("=====================================")
    
    try:
        demo_basic_validation()
        demo_model_specific_validation()
        demo_image_compatibility()
        demo_html_feedback()
        demo_convenience_functions()
        demo_thumbnail_generation()
        demo_error_handling()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✅ Comprehensive image validation")
        print("✅ Model-specific requirements")
        print("✅ Image compatibility checking")
        print("✅ Rich HTML feedback generation")
        print("✅ Thumbnail generation")
        print("✅ Error handling and recovery")
        print("✅ Convenience functions for UI integration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo execution failed")

if __name__ == "__main__":
    main()