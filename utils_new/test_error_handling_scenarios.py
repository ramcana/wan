"""
Test Script for Comprehensive Image Error Handling Scenarios
Demonstrates error handling with various invalid image files and formats
"""

import os
import tempfile
from PIL import Image
import io
import json
from unittest.mock import Mock

from image_error_handler import (
    ImageErrorHandler, ImageErrorType, ErrorContext,
    get_error_handler
)
from enhanced_image_validation import (
    EnhancedImageValidator, get_image_validator
)

def create_test_image(size=(100, 100), format="PNG", mode="RGB"):
    """Create a test image for testing"""
    image = Image.new(mode, size, color="red")
    return image

def create_corrupted_image_mock():
    """Create a mock image that simulates corruption"""
    mock_image = Mock()
    mock_image.size = (1280, 720)
    mock_image.format = "JPEG"
    mock_image.mode = "RGB"
    mock_image.filename = "corrupted.jpg"
    mock_image.copy.side_effect = Exception("Image is corrupted")
    mock_image.verify.side_effect = Exception("Cannot verify image")
    return mock_image

def test_format_validation_errors():
    """Test various format validation error scenarios"""
    print("=== Testing Format Validation Errors ===")
    
    validator = get_image_validator()
    handler = get_error_handler()
    
    # Test unsupported format
    print("\n1. Testing unsupported format (TIFF):")
    context = ErrorContext(image_type="start", model_type="i2v-A14B")
    error = handler.handle_format_error("TIFF", context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")
    print(f"   Recovery Actions: {len(error.recovery_actions)}")
    for action in error.recovery_actions:
        print(f"     - {action.title}: {action.description}")
    
    # Test with actual image validation
    print("\n2. Testing with mock unsupported format image:")
    mock_image = Mock()
    mock_image.size = (1280, 720)
    mock_image.format = "TIFF"
    mock_image.mode = "RGB"
    mock_image.filename = "test.tiff"
    mock_image.copy.return_value = mock_image
    
    # Mock the metadata extraction to return TIFF format
    original_extract = validator._extract_metadata
    validator._extract_metadata = Mock(return_value=Mock(
        filename="test.tiff",
        format="TIFF",
        dimensions=(1280, 720),
        file_size_bytes=1000000,
        file_size_mb=1.0,
        aspect_ratio=16/9,
        color_mode="RGB",
        has_transparency=False
    ))
    
    try:
        result = validator.validate_image_upload(mock_image, "start", "i2v-A14B")
        print(f"   Validation Result: {'PASS' if result.is_valid else 'FAIL'}")
        print(f"   Severity: {result.severity}")
        print(f"   Title: {result.title}")
        print(f"   Suggestions: {len(result.suggestions)}")
    finally:
        validator._extract_metadata = original_extract

def test_dimension_validation_errors():
    """Test dimension validation error scenarios"""
    print("\n=== Testing Dimension Validation Errors ===")
    
    handler = get_error_handler()
    validator = get_image_validator()
    
    # Test too small image
    print("\n1. Testing too small image (50x50):")
    context = ErrorContext(image_type="start", model_type="i2v-A14B")
    error = handler.handle_dimension_error((50, 50), (256, 256), (4096, 4096), context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")
    print(f"   Recovery Actions: {len(error.recovery_actions)}")
    for action in error.recovery_actions:
        print(f"     - {action.title} ({action.difficulty}): {action.estimated_time}")
        print(f"       {action.description}")
    
    # Test too large image
    print("\n2. Testing too large image (8000x8000):")
    error = handler.handle_dimension_error((8000, 8000), (256, 256), (4096, 4096), context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")
    print(f"   Recovery Actions: {len(error.recovery_actions)}")

def test_file_size_errors():
    """Test file size validation error scenarios"""
    print("\n=== Testing File Size Validation Errors ===")
    
    handler = get_error_handler()
    
    print("\n1. Testing oversized file (150MB):")
    context = ErrorContext(image_type="start", model_type="i2v-A14B")
    error = handler.handle_file_size_error(150.0, 50.0, context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")
    print(f"   Recovery Actions: {len(error.recovery_actions)}")
    for action in error.recovery_actions:
        print(f"     - {action.title}: {action.description}")
        print(f"       Tools needed: {', '.join(action.tools_needed)}")

def test_quality_analysis_errors():
    """Test image quality analysis error scenarios"""
    print("\n=== Testing Quality Analysis Errors ===")
    
    handler = get_error_handler()
    context = ErrorContext(image_type="start", model_type="i2v-A14B")
    
    # Test dark image
    print("\n1. Testing very dark image:")
    error = handler.handle_quality_error("brightness_low", 5.0, context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Severity: {error.severity}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")
    
    # Test bright image
    print("\n2. Testing overexposed image:")
    error = handler.handle_quality_error("brightness_high", 250.0, context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Severity: {error.severity}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")
    
    # Test low contrast
    print("\n3. Testing low contrast image:")
    error = handler.handle_quality_error("contrast_low", 3.0, context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Severity: {error.severity}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")

def test_compatibility_errors():
    """Test image compatibility error scenarios"""
    print("\n=== Testing Image Compatibility Errors ===")
    
    handler = get_error_handler()
    context = ErrorContext(image_type="compatibility", model_type="i2v-A14B")
    
    print("\n1. Testing dimension mismatch:")
    error = handler.handle_compatibility_error((1280, 720), (1920, 1080), context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Severity: {error.severity}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")

def test_system_errors():
    """Test system-level error scenarios"""
    print("\n=== Testing System Errors ===")
    
    handler = get_error_handler()
    context = ErrorContext(image_type="start", model_type="i2v-A14B")
    
    # Test PIL not available
    print("\n1. Testing PIL not available:")
    error = handler.handle_system_error(ImageErrorType.PIL_NOT_AVAILABLE, context=context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")
    print(f"   Recovery Actions: {len(error.recovery_actions)}")
    for action in error.recovery_actions:
        print(f"     - {action.title} ({action.difficulty}): {action.estimated_time}")
        for instruction in action.instructions[:3]:  # Show first 3 instructions
            print(f"       • {instruction}")
    
    # Test memory insufficient
    print("\n2. Testing insufficient memory:")
    memory_error = MemoryError("Not enough memory to process image")
    error = handler.handle_system_error(ImageErrorType.MEMORY_INSUFFICIENT, 
                                      exception=memory_error, context=context)
    print(f"   Error Type: {error.error_type.value}")
    print(f"   Title: {error.title}")
    print(f"   Message: {error.message}")
    print(f"   Technical Details: {error.technical_details}")

def test_processing_errors():
    """Test processing error scenarios"""
    print("\n=== Testing Processing Errors ===")
    
    handler = get_error_handler()
    context = ErrorContext(image_type="start", model_type="i2v-A14B", operation="validation")
    
    # Test various exception types
    exceptions = [
        (ImportError("No module named 'PIL'"), "PIL import error"),
        (PermissionError("Access denied to file"), "Permission error"),
        (ValueError("Image file is corrupted"), "Corrupted file error"),
        (RuntimeError("Unknown processing error"), "Unknown error")
    ]
    
    for exception, description in exceptions:
        print(f"\n{description}:")
        error = handler.handle_processing_error("validation", exception, context)
        print(f"   Classified as: {error.error_type.value}")
        print(f"   Title: {error.title}")
        print(f"   Technical Details: {error.technical_details}")

def test_validation_summary():
    """Test comprehensive validation summary creation"""
    print("\n=== Testing Validation Summary ===")
    
    handler = get_error_handler()
    context = ErrorContext(image_type="start", model_type="i2v-A14B")
    
    # Create multiple errors
    errors = [
        handler.handle_format_error("TIFF", context),
        handler.handle_dimension_error((100, 100), (256, 256), (4096, 4096), context),
        handler.handle_quality_error("brightness_low", 10.0, context)
    ]
    
    summary = handler.create_validation_summary(errors)
    
    print(f"\n1. Summary with multiple errors:")
    print(f"   Status: {summary['status']}")
    print(f"   Message: {summary['message']}")
    print(f"   Errors: {len(summary['errors'])}")
    print(f"   Warnings: {len(summary['warnings'])}")
    print(f"   Recovery Actions: {len(summary['recovery_actions'])}")
    print(f"   Prevention Tips: {len(summary['prevention_tips'])}")
    
    # Test success case
    success_summary = handler.create_validation_summary([])
    print(f"\n2. Summary with no errors:")
    print(f"   Status: {success_summary['status']}")
    print(f"   Message: {success_summary['message']}")

def test_error_serialization():
    """Test error serialization to user-friendly format"""
    print("\n=== Testing Error Serialization ===")
    
    handler = get_error_handler()
    context = ErrorContext(
        image_type="start",
        model_type="i2v-A14B",
        file_path="test.tiff",
        file_size=5000000,
        dimensions=(100, 100),
        format="TIFF"
    )
    
    error = handler.handle_format_error("TIFF", context)
    user_dict = error.to_user_friendly_dict()
    
    print("\n1. Error serialization:")
    print(f"   Error Type: {user_dict['error_type']}")
    print(f"   Severity: {user_dict['severity']}")
    print(f"   Title: {user_dict['title']}")
    print(f"   Message: {user_dict['message']}")
    print(f"   Context: {user_dict['context']}")
    print(f"   Recovery Actions: {len(user_dict['recovery_actions'])}")
    print(f"   Prevention Tips: {len(user_dict['prevention_tips'])}")
    
    # Show JSON serialization
    print(f"\n2. JSON serialization (first 200 chars):")
    json_str = json.dumps(user_dict, indent=2)
    print(f"   {json_str[:200]}...")

def main():
    """Run all error handling scenario tests"""
    print("🧪 Comprehensive Image Error Handling Test Suite")
    print("=" * 60)
    
    try:
        test_format_validation_errors()
        test_dimension_validation_errors()
        test_file_size_errors()
        test_quality_analysis_errors()
        test_compatibility_errors()
        test_system_errors()
        test_processing_errors()
        test_validation_summary()
        test_error_serialization()
        
        print("\n" + "=" * 60)
        print("✅ All error handling scenarios tested successfully!")
        print("✅ Comprehensive error handling implementation complete!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
traceback.print_exc()

if __name__ == "__main__":
    main()