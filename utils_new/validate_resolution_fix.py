"""
Validation script for resolution dropdown fix
Tests the complete implementation against requirements
"""

import sys
import logging
from resolution_manager import get_resolution_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_requirement_10_1():
    """Validate Requirement 10.1: t2v-A14B resolution options"""
    print("🔍 Testing Requirement 10.1: t2v-A14B resolution options")
    
    manager = get_resolution_manager()
    options = manager.get_resolution_options("t2v-A14B")
    expected = ["854x480", "480x854", "1280x720", "1280x704", "1920x1080"]
    
    if options == expected:
        print("✅ PASS: t2v-A14B has correct resolution options")
        return True
    else:
        print(f"❌ FAIL: Expected {expected}, got {options}")
        return False

def validate_requirement_10_2():
    """Validate Requirement 10.2: i2v-A14B resolution options"""
    print("🔍 Testing Requirement 10.2: i2v-A14B resolution options")
    
    manager = get_resolution_manager()
    options = manager.get_resolution_options("i2v-A14B")
    expected = ["854x480", "480x854", "1280x720", "1280x704", "1920x1080"]
    
    if options == expected:
        print("✅ PASS: i2v-A14B has correct resolution options")
        return True
    else:
        print(f"❌ FAIL: Expected {expected}, got {options}")
        return False

def validate_requirement_10_3():
    """Validate Requirement 10.3: ti2v-5B resolution options"""
    print("🔍 Testing Requirement 10.3: ti2v-5B resolution options")
    
    manager = get_resolution_manager()
    options = manager.get_resolution_options("ti2v-5B")
    expected = ["854x480", "480x854", "1280x720", "1280x704", "1920x1080", "1024x1024"]
    
    if options == expected:
        print("✅ PASS: ti2v-5B has correct resolution options")
        return True
    else:
        print(f"❌ FAIL: Expected {expected}, got {options}")
        return False

def validate_requirement_10_4():
    """Validate Requirement 10.4: Immediate dropdown updates"""
    print("🔍 Testing Requirement 10.4: Immediate dropdown updates")
    
    manager = get_resolution_manager()
    
    try:
        # Test that updates return immediately
        import time
        start_time = time.time()
        
        result1 = manager.update_resolution_dropdown("t2v-A14B")
        result2 = manager.update_resolution_dropdown("ti2v-5B")
        result3 = manager.update_resolution_dropdown("i2v-A14B")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if elapsed < 0.1:  # Should complete in less than 100ms
            print(f"✅ PASS: Dropdown updates complete immediately ({elapsed:.3f}s)")
            return True
        else:
            print(f"❌ FAIL: Updates took too long ({elapsed:.3f}s)")
            return False
    except Exception as e:
        print(f"❌ FAIL: Error during update: {e}")
        return False

def validate_requirement_10_5():
    """Validate Requirement 10.5: Automatic closest resolution selection"""
    print("🔍 Testing Requirement 10.5: Automatic closest resolution selection")
    
    manager = get_resolution_manager()
    
    # Test unsupported resolution with t2v-A14B (doesn't support 1024x1024)
    closest = manager.find_closest_supported_resolution("1024x1024", "t2v-A14B")
    supported_options = manager.get_resolution_options("t2v-A14B")
    
    if closest in supported_options:
        print(f"✅ PASS: Closest resolution {closest} is supported by t2v-A14B")
        
        # Test validation
        is_valid, message = manager.validate_resolution_compatibility("1024x1024", "t2v-A14B")
        if not is_valid and "not supported" in message:
            print("✅ PASS: Validation correctly identifies unsupported resolution")
            return True
        else:
            print(f"❌ FAIL: Validation should reject 1024x1024 for t2v-A14B: {message}")
            return False
    else:
        print(f"❌ FAIL: Closest resolution {closest} not in supported options {supported_options}")
        return False

def validate_ui_integration():
    """Validate UI integration works correctly"""
    print("🔍 Testing UI Integration")
    
    try:
        # Test that UI can import and use resolution manager
        from resolution_manager import get_resolution_manager
        manager = get_resolution_manager()
        
        # Test model type change simulation
        test_cases = [
            ("t2v-A14B", False, 5),  # (model_type, should_show_images, expected_resolution_count)
            ("i2v-A14B", True, 5),
            ("ti2v-5B", True, 6),
        ]
        
        for model_type, should_show_images, expected_count in test_cases:
            options = manager.get_resolution_options(model_type)
            info = manager.get_resolution_info(model_type)
            
            if len(options) != expected_count:
                print(f"❌ FAIL: {model_type} should have {expected_count} options, got {len(options)}")
                return False
            
            # Check if info contains the model type (case insensitive)
            if model_type.upper() not in info.upper():
                print(f"❌ FAIL: Info text should contain model type {model_type}, got: {info}")
                return False
        
        print("✅ PASS: UI integration works correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: UI integration error: {e}")
        return False

def validate_error_handling():
    """Validate error handling and graceful degradation"""
    print("🔍 Testing Error Handling")
    
    manager = get_resolution_manager()
    
    try:
        # Test with invalid model type
        options = manager.get_resolution_options("invalid-model")
        if not isinstance(options, list) or len(options) == 0:
            print("❌ FAIL: Should return fallback options for invalid model")
            return False
        
        # Test with invalid resolution format
        width, height = manager.get_resolution_dimensions("invalid")
        if width != 1280 or height != 720:
            print("❌ FAIL: Should return fallback dimensions for invalid format")
            return False
        
        # Test validation with empty inputs
        is_valid, message = manager.validate_resolution_compatibility("", "t2v-A14B")
        if not isinstance(is_valid, bool) or not isinstance(message, str):
            print("❌ FAIL: Should handle empty inputs gracefully")
            return False
        
        print("✅ PASS: Error handling works correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error handling failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Starting Resolution Dropdown Fix Validation")
    print("=" * 60)
    
    tests = [
        validate_requirement_10_1,
        validate_requirement_10_2,
        validate_requirement_10_3,
        validate_requirement_10_4,
        validate_requirement_10_5,
        validate_ui_integration,
        validate_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ FAIL: Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Resolution dropdown fix is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)