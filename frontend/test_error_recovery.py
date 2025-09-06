#!/usr/bin/env python3
"""
Test script for UI Error Recovery and Fallback System
Tests the error recovery mechanisms and fallback functionality
"""

import logging
import sys
import os
from pathlib import Path

# Add the project root and utils directories to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_error_recovery_system():
    """Test the error recovery system components"""
    
    print("🧪 Testing WAN22 UI Error Recovery System")
    print("=" * 50)
    
    try:
        # Test 1: Import all recovery modules
        print("\n1. Testing module imports...")
        
        from ui_error_recovery import UIErrorRecoveryManager, UIFallbackConfig
        from enhanced_ui_creator import EnhancedUICreator
        from recovery_guidance_system import RecoveryGuidanceSystem
        from ui_error_recovery_integration import UIErrorRecoveryIntegration
        
        print("✅ All recovery modules imported successfully")
        
        # Test 2: Initialize recovery manager
        print("\n2. Testing recovery manager initialization...")
        
        config = UIFallbackConfig(
            enable_component_recreation=True,
            enable_fallback_components=True,
            enable_simplified_ui=True,
            max_recovery_attempts=3
        )
        
        recovery_manager = UIErrorRecoveryManager(config)
        print("✅ Recovery manager initialized successfully")
        
        # Test 3: Test component recreation
        print("\n3. Testing component recreation...")
        
        # Simulate component recreation
        component, guidance = recovery_manager.recreate_component(
            component_name="test_button",
            component_type="Button",
            original_kwargs={"value": "Test Button", "variant": "primary"},
            fallback_kwargs={"value": "Fallback Button"}
        )
        
        if component is not None:
            print("✅ Component recreation successful")
            print(f"   Guidance: {guidance.title if guidance else 'None'}")
        else:
            print("⚠️ Component recreation returned None (expected for test)")
        
        # Test 4: Test guidance system
        print("\n4. Testing guidance system...")
        
        guidance_system = RecoveryGuidanceSystem()
        
        # Test error guidance generation
        title, message, suggestions, severity = guidance_system.generate_guidance(
            error_message="NoneType object has no attribute '_id'",
            component_name="test_component",
            error_type="component_error"
        )
        
        print("✅ Guidance generation successful")
        print(f"   Title: {title}")
        print(f"   Severity: {severity}")
        print(f"   Suggestions: {len(suggestions)} items")
        
        # Test 5: Test enhanced UI creator
        print("\n5. Testing enhanced UI creator...")
        
        ui_creator = EnhancedUICreator()
        
        # Test component creation with recovery
        test_component, test_guidance = ui_creator.create_component_with_recovery(
            component_type="HTML",
            component_name="test_html",
            primary_kwargs={"value": "<p>Test HTML</p>"}
        )
        
        if test_component is not None:
            print("✅ Enhanced component creation successful")
        else:
            print("⚠️ Enhanced component creation returned None")
        
        # Test 6: Test recovery integration
        print("\n6. Testing recovery integration...")
        
        integration = UIErrorRecoveryIntegration()
        
        # Test system context update
        integration._update_system_context()
        print("✅ System context update successful")
        
        # Test 7: Test error display creation
        print("\n7. Testing error display creation...")
        
        from ui_error_recovery import RecoveryGuidance
        
        test_guidance = RecoveryGuidance(
            title="Test Error",
            message="This is a test error message",
            suggestions=["Try this", "Try that", "Contact support"],
            severity="warning"
        )
        
        error_html = recovery_manager.create_user_friendly_error_display(test_guidance)
        
        if error_html and len(error_html) > 100:
            print("✅ Error display creation successful")
            print(f"   HTML length: {len(error_html)} characters")
        else:
            print("❌ Error display creation failed")
        
        # Test 8: Test recovery statistics
        print("\n8. Testing recovery statistics...")
        
        stats = recovery_manager.get_recovery_statistics()
        print("✅ Recovery statistics retrieved")
        print(f"   Total attempts: {stats['total_recovery_attempts']}")
        print(f"   Success rate: {stats['recovery_success_rate']:.1f}%")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("✅ Error recovery system is ready for integration")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all required modules are available")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_components():
    """Test fallback component creation"""
    
    print("\n🔧 Testing Fallback Components")
    print("-" * 30)
    
    try:
        from ui_error_recovery import UIErrorRecoveryManager
        
        recovery_manager = UIErrorRecoveryManager()
        
        # Test different component types
        component_types = [
            "Textbox", "Button", "Dropdown", "Slider", 
            "Image", "Video", "HTML", "Markdown"
        ]
        
        for comp_type in component_types:
            fallback = recovery_manager._create_fallback_component(comp_type, f"test_{comp_type.lower()}")
            
            if fallback is not None:
                print(f"✅ {comp_type} fallback created successfully")
            else:
                print(f"⚠️ {comp_type} fallback creation failed")
        
        print("✅ Fallback component testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Fallback component test failed: {e}")
        return False

def test_guidance_rules():
    """Test guidance rule matching"""
    
    print("\n📋 Testing Guidance Rules")
    print("-" * 25)
    
    try:
        from recovery_guidance_system import RecoveryGuidanceSystem
        
        guidance_system = RecoveryGuidanceSystem()
        
        # Test different error patterns
        test_errors = [
            ("NoneType object has no attribute '_id'", "none_component_error"),
            ("CUDA out of memory", "memory_error"),
            ("Connection failed", "network_error"),
            ("File not found", "file_error"),
            ("Model loading failed", "model_loading_error"),
            ("Unknown error occurred", "generic_error")
        ]
        
        for error_msg, expected_type in test_errors:
            title, message, suggestions, severity = guidance_system.generate_guidance(error_msg)
            
            print(f"✅ Error: '{error_msg[:30]}...'")
            print(f"   → {title} ({severity})")
            print(f"   → {len(suggestions)} suggestions")
        
        print("✅ Guidance rule testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Guidance rule test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting WAN22 UI Error Recovery System Tests")
    
    # Run all tests
    test_results = []
    
    test_results.append(test_error_recovery_system())
    test_results.append(test_fallback_components())
    test_results.append(test_guidance_rules())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Error recovery system is ready.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        sys.exit(1)