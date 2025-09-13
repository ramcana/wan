from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Test script for error handling integration with the main application
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

def test_config_loading():
    """Test configuration loading with error handling"""
    print("Testing configuration loading with error handling...")
    
    try:
        from main import ApplicationConfig
        
        # Test with non-existent config file
        config = ApplicationConfig("non_existent_config.json")
        assert config.config is not None
        print("✓ Config loading with missing file works (uses defaults)")
        
        # Test with invalid JSON
        invalid_config_path = "invalid_config.json"
        with open(invalid_config_path, 'w') as f:
            f.write("{ invalid json content")
        
        config = ApplicationConfig(invalid_config_path)
        assert config.config is not None
        print("✓ Config loading with invalid JSON works (uses defaults)")
        
        # Cleanup
        Path(invalid_config_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"✗ Config loading test failed: {e}")
        return False
    
    return True

def test_model_manager_error_handling():
    """Test model manager error handling"""
    print("\nTesting model manager error handling...")
    
    try:
        from utils import get_model_manager
        
        model_manager = get_model_manager()
        
        # Test with invalid model type
        try:
            model_manager.get_model_id("invalid_model_type")
            # This should work (returns the input as-is)
            print("✓ Model ID handling works")
        except Exception as e:
            print(f"✗ Model ID handling failed: {e}")
            return False
        
        # Test model status for non-existent model
        try:
            status = model_manager.get_model_status("non_existent_model")
            assert isinstance(status, dict)
            print("✓ Model status retrieval works")
        except Exception as e:
            print(f"✗ Model status test failed: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Model manager test failed: {e}")
        return False
    
    return True

def test_vram_optimizer_error_handling():
    """Test VRAM optimizer error handling"""
    print("\nTesting VRAM optimizer error handling...")
    
    try:
        from utils import VRAMOptimizer
        
        # Create config for optimizer
        config = {
            "optimization": {
                "max_vram_usage_gb": 12,
                "vae_tile_size_range": [128, 512]
            }
        }
        
        optimizer = VRAMOptimizer(config)
        
        # Test VRAM usage retrieval
        vram_info = optimizer.get_vram_usage()
        assert isinstance(vram_info, dict)
        assert "used_mb" in vram_info
        print("✓ VRAM usage retrieval works")
        
        # Test with invalid quantization level
        class MockModel:
            def half(self):
                return self
            def to(self, dtype):
                return self
        
        mock_model = MockModel()
        result = optimizer.apply_quantization(mock_model, "invalid_level")
        assert result is not None
        print("✓ Quantization with invalid level works (falls back)")
        
    except Exception as e:
        print(f"✗ VRAM optimizer test failed: {e}")
        return False
    
    return True

def test_error_recovery_manager():
    """Test error recovery manager functionality"""
    print("\nTesting error recovery manager...")
    
    try:
        from error_handler import get_error_recovery_manager, create_error_info
        
        recovery_manager = get_error_recovery_manager()
        
        # Test adding an error
        test_error = ValueError("Test error")
        error_info = create_error_info(test_error, "test_context")
        recovery_manager.add_error(error_info)
        
        # Test getting statistics
        stats = recovery_manager.get_error_statistics()
        assert stats["total_errors"] > 0
        print("✓ Error recovery manager works")
        
        # Test clearing errors
        recovery_manager.error_history.clear()
        stats = recovery_manager.get_error_statistics()
        assert stats["total_errors"] == 0
        print("✓ Error history clearing works")
        
    except Exception as e:
        print(f"✗ Error recovery manager test failed: {e}")
        return False
    
    return True

def test_ui_error_handling():
    """Test UI error handling components"""
    print("\nTesting UI error handling...")
    
    try:
        # Test if UI can be imported without errors
        from ui import Wan22UI
        
        # Create a minimal config file for testing
        test_config = {
            "directories": {
                "models_directory": "models",
                "loras_directory": "loras",
                "outputs_directory": "outputs"
            },
            "optimization": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_vram_usage_gb": 12
            },
            "generation": {
                "default_resolution": "1280x720",
                "default_steps": 50,
                "max_prompt_length": 500
            }
        }
        
        config_path = "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        # Test UI initialization
        ui = Wan22UI(config_path)
        assert ui.config is not None
        print("✓ UI initialization with error handling works")
        
        # Test error display creation
        from error_handler import create_error_info
        test_error = RuntimeError("Test UI error")
        error_info = create_error_info(test_error, "ui_test")
        
        error_html = ui._create_error_display(error_info)
        assert isinstance(error_html, str)
        assert "Test UI error" in error_html
        print("✓ UI error display creation works")
        
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"✗ UI error handling test failed: {e}")
        return False
    
    return True

def main():
    """Run all error handling integration tests"""
    print("Running error handling integration tests...\n")
    
    tests = [
        test_config_loading,
        test_model_manager_error_handling,
        test_vram_optimizer_error_handling,
        test_error_recovery_manager,
        test_ui_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All error handling integration tests passed!")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
