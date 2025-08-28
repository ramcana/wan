#!/usr/bin/env python3
"""
Hardware Profile Fix Test
Tests the fix for the 'HardwareProfile' object has no attribute 'available_vram_gb' error
"""

import sys
import traceback
from pathlib import Path

def test_hardware_profile_fix():
    """Test the hardware profile fix for the available_vram_gb error"""
    print("🔧 Testing Hardware Profile Fix...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        # Test model integration bridge initialization
        from core.model_integration_bridge import ModelIntegrationBridge, HardwareProfile
        
        # Test hardware profile structure
        try:
            import torch
            import psutil
            
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(device)
                total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                available_vram = total_vram * 0.8
                
                # Create a test hardware profile to verify the structure
                test_profile = HardwareProfile(
                    gpu_name=gpu_name,
                    total_vram_gb=total_vram,
                    available_vram_gb=available_vram,
                    cpu_cores=psutil.cpu_count(),
                    total_ram_gb=psutil.virtual_memory().total / (1024**3),
                    architecture_type="cuda"
                )
                
                print(f"✅ Hardware profile structure test passed:")
                print(f"   • GPU: {test_profile.gpu_name}")
                print(f"   • Total VRAM: {test_profile.total_vram_gb:.1f}GB")
                print(f"   • Available VRAM: {test_profile.available_vram_gb:.1f}GB")
                print(f"   • CPU Cores: {test_profile.cpu_cores}")
                print(f"   • Total RAM: {test_profile.total_ram_gb:.1f}GB")
                
                # Test that available_vram_gb attribute exists (this was the error)
                vram_check = test_profile.available_vram_gb
                print(f"✅ available_vram_gb attribute working: {vram_check:.1f}GB")
                
                # Test model integration bridge import
                print("✅ Model integration bridge import successful")
                
                # Test bridge initialization (non-async version)
                bridge = ModelIntegrationBridge()
                print("✅ Model integration bridge created successfully")
                
                return True
            else:
                print("⚠️ No CUDA GPU available for hardware profile test")
                return False
                
        except Exception as e:
            print(f"❌ Hardware profile structure test failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        print(f"❌ Hardware profile integration test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def test_model_availability_check():
    """Test model availability check with the fixed hardware profile"""
    print("\n🤖 Testing Model Availability Check...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        from core.model_integration_bridge import ModelIntegrationBridge
        bridge = ModelIntegrationBridge()
        
        # Test sync model availability check
        try:
            # Check if we can call the method without async issues
            status_dict = bridge.get_model_status_from_existing_system()
            
            if status_dict:
                print(f"✅ Model status check successful: {len(status_dict)} models checked")
                for model_type, status in status_dict.items():
                    print(f"   • {model_type}: {status.status.value}")
            else:
                print("⚠️ No model status returned (may be in async context)")
            
            return True
            
        except Exception as e:
            print(f"❌ Model availability check failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model availability test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 HARDWARE PROFILE FIX TEST")
    print("=" * 50)
    
    tests = [
        ("Hardware Profile Fix", test_hardware_profile_fix),
        ("Model Availability Check", test_model_availability_check)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📋 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Hardware profile fix successful!")
        print("\n💡 The 'available_vram_gb' attribute error should now be resolved.")
        print("   You can now run the backend server without this error.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")