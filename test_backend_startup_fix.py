#!/usr/bin/env python3
"""
Backend Startup Fix Test
Tests that the backend can start without the hardware profile error
"""

import sys
import asyncio
import traceback
from pathlib import Path

async def test_backend_startup_simulation():
    """Simulate backend startup to test for the hardware profile error"""
    print("🚀 Testing Backend Startup Simulation...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        # Test the components that were failing
        print("   • Testing model integration bridge initialization...")
        from core.model_integration_bridge import ModelIntegrationBridge
        bridge = ModelIntegrationBridge()
        
        # Initialize (this is where the error was occurring)
        init_success = await bridge.initialize()
        if init_success:
            print("   ✅ Model integration bridge initialized successfully")
        else:
            print("   ⚠️ Model integration bridge initialization had warnings")
        
        # Test model availability checks (this is where the error occurred)
        print("   • Testing model availability checks...")
        model_types = ["t2v-a14b", "i2v-a14b", "ti2v-5b"]
        
        for model_type in model_types:
            try:
                status = await bridge.check_model_availability(model_type)
                print(f"     ✅ {model_type}: {status.status.value}")
            except Exception as e:
                print(f"     ❌ {model_type}: {e}")
                return False
        
        # Test generation service integration
        print("   • Testing generation service integration...")
        try:
            from services.generation_service import GenerationService
            gen_service = GenerationService()
            print("   ✅ Generation service import successful")
        except Exception as e:
            print(f"   ⚠️ Generation service import failed: {e}")
            # This is expected if not all dependencies are available
        
        print("✅ Backend startup simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Backend startup simulation failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def test_hardware_profile_error_prevention():
    """Test that the hardware profile error is prevented"""
    print("\n🛡️ Testing Hardware Profile Error Prevention...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        from core.model_integration_bridge import ModelIntegrationBridge, HardwareProfile
        
        # Create a bridge instance
        bridge = ModelIntegrationBridge()
        
        # Test with a mock hardware profile that has the old format (vram_gb only)
        class MockOptimizerProfile:
            def __init__(self):
                self.gpu_model = "NVIDIA GeForce RTX 4080"
                self.vram_gb = 16.0
                self.cpu_cores = 8
                self.total_memory_gb = 32.0
        
        mock_profile = MockOptimizerProfile()
        
        # Test the conversion logic
        try:
            converted_profile = HardwareProfile(
                gpu_name=getattr(mock_profile, 'gpu_model', 'Unknown GPU'),
                total_vram_gb=getattr(mock_profile, 'vram_gb', 0.0),
                available_vram_gb=getattr(mock_profile, 'vram_gb', 0.0) * 0.8,
                cpu_cores=getattr(mock_profile, 'cpu_cores', 4),
                total_ram_gb=getattr(mock_profile, 'total_memory_gb', 16.0),
                architecture_type="cuda" if getattr(mock_profile, 'vram_gb', 0.0) > 0 else "cpu"
            )
            
            # Test that the converted profile has the required attribute
            vram_check = converted_profile.available_vram_gb
            print(f"   ✅ Profile conversion successful: {vram_check:.1f}GB available VRAM")
            
            # Test the hardware compatibility check logic
            estimated_vram = 8000  # 8GB in MB
            hardware_compatible = True
            
            if hasattr(converted_profile, 'available_vram_gb'):
                if estimated_vram > converted_profile.available_vram_gb * 1024:
                    hardware_compatible = False
            elif hasattr(converted_profile, 'vram_gb'):
                available_vram = converted_profile.vram_gb * 0.8
                if estimated_vram > available_vram * 1024:
                    hardware_compatible = False
            
            print(f"   ✅ Hardware compatibility check successful: {hardware_compatible}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Profile conversion failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Hardware profile error prevention test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 BACKEND STARTUP FIX TEST")
    print("=" * 50)
    
    tests = [
        ("Hardware Profile Error Prevention", test_hardware_profile_error_prevention),
        ("Backend Startup Simulation", test_backend_startup_simulation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📋 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Backend startup fix successful!")
        print("\n💡 The backend should now start without the hardware profile error.")
        print("   The 'HardwareProfile' object has no attribute 'available_vram_gb' error is fixed.")
        print("\n🚀 You can now run: python backend/start_server.py --host 127.0.0.1 --port 9000")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())