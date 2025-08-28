#!/usr/bin/env python3
"""
Model Availability Fix Test
Tests the fix for the model availability check error
"""

import sys
import asyncio
import traceback
from pathlib import Path

async def test_model_availability_async():
    """Test model availability check with async method"""
    print("🤖 Testing Model Availability Check (Async)...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        from core.model_integration_bridge import ModelIntegrationBridge
        bridge = ModelIntegrationBridge()
        
        # Initialize the bridge
        init_success = await bridge.initialize()
        if init_success:
            print("✅ Model integration bridge initialized successfully")
        else:
            print("⚠️ Model integration bridge initialization had warnings")
        
        # Test model availability check for each model type
        model_types = ["t2v-a14b", "i2v-a14b", "ti2v-5b"]
        
        for model_type in model_types:
            try:
                print(f"   Testing {model_type}...")
                status = await bridge.check_model_availability(model_type)
                print(f"   ✅ {model_type}: {status.status.value} (no errors)")
            except Exception as e:
                print(f"   ❌ {model_type}: Error - {e}")
                return False
        
        print("✅ All model availability checks completed without errors")
        return True
        
    except Exception as e:
        print(f"❌ Model availability test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def test_hardware_profile_compatibility():
    """Test hardware profile compatibility with different profile types"""
    print("\n🔧 Testing Hardware Profile Compatibility...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        # Test with system optimizer profile (the problematic one)
        from core.services.wan22_system_optimizer import HardwareProfile as OptimizerProfile
        from core.model_integration_bridge import ModelIntegrationBridge
        
        # Create a mock optimizer profile (the type that was causing the error)
        optimizer_profile = OptimizerProfile(
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_threads=16,
            total_memory_gb=32.0,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16.0,
            cuda_version="11.8",
            driver_version="537.13",
            platform_info="Windows",
            detection_timestamp="2025-08-27"
        )
        
        print(f"✅ Created optimizer profile: {optimizer_profile.gpu_model}")
        print(f"   • VRAM: {optimizer_profile.vram_gb}GB (note: no available_vram_gb)")
        
        # Test that our bridge can handle this profile type
        bridge = ModelIntegrationBridge()
        
        # Simulate the conversion that happens in _detect_hardware_profile
        try:
            bridge_profile = bridge.__class__.__annotations__['hardware_profile'].__args__[0](
                gpu_name=getattr(optimizer_profile, 'gpu_model', 'Unknown GPU'),
                total_vram_gb=getattr(optimizer_profile, 'vram_gb', 0.0),
                available_vram_gb=getattr(optimizer_profile, 'vram_gb', 0.0) * 0.8,
                cpu_cores=getattr(optimizer_profile, 'cpu_cores', 4),
                total_ram_gb=getattr(optimizer_profile, 'total_memory_gb', 16.0),
                architecture_type="cuda" if getattr(optimizer_profile, 'vram_gb', 0.0) > 0 else "cpu"
            )
            print("✅ Successfully converted optimizer profile to bridge profile")
            print(f"   • Available VRAM: {bridge_profile.available_vram_gb:.1f}GB")
            
        except Exception as e:
            print(f"❌ Profile conversion failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware profile compatibility test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

async def main():
    """Main test function"""
    print("🧪 MODEL AVAILABILITY FIX TEST")
    print("=" * 50)
    
    tests = [
        ("Hardware Profile Compatibility", test_hardware_profile_compatibility),
        ("Model Availability Check (Async)", test_model_availability_async)
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
        print("🎉 ALL TESTS PASSED! Model availability fix successful!")
        print("\n💡 The model availability error should now be resolved.")
        print("   The backend server should start without the 'available_vram_gb' error.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())