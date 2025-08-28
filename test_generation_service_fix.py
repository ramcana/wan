#!/usr/bin/env python3
"""
Generation Service Fix Test
Tests the generation service with the fixed model integration bridge
"""

import sys
import asyncio
import traceback
from pathlib import Path

async def test_generation_service_integration():
    """Test generation service integration with the fixed bridge"""
    print("🎬 Testing Generation Service Integration...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        # Test generation service import
        from services.generation_service import GenerationService
        gen_service = GenerationService()
        print("✅ Generation service imported successfully")
        
        # Test model integration bridge within generation service
        from core.model_integration_bridge import ModelIntegrationBridge
        bridge = ModelIntegrationBridge()
        
        # Initialize the bridge
        init_success = await bridge.initialize()
        if init_success:
            print("✅ Model integration bridge initialized in generation context")
        else:
            print("⚠️ Model integration bridge initialization had warnings")
        
        # Test model availability check (this was failing before)
        try:
            status = await bridge.check_model_availability("t2v-a14b")
            print(f"✅ Model availability check successful: {status.status.value}")
        except Exception as e:
            print(f"❌ Model availability check failed: {e}")
            return False
        
        # Test model loading with fallback
        try:
            success, message = await bridge.load_model_with_optimization("t2v-a14b")
            if success:
                print(f"✅ Model loading successful: {message}")
            else:
                print(f"⚠️ Model loading failed (expected): {message}")
                # This is expected since we don't have full model infrastructure
        except Exception as e:
            print(f"❌ Model loading crashed: {e}")
            return False
        
        print("✅ Generation service integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Generation service integration test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def test_model_manager_fallback():
    """Test that the system works without ModelManager"""
    print("\n🔄 Testing ModelManager Fallback...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        from core.model_integration_bridge import ModelIntegrationBridge
        bridge = ModelIntegrationBridge()
        
        # Simulate ModelManager not being available (which is the current state)
        bridge.model_manager = None
        
        # Test that the bridge still works
        try:
            # Test hardware profile detection
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Test hardware profile detection
                loop.run_until_complete(bridge._detect_hardware_profile())
                
                if bridge.hardware_profile:
                    print(f"✅ Hardware profile detected without ModelManager:")
                    print(f"   • GPU: {bridge.hardware_profile.gpu_name}")
                    print(f"   • Available VRAM: {bridge.hardware_profile.available_vram_gb:.1f}GB")
                else:
                    print("⚠️ No hardware profile detected (fallback mode)")
                
                # Test model availability check without ModelManager
                status = loop.run_until_complete(bridge.check_model_availability("t2v-a14b"))
                print(f"✅ Model availability check works without ModelManager: {status.status.value}")
                
                return True
                
            finally:
                loop.close()
                
        except Exception as e:
            print(f"❌ Fallback test failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ ModelManager fallback test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 GENERATION SERVICE FIX TEST")
    print("=" * 50)
    
    tests = [
        ("ModelManager Fallback", test_model_manager_fallback),
        ("Generation Service Integration", test_generation_service_integration)
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
        print("🎉 ALL TESTS PASSED! Generation service fix successful!")
        print("\n💡 Key improvements:")
        print("   • Hardware profile error resolved")
        print("   • ModelManager fallback working")
        print("   • Generation service can initialize")
        print("   • Model availability checks functional")
        print("\n🚀 The backend should now handle model operations gracefully.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())