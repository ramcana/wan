#!/usr/bin/env python3
"""
Generation Pipeline Fix Test
Tests that the generate_video_with_optimization method is now available
"""

import sys
import asyncio
import traceback
from pathlib import Path

async def test_generation_pipeline_method():
    """Test that the generate_video_with_optimization method exists"""
    print("🎬 Testing Generation Pipeline Method...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        # Import the real generation pipeline
        from services.real_generation_pipeline import RealGenerationPipeline
        
        # Create pipeline instance
        pipeline = RealGenerationPipeline()
        
        # Check if the method exists
        if hasattr(pipeline, 'generate_video_with_optimization'):
            print("✅ generate_video_with_optimization method exists")
            
            # Check if it's callable
            if callable(getattr(pipeline, 'generate_video_with_optimization')):
                print("✅ generate_video_with_optimization method is callable")
                
                # Test method signature (without actually calling it)
                import inspect
                sig = inspect.signature(pipeline.generate_video_with_optimization)
                params = list(sig.parameters.keys())
                print(f"✅ Method signature: {params}")
                
                return True
            else:
                print("❌ generate_video_with_optimization is not callable")
                return False
        else:
            print("❌ generate_video_with_optimization method does not exist")
            return False
        
    except Exception as e:
        print(f"❌ Generation pipeline method test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def test_generation_service_integration():
    """Test that the generation service can now call the method"""
    print("\n🔗 Testing Generation Service Integration...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        # Import generation service
        from services.generation_service import GenerationService
        gen_service = GenerationService()
        
        # Check if generation service has real_generation_pipeline
        if hasattr(gen_service, 'real_generation_pipeline'):
            pipeline = gen_service.real_generation_pipeline
            
            if hasattr(pipeline, 'generate_video_with_optimization'):
                print("✅ Generation service can access generate_video_with_optimization")
                return True
            else:
                print("❌ Generation service pipeline missing generate_video_with_optimization")
                return False
        else:
            print("⚠️ Generation service has no real_generation_pipeline attribute")
            return False
        
    except Exception as e:
        print(f"❌ Generation service integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 GENERATION PIPELINE FIX TEST")
    print("=" * 50)
    
    tests = [
        ("Generation Pipeline Method", test_generation_pipeline_method),
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
        print("🎉 ALL TESTS PASSED! Generation pipeline fix successful!")
        print("\n💡 The 'generate_video_with_optimization' method is now available.")
        print("   The backend should now handle video generation requests properly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())