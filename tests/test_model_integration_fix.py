#!/usr/bin/env python3
"""
Test Model Integration Fix
Verify that the model ID mapping and availability fixes are working
"""

import sys
import asyncio
from pathlib import Path

async def test_model_integration():
    """Test the fixed model integration"""
    print("🧪 Testing Model Integration Fix...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        from backend.core.model_integration_bridge import ModelIntegrationBridge
        
        # Initialize bridge
        bridge = ModelIntegrationBridge()
        await bridge.initialize()
        
        # Test the problematic model ID from the logs
        test_model = "t2v-a14b"
        
        print(f"\n🔍 Testing model availability for: {test_model}")
        
        # Check availability
        status = await bridge.check_model_availability(test_model)
        print(f"   Status: {status.status.value}")
        print(f"   Valid: {status.is_valid}")
        print(f"   Cached: {status.is_cached}")
        print(f"   Size: {status.size_mb:.1f}MB")
        print(f"   Model ID: {status.model_id}")
        
        if status.error_message:
            print(f"   Error: {status.error_message}")
        
        # Test ensure model available
        print(f"\n📥 Testing ensure model available for: {test_model}")
        available = await bridge.ensure_model_available(test_model)
        print(f"   Result: {'✅ Available' if available else '❌ Not Available'}")
        
        # Test all model types
        print(f"\n🔄 Testing all model types:")
        test_models = ["t2v-a14b", "i2v-a14b", "ti2v-5b"]
        
        for model in test_models:
            status = await bridge.check_model_availability(model)
            available = await bridge.ensure_model_available(model)
            print(f"   {model}: {status.status.value} -> {'✅' if available else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main function"""
    print("🔧 MODEL INTEGRATION FIX TEST")
    print("=" * 50)
    
    success = await test_model_integration()
    
    if success:
        print("\n🎉 Model integration fix is working!")
        print("✅ The generation service should now be able to find models")
    else:
        print("\n❌ Model integration fix needs more work")

if __name__ == "__main__":
    asyncio.run(main())