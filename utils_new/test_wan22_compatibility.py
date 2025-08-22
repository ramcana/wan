#!/usr/bin/env python3
"""
Test the Wan2.2 compatibility layer
"""

import sys
sys.path.insert(0, ".")

def test_wan22_compatibility():
    """Test the Wan2.2 compatibility layer"""
    
    print("🧪 Testing Wan2.2 Compatibility Layer")
    print("=" * 50)
    
    # Test 1: Import compatibility layer
    print("1. Testing compatibility layer import...")
    try:
        import wan22_compatibility_layer
        print("   ✅ Compatibility layer imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import compatibility layer: {e}")
        return False
    
    # Test 2: Check if WanPipeline is available
    print("2. Testing WanPipeline availability...")
    try:
        from diffusers import WanPipeline, AutoencoderKLWan
        print("   ✅ WanPipeline and AutoencoderKLWan are now available")
    except ImportError as e:
        print(f"   ❌ WanPipeline still not available: {e}")
        return False
    
    # Test 3: Test model loading with compatibility layer
    print("3. Testing model loading with compatibility...")
    try:
        import utils
        
        # Create model manager
        manager = utils.ModelManager()
        
        # Test model loading (this should now work)
        try:
            pipeline, model_info = manager.load_model("t2v-A14B")
            print(f"   ✅ Model loaded successfully!")
            print(f"       Pipeline type: {type(pipeline).__name__}")
            print(f"       Model type: {model_info.model_type}")
            return True
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Utils import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_wan22_compatibility()
    
    if success:
        print("\n🎉 Wan2.2 compatibility layer is working!")
        print("You should now be able to run video generation without WanPipeline errors.")
    else:
        print("\n❌ Compatibility layer needs more work.")
        print("Check the error messages above for details.")