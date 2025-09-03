#!/usr/bin/env python3
"""
Final test for Wan2.2 fix - simplified version
"""

import sys
import os
sys.path.insert(0, ".")

def test_final_fix():
    """Test the final Wan2.2 fix"""
    
    print("🔧 Testing Final Wan2.2 Fix")
    print("=" * 50)
    
    # Test 1: Import compatibility layer
    print("1. Loading compatibility layer...")
    try:
        import wan22_compatibility_layer
from diffusers import WanPipeline
        print("   ✅ WanPipeline compatibility layer loaded")
    except Exception as e:
        print(f"   ❌ Compatibility layer failed: {e}")
        return False
    
    # Test 2: Test model manager with local files
    print("2. Testing model manager with local files...")
    try:
        import utils

        manager = utils.ModelManager()
        
        # Force use of local model
        local_model_path = "models/Wan-AI_Wan2.2-T2V-A14B-Diffusers"
        
        if os.path.exists(local_model_path):
            print(f"   ✅ Local model found: {local_model_path}")
            
            try:
                # Test pipeline loading directly
                pipeline = manager._load_pipeline_with_fallback(
                    local_model_path,
                    torch_dtype=None,
                    local_files_only=True
                )
                print(f"   ✅ Pipeline loaded: {type(pipeline).__name__}")
                return True
                
            except Exception as e:
                print(f"   ❌ Pipeline loading failed: {e}")
                return False
        else:
            print(f"   ⚠️  Local model not found at {local_model_path}")
            print("   Checking available models...")
            
            # List available models
            models_dir = "models"
            if os.path.exists(models_dir):
                available = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
                print(f"   Available models: {available}")
            
            return False
            
    except Exception as e:
        print(f"   ❌ Model manager test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_final_fix()
    
    if success:
        print("\n🎉 Final fix is working!")
        print("Your Wan2.2 models should now load without errors.")
        print("\nYou can now run your video generation application.")
    else:
        print("\n⚠️  Fix needs more work or models need to be properly placed.")
        print("Make sure your models are in the correct directory structure.")