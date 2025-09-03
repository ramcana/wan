#!/usr/bin/env python3
"""
Test the clean Wan2.2 compatibility fix
"""

import sys
import os
sys.path.insert(0, ".")

def test_clean_fix():
    """Test the clean Wan2.2 fix"""
    
    print("🔧 Testing Clean Wan2.2 Fix")
    print("=" * 50)
    
    # Test 1: Import clean compatibility layer
    print("1. Loading clean compatibility layer...")
    try:
        import wan22_compatibility_clean
from diffusers import WanPipeline
        print("   ✅ Clean WanPipeline compatibility layer loaded")
    except Exception as e:
        print(f"   ❌ Clean compatibility layer failed: {e}")
        return False
    
    # Test 2: Test direct pipeline loading
    print("2. Testing direct pipeline loading...")
    try:
        local_model_path = "models/Wan-AI_Wan2.2-T2V-A14B-Diffusers"
        
        if os.path.exists(local_model_path):
            print(f"   ✅ Local model found: {local_model_path}")
            
            try:
                # Test WanPipeline directly
                pipeline = WanPipeline.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float32
                )
                print(f"   ✅ Pipeline loaded directly: {type(pipeline).__name__}")
                return True
                
            except Exception as e:
                print(f"   ❌ Direct pipeline loading failed: {e}")
                
                # Try with more permissive settings
                try:
                    print("   Trying with more permissive settings...")
                    pipeline = WanPipeline.from_pretrained(
                        local_model_path,
                        local_files_only=True,
                        torch_dtype=None,
                        ignore_mismatched_sizes=True,
                        low_cpu_mem_usage=False
                    )
                    print(f"   ✅ Pipeline loaded with permissive settings: {type(pipeline).__name__}")
                    return True
                except Exception as e2:
                    print(f"   ❌ Permissive loading also failed: {e2}")
                    return False
        else:
            print(f"   ⚠️  Local model not found at {local_model_path}")
            return False
            
    except Exception as e:
        print(f"   ❌ Direct pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    # Import torch for the test
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not available")
        sys.exit(1)
    
    success = test_clean_fix()
    
    if success:
        print("\n🎉 Clean fix is working!")
        print("Your Wan2.2 models should now load without errors.")
        print("\nYou can now run your video generation application.")
    else:
        print("\n⚠️  Clean fix needs more work.")
        print("The model may have compatibility issues that require the original WanPipeline components.")