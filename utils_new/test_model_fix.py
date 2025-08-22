#!/usr/bin/env python3
"""
Test script to verify the model loading fix
"""

import sys
import os
sys.path.insert(0, ".")

def test_model_loading_fix():
    """Test that model loading uses local models instead of trying to download"""
    
    print("🔧 Testing Model Loading Fix")
    print("=" * 50)
    
    try:
        # Import the model override system
        from model_override import get_local_model_path, patch_model_loading
        
        # Test local model path detection
        print("\n1. Testing local model path detection:")
        
        test_models = [
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "WAN2.2-T2V-A14B",
            "WAN2.2-I2V-A14B",
            "WAN2.2-TI2V-5B"
        ]
        
        for model_id in test_models:
            try:
                local_path = get_local_model_path(model_id)
                print(f"   ✅ {model_id} -> {local_path}")
            except FileNotFoundError as e:
                print(f"   ❌ {model_id} -> {e}")
        
        # Test model manager integration
        print("\n2. Testing model manager integration:")
        
        # Apply the patch
        success = patch_model_loading()
        if success:
            print("   ✅ Model loading patch applied successfully")
        else:
            print("   ❌ Failed to apply model loading patch")
            return False
        
        # Test with utils module
        try:
            import utils
            
            # Create a model manager instance
            manager = utils.ModelManager()
            
            # Test model ID mapping
            print("\n3. Testing model ID mappings:")
            test_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            
            for model_type in test_types:
                try:
                    model_id = manager.get_model_id(model_type)
                    print(f"   ✅ {model_type} -> {model_id}")
                except Exception as e:
                    print(f"   ❌ {model_type} -> Error: {e}")
            
            print("\n4. Testing model download (should use local):")
            try:
                # This should now use local models instead of downloading
                local_path = manager.download_model("t2v-A14B")
                print(f"   ✅ t2v-A14B model path: {local_path}")
                return True
            except Exception as e:
                print(f"   ❌ Failed to get t2v-A14B model: {e}")
                return False
                
        except ImportError as e:
            print(f"   ❌ Failed to import utils: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading_fix()
    
    if success:
        print("\n🎉 Model loading fix is working!")
        print("\nThe system should now use local models instead of trying to download from Hugging Face.")
        print("You can now run your video generation without the 404 errors.")
    else:
        print("\n❌ Model loading fix needs attention.")
        print("Please check the error messages above.")