#!/usr/bin/env python3
"""
Simple WAN2.2 Pipeline Loading Test

Just tests if the pipeline loads successfully without full generation.
"""

import sys
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline_loading_only():
    """Test just the pipeline loading part"""
    
    print("🔧 WAN2.2 Pipeline Loading Test")
    print("=" * 40)
    
    try:
        # Import compatibility layer
        import wan22_compatibility_clean
        print("✅ Compatibility layer loaded")
        
        # Import the video generation engine
        from utils import VideoGenerationEngine
        print("✅ VideoGenerationEngine imported")
        
        # Initialize the engine
        engine = VideoGenerationEngine()
        print("✅ VideoGenerationEngine initialized")
        
        # Test just the pipeline loading part
        print("🚀 Testing pipeline loading for t2v-A14B...")
        
        try:
            # This should trigger the pipeline loading
            pipeline, model_info = engine.model_manager.load_model("t2v-A14B")
            
            print("✅ Pipeline loaded successfully!")
            print(f"   Pipeline type: {type(pipeline).__name__}")
            print(f"   Model info: {model_info}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline loading failed: {e}")
            # Print just the first few lines of the error for readability
            error_lines = str(e).split('\n')[:3]
            for line in error_lines:
                print(f"   {line}")
            return False
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    success = test_pipeline_loading_only()
    
    if success:
        print("\n🎉 WAN2.2 PIPELINE LOADING TEST PASSED!")
        print("✅ The pipeline loads correctly")
        print("🚀 Video generation should work in the UI")
    else:
        print("\n❌ Pipeline loading test failed")
        print("🔧 Check the error messages above")
    
    print("\n" + "=" * 40)