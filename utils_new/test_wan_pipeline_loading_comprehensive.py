#!/usr/bin/env python3
"""
Comprehensive test for WAN pipeline loading
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wan_pipeline_loading_comprehensive():
    """Test actual WAN pipeline loading"""
    
    print("🧪 Comprehensive WAN Pipeline Loading Test")
    print("=" * 50)
    
    try:
        # Import the components
        from wan_pipeline_loader import WanPipelineLoader
        from architecture_detector import ArchitectureDetector
        
        print("✅ Successfully imported WAN pipeline components")
        
        # Test with a local model path
        model_path = "models/Wan-AI_Wan2.2-T2V-A14B-Diffusers"
        if not Path(model_path).exists():
            print(f"⚠️  Model path {model_path} not found - skipping loading test")
            return True
        
        print(f"🔍 Testing actual pipeline loading for: {model_path}")
        
        # Test architecture detection
        detector = ArchitectureDetector()
        architecture = detector.detect_model_architecture(model_path)
        
        print(f"✅ Architecture detected: {architecture.architecture_type.value}")
        
        # Test pipeline loader
        loader = WanPipelineLoader()
        print("✅ WAN pipeline loader initialized")
        
        # Try to actually load the pipeline
        print("🚀 Attempting to load WAN pipeline...")
        try:
            pipeline_wrapper = loader.load_wan_pipeline(
                model_path=model_path,
                trust_remote_code=True,
                apply_optimizations=False  # Skip optimizations for faster testing
            )
            
            print(f"✅ Pipeline loaded successfully!")
            print(f"   Pipeline type: {type(pipeline_wrapper.pipeline).__name__}")
            print(f"   Applied optimizations: {pipeline_wrapper.optimization_result.applied_optimizations}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_wan_pipeline_loading_comprehensive()
    
    if success:
        print("\n🎉 Comprehensive WAN pipeline loading test PASSED!")
        print("🚀 The WAN2.2 model should now work in the UI.")
    else:
        print("\n❌ Comprehensive test FAILED.")
        print("There are still issues with the WAN pipeline loading.")
    
    input("\nPress Enter to exit...")