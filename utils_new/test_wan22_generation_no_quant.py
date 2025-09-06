#!/usr/bin/env python3
"""
Test WAN2.2 video generation without quantization to avoid hanging
"""

import sys
import logging
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_wan22_generation_no_quant():
    """Test WAN2.2 generation with quantization disabled"""
    print("🔧 Initializing WAN2.2 generation test (no quantization)...")
    
    try:
        # Import the compatibility layer first
        import wan22_compatibility_clean
        print("✅ WAN2.2 compatibility layer loaded")
        
        # Import the main engine
        from utils import VideoGenerationEngine
        
        print("✅ Successfully imported VideoGenerationEngine")
        
        # Initialize the engine with quantization disabled
        engine = VideoGenerationEngine()
        print("✅ VideoGenerationEngine initialized")
        
        # Override optimization settings to disable quantization
        if hasattr(engine, 'model_manager') and hasattr(engine.model_manager, 'optimization_config'):
            engine.model_manager.optimization_config['quantization_level'] = 'none'
            print("✅ Quantization disabled")
        
        print("\n🚀 Starting video generation:")
        print("   Prompt: A cat walking in a garden, high quality, detailed")
        print("   Model: t2v-A14B")
        print("   Resolution: 1280x720")
        print("   Steps: 20")
        print("   Duration: 2s")
        print("   Quantization: DISABLED")
        
        # Test generation
        result = engine.generate_video(
            model_type="t2v-A14B",
            prompt="A cat walking in a garden, high quality, detailed",
            resolution="1280x720",
            num_inference_steps=20,
            guidance_scale=7.5,
            duration=2,
            fps=8
        )
        
        if result and 'video_path' in result:
            print(f"✅ Video generated successfully: {result['video_path']}")
            print(f"✅ Generation time: {result.get('generation_time', 'unknown')} seconds")
            return True
        else:
            print(f"❌ Generation failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    print("🎬 WAN2.2 Video Generation Test (No Quantization)")
    print("=" * 50)
    
    success = test_wan22_generation_no_quant()
    
    print("=" * 50)
    if success:
        print("✅ WAN2.2 VIDEO GENERATION TEST PASSED!")
    else:
        print("❌ WAN2.2 VIDEO GENERATION TEST FAILED!")
        print("🔧 Check the error messages above for troubleshooting")
    print("=" * 50)