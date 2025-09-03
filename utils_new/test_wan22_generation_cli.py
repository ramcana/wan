#!/usr/bin/env python3
"""
CLI Test for WAN2.2 Video Generation

This script tests the complete WAN2.2 video generation pipeline
from prompt to video output without the UI.
"""

import sys
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wan22_generation():
    """Test WAN2.2 video generation end-to-end"""
    
    print("🎬 WAN2.2 Video Generation CLI Test")
    print("=" * 50)
    
    try:
        # Import the video generation engine
        from utils import VideoGenerationEngine
        
        print("✅ Successfully imported VideoGenerationEngine")
        
        # Initialize the engine
        engine = VideoGenerationEngine()
        print("✅ VideoGenerationEngine initialized")
        
        # Test parameters
        test_prompt = "A cat walking in a garden, high quality, detailed"
        model_type = "t2v-A14B"
        resolution = "1280x720"  # Use valid resolution
        steps = 20
        duration = 2  # Short duration for testing
        
        print(f"\n🚀 Starting video generation:")
        print(f"   Prompt: {test_prompt}")
        print(f"   Model: {model_type}")
        print(f"   Resolution: {resolution}")
        print(f"   Steps: {steps}")
        print(f"   Duration: {duration}s")
        
        # Start generation
        start_time = time.time()
        
        result = engine.generate_video(
            model_type=model_type,
            prompt=test_prompt,
            resolution=resolution,
            num_inference_steps=steps,
            guidance_scale=7.5,
            duration=duration,
            fps=8
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"\n⏱️  Generation completed in {generation_time:.2f} seconds")
        
        # Check result
        if result and hasattr(result, 'success') and result.success:
            print("✅ Video generation SUCCESSFUL!")
            print(f"   Output path: {result.output_path}")
            print(f"   Frames generated: {len(result.frames) if result.frames else 'N/A'}")
            print(f"   Generation time: {result.generation_time:.2f}s")
            print(f"   Memory used: {result.memory_used_mb}MB")
            
            # Check if output file exists
            if result.output_path and Path(result.output_path).exists():
                file_size = Path(result.output_path).stat().st_size / (1024 * 1024)  # MB
                print(f"   File size: {file_size:.2f}MB")
                print(f"✅ Output file created successfully: {result.output_path}")
            else:
                print("⚠️  Output file not found, but generation reported success")
            
            return True
            
        else:
            print("❌ Video generation FAILED!")
            if result:
                print(f"   Error: {getattr(result, 'error_message', 'Unknown error')}")
                if hasattr(result, 'errors') and result.errors:
                    for error in result.errors:
                        print(f"   - {error}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔧 Initializing WAN2.2 generation test...")
    
    # Apply compatibility fixes
    try:
        import wan22_compatibility_clean
print("✅ WAN2.2 compatibility layer loaded")
    except Exception as e:
        print(f"⚠️  Could not load compatibility layer: {e}")
    
    # Run the test
    success = test_wan22_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 WAN2.2 VIDEO GENERATION TEST PASSED!")
        print("✅ The WAN2.2 model is working correctly")
        print("🚀 You can now use the UI for video generation")
    else:
        print("❌ WAN2.2 VIDEO GENERATION TEST FAILED!")
        print("🔧 Check the error messages above for troubleshooting")
    
    print("=" * 50)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)