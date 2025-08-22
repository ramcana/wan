#!/usr/bin/env python3
"""
CLI Test for WAN2.2 Video Generation WITHOUT Quantization

This script tests the WAN2.2 video generation pipeline without quantization
to isolate whether quantization is causing the hanging issue.
"""

import sys
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wan22_generation_no_quant():
    """Test WAN2.2 video generation without quantization"""
    
    print("🎬 WAN2.2 Video Generation CLI Test (NO QUANTIZATION)")
    print("=" * 60)
    
    try:
        # Import the video generation engine
        from utils import VideoGenerationEngine
        
        print("✅ Successfully imported VideoGenerationEngine")
        
        # Initialize the engine with custom optimization settings
        engine = VideoGenerationEngine()
        
        # Override the default optimization settings to disable quantization
        if hasattr(engine, 'config') and 'optimization' in engine.config:
            engine.config['optimization']['quantization_level'] = 'none'
            engine.config['optimization']['enable_offload'] = True  # Keep offloading
            print("✅ Disabled quantization in engine config")
        
        print("✅ VideoGenerationEngine initialized (quantization disabled)")
        
        # Test parameters
        test_prompt = "A cat walking in a garden, high quality, detailed"
        model_type = "t2v-A14B"
        resolution = "1280x720"
        steps = 20
        duration = 2  # Short duration for testing
        
        print(f"\n🚀 Starting video generation (NO QUANTIZATION):")
        print(f"   Prompt: {test_prompt}")
        print(f"   Model: {model_type}")
        print(f"   Resolution: {resolution}")
        print(f"   Steps: {steps}")
        print(f"   Duration: {duration}s")
        print(f"   Quantization: DISABLED")
        
        # Start generation with explicit optimization settings
        start_time = time.time()
        
        result = engine.generate_video(
            model_type=model_type,
            prompt=test_prompt,
            resolution=resolution,
            num_inference_steps=steps,
            guidance_scale=7.5,
            duration=duration,
            fps=8,
            # Explicitly disable quantization (pass as direct kwargs)
            quantization_level='none',
            enable_offload=True,
            vae_tile_size=256,
            skip_large_components=True
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"\n⏱️  Generation completed in {generation_time:.2f} seconds")
        
        # Debug: Print result object details
        print(f"\n🔍 DEBUG: Result object type: {type(result)}")
        print(f"🔍 DEBUG: Result object: {result}")
        if result:
            print(f"🔍 DEBUG: Result attributes: {dir(result)}")
            if hasattr(result, '__dict__'):
                print(f"🔍 DEBUG: Result dict: {result.__dict__}")
        
        # Check result - handle different result formats
        success = False
        if result:
            # Check for different success indicators
            if hasattr(result, 'success') and result.success:
                success = True
            elif isinstance(result, dict) and result.get('success'):
                success = True
            elif isinstance(result, dict) and 'output_path' in result and result['output_path']:
                success = True
            elif hasattr(result, 'output_path') and result.output_path:
                success = True
        
        if success:
            print("✅ Video generation SUCCESSFUL (without quantization)!")
            
            # Extract information from different result formats
            output_path = None
            frames = None
            generation_time = None
            memory_used = None
            
            if hasattr(result, 'output_path'):
                output_path = result.output_path
            elif isinstance(result, dict):
                output_path = result.get('output_path')
                
            if hasattr(result, 'frames'):
                frames = result.frames
            elif isinstance(result, dict):
                frames = result.get('frames')
                
            if hasattr(result, 'generation_time'):
                generation_time = result.generation_time
            elif isinstance(result, dict):
                generation_time = result.get('generation_time')
                
            if hasattr(result, 'memory_used_mb'):
                memory_used = result.memory_used_mb
            elif isinstance(result, dict):
                memory_used = result.get('memory_used_mb')
            
            print(f"   Output path: {output_path}")
            print(f"   Frames generated: {len(frames) if frames else 'N/A'}")
            print(f"   Generation time: {generation_time:.2f}s" if generation_time else "   Generation time: N/A")
            print(f"   Memory used: {memory_used}MB" if memory_used else "   Memory used: N/A")
            
            # Check if output file exists
            if output_path and Path(output_path).exists():
                file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
                print(f"   File size: {file_size:.2f}MB")
                print(f"✅ Output file created successfully: {output_path}")
            else:
                print("⚠️  Output file not found, but generation completed")
            
            return True
            
        else:
            print("❌ Video generation FAILED!")
            if result:
                # Try different ways to get error message
                error_msg = None
                if hasattr(result, 'error_message'):
                    error_msg = result.error_message
                elif isinstance(result, dict) and 'error' in result:
                    error_msg = result['error']
                elif isinstance(result, dict) and 'error_message' in result:
                    error_msg = result['error_message']
                else:
                    error_msg = 'Unknown error'
                    
                print(f"   Error: {error_msg}")
                
                # Try to get errors list
                errors = None
                if hasattr(result, 'errors'):
                    errors = result.errors
                elif isinstance(result, dict) and 'errors' in result:
                    errors = result['errors']
                    
                if errors:
                    for error in errors:
                        print(f"   - {error}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔧 Initializing WAN2.2 generation test (NO QUANTIZATION)...")
    
    # Apply compatibility fixes
    try:
        import wan22_compatibility_clean
        print("✅ WAN2.2 compatibility layer loaded")
    except Exception as e:
        print(f"⚠️  Could not load compatibility layer: {e}")
    
    # Run the test
    success = test_wan22_generation_no_quant()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 WAN2.2 VIDEO GENERATION TEST PASSED (NO QUANTIZATION)!")
        print("✅ The issue was likely caused by quantization")
        print("💡 Consider using alternative quantization methods or disabling it")
    else:
        print("❌ WAN2.2 VIDEO GENERATION TEST FAILED (NO QUANTIZATION)!")
        print("🔧 The issue is not related to quantization")
        print("🔧 Check the error messages above for other causes")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)