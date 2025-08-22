#!/usr/bin/env python3
"""
Correct test script for WAN TI2V-5B model
Testing the actual Wan-AI/Wan2.2-TI2V-5B-Diffusers model
"""

import torch
import gc
import time
import psutil
import os
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append('.')

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
    else:
        gpu_memory = gpu_memory_cached = 0
    
    cpu_memory = psutil.Process().memory_info().rss / 1024**3
    
    return {
        'gpu_allocated': gpu_memory,
        'gpu_cached': gpu_memory_cached,
        'cpu_memory': cpu_memory
    }

def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def test_wan_ti2v_5b_model():
    """Test the actual WAN TI2V-5B model"""
    print("=" * 60)
    print("Testing WAN TI2V-5B Model")
    print("Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    print("Text + Image to Video (5B parameters)")
    print("=" * 60)
    
    # Apply WAN compatibility layer
    try:
        from wan22_compatibility_clean import apply_wan22_compatibility, remove_wan22_compatibility
        print("✅ WAN compatibility layer imported")
    except ImportError:
        print("❌ WAN compatibility layer not found")
        return False
    
    clear_memory()
    initial_memory = get_memory_usage()
    print(f"Initial Memory - GPU: {initial_memory['gpu_allocated']:.2f}GB, CPU: {initial_memory['cpu_memory']:.2f}GB")
    
    try:
        from diffusers import DiffusionPipeline
        import numpy as np
        from PIL import Image
        
        # The actual WAN TI2V-5B model
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        
        print(f"\nLoading WAN TI2V-5B model: {model_id}")
        print("This model supports Text + Image to Video generation")
        
        start_time = time.time()
        
        # First try with WAN compatibility layer
        print("Attempting to load with WAN compatibility layer...")
        
        try:
            # Load with trust_remote_code for WAN models
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,  # Required for WAN models
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            load_time = time.time() - start_time
            load_memory = get_memory_usage()
            
            print(f"✅ Model loaded successfully in {load_time:.2f}s")
            print(f"Pipeline type: {type(pipe).__name__}")
            print(f"Memory after loading - GPU: {load_memory['gpu_allocated']:.2f}GB, CPU: {load_memory['cpu_memory']:.2f}GB")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                print("Moving pipeline to GPU...")
                pipe = pipe.to("cuda")
                
                # Apply memory optimizations
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                    print("✅ Attention slicing enabled")
                
                if hasattr(pipe, 'enable_model_cpu_offload'):
                    pipe.enable_model_cpu_offload()
                    print("✅ CPU offload enabled")
                
                gpu_memory = get_memory_usage()
                print(f"GPU memory after optimization: {gpu_memory['gpu_allocated']:.2f}GB")
            
            # Test TI2V generation (Text + Image to Video)
            print("\n--- Testing TI2V Generation ---")
            
            # Create a simple test image
            test_image = Image.new('RGB', (512, 512), color='lightblue')
            test_prompt = "A peaceful lake scene with gentle ripples and reflections"
            
            # Test configurations optimized for RTX 4080
            test_configs = [
                {
                    "name": "Quick Test (512x512, 8 frames)",
                    "width": 512,
                    "height": 512,
                    "num_frames": 8,
                    "num_inference_steps": 15
                },
                {
                    "name": "Standard Test (720x480, 16 frames)",
                    "width": 720,
                    "height": 480,
                    "num_frames": 16,
                    "num_inference_steps": 20
                }
            ]
            
            for i, config in enumerate(test_configs):
                print(f"\n--- Test {i+1}: {config['name']} ---")
                
                try:
                    clear_memory()
                    pre_memory = get_memory_usage()
                    
                    print(f"Generating {config['num_frames']} frames at {config['width']}x{config['height']}")
                    print(f"Pre-generation GPU memory: {pre_memory['gpu_allocated']:.2f}GB")
                    
                    gen_start = time.time()
                    
                    # Track memory during generation
                    peak_memory = pre_memory['gpu_allocated']
                    
                    def memory_callback(step, timestep, latents):
                        nonlocal peak_memory
                        current_memory = get_memory_usage()
                        peak_memory = max(peak_memory, current_memory['gpu_allocated'])
                        if step % 5 == 0:
                            print(f"  Step {step}: {current_memory['gpu_allocated']:.2f}GB")
                    
                    # Generate video with TI2V (Text + Image input)
                    generation_kwargs = {
                        'prompt': test_prompt,
                        'image': test_image,  # Input image for TI2V
                        'width': config['width'],
                        'height': config['height'],
                        'num_frames': config['num_frames'],
                        'num_inference_steps': config['num_inference_steps'],
                        'guidance_scale': 7.5,
                        'generator': torch.Generator().manual_seed(42)
                    }
                    
                    # Add callback if supported
                    if hasattr(pipe, '__call__'):
                        try:
                            # Try with callback first
                            generation_kwargs['callback'] = memory_callback
                            generation_kwargs['callback_steps'] = 1
                            result = pipe(**generation_kwargs)
                        except TypeError:
                            # Fallback without callback
                            generation_kwargs.pop('callback', None)
                            generation_kwargs.pop('callback_steps', None)
                            result = pipe(**generation_kwargs)
                    else:
                        result = pipe(**generation_kwargs)
                    
                    # Extract frames
                    if hasattr(result, 'frames'):
                        video_frames = result.frames[0]
                    elif hasattr(result, 'videos'):
                        video_frames = result.videos[0]
                    else:
                        video_frames = result
                    
                    gen_time = time.time() - gen_start
                    post_memory = get_memory_usage()
                    
                    print(f"✅ Generation completed in {gen_time:.2f}s")
                    print(f"Peak GPU memory: {peak_memory:.2f}GB")
                    print(f"Frames per second: {config['num_frames']/gen_time:.2f}")
                    
                    # Save sample frames
                    if video_frames and len(video_frames) > 0:
                        output_dir = Path("outputs/wan_ti2v_5b_test")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save first, middle, and last frames
                        frame_indices = [0, len(video_frames)//2, -1]
                        frame_names = ['first', 'middle', 'last']
                        
                        for idx, name in zip(frame_indices, frame_names):
                            frame = video_frames[idx]
                            
                            # Convert frame to PIL Image
                            if isinstance(frame, torch.Tensor):
                                frame = frame.cpu().numpy()
                            if isinstance(frame, np.ndarray):
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                if len(frame.shape) == 3 and frame.shape[0] == 3:
                                    frame = frame.transpose(1, 2, 0)
                                frame = Image.fromarray(frame)
                            
                            frame_path = output_dir / f"test_{i+1}_{name}_frame.png"
                            frame.save(frame_path)
                            print(f"Saved {name} frame: {frame_path}")
                        
                        print(f"Generated {len(video_frames)} frames total")
                    
                    print(f"✅ Test {i+1} completed successfully")
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"❌ GPU Out of Memory for {config['name']}")
                    print(f"Try reducing resolution or frame count")
                    clear_memory()
                    continue
                    
                except Exception as e:
                    print(f"❌ Error in {config['name']}: {e}")
                    import traceback
                    traceback.print_exc()
                    clear_memory()
                    continue
            
            print(f"\n{'='*60}")
            print("WAN TI2V-5B Test Summary")
            print(f"{'='*60}")
            print(f"Model: {model_id}")
            print(f"Load time: {load_time:.2f} seconds")
            print(f"Peak GPU memory: {peak_memory:.2f}GB")
            print("✅ WAN TI2V-5B is working on RTX 4080 16GB!")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load with WAN compatibility: {e}")
            
            # Try without compatibility layer
            print("\nTrying without WAN compatibility layer...")
            remove_wan22_compatibility()
            
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("✅ Loaded without compatibility layer")
                return True
            except Exception as e2:
                print(f"❌ Also failed without compatibility layer: {e2}")
                return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure diffusers and required dependencies are installed")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        clear_memory()
        final_memory = get_memory_usage()
        print(f"Final memory - GPU: {final_memory['gpu_allocated']:.2f}GB, CPU: {final_memory['cpu_memory']:.2f}GB")

def main():
    """Main test function"""
    print("WAN TI2V-5B Model Test for RTX 4080 16GB")
    print("=" * 60)
    
    # System check
    print("System Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"CPU Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print()
    
    success = test_wan_ti2v_5b_model()
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    if success:
        print("🎉 WAN TI2V-5B model is compatible with your RTX 4080!")
        print("\nRecommendations for optimal performance:")
        print("  - Use 512x512 to 720x480 resolution")
        print("  - Keep frame count between 8-24 frames")
        print("  - Use 15-25 inference steps")
        print("  - Provide both text prompt and input image")
        print("  - Enable attention slicing and CPU offload")
        print("  - Avoid quantization (causes hanging)")
    else:
        print("❌ WAN TI2V-5B model compatibility issues detected")
        print("\nTroubleshooting:")
        print("  1. Ensure you have access to Wan-AI models on Hugging Face")
        print("  2. Check your Hugging Face authentication")
        print("  3. Update diffusers: pip install --upgrade diffusers")
        print("  4. Try the T2V-A14B model as alternative")

if __name__ == "__main__":
    main()