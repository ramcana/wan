#!/usr/bin/env python3
"""
Fixed test script for TI2V-5B model with proper dependency handling
"""

import torch
import gc
import time
import psutil
import os
from pathlib import Path

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

def check_dependencies():
    """Check if required dependencies are properly installed"""
    print("Checking dependencies...")
    
    try:
        import transformers
print(f"✅ transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers not found: {e}")
        return False
    
    try:
        import diffusers
print(f"✅ diffusers: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ diffusers not found: {e}")
        return False
    
    try:
        from transformers import T5Tokenizer, T5EncoderModel
        print("✅ T5 components available")
    except ImportError as e:
        print(f"❌ T5 components not available: {e}")
        return False
    
    return True

def test_model_loading_step_by_step():
    """Test model loading step by step to identify issues"""
    print("\n" + "="*60)
    print("Step-by-step Model Loading Test")
    print("="*60)
    
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages:")
        print("pip install transformers diffusers torch torchvision accelerate")
        return False
    
    try:
        from diffusers import CogVideoXPipeline
        from transformers import T5Tokenizer, T5EncoderModel
        import numpy as np
from PIL import Image
        
        # Test with a smaller, more compatible model first
        model_configs = [
            {
                'name': 'CogVideoX-2B (Smaller test)',
                'model_id': 'THUDM/CogVideoX-2b',
                'description': 'Smaller model for testing compatibility'
            },
            {
                'name': 'CogVideoX-5B (Target model)',
                'model_id': 'THUDM/CogVideoX-5b',
                'description': 'Target 5B model for production'
            }
        ]
        
        for config in model_configs:
            print(f"\n--- Testing {config['name']} ---")
            print(f"Model ID: {config['model_id']}")
            
            clear_memory()
            initial_memory = get_memory_usage()
            
            try:
                print("Loading pipeline...")
                start_time = time.time()
                
                # Try loading with minimal configuration first
                pipe = CogVideoXPipeline.from_pretrained(
                    config['model_id'],
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                load_time = time.time() - start_time
                load_memory = get_memory_usage()
                
                print(f"✅ Model loaded successfully in {load_time:.2f}s")
                print(f"Memory usage: {load_memory['gpu_allocated']:.2f}GB GPU")
                
                # Move to GPU
                if torch.cuda.is_available():
                    print("Moving to GPU...")
                    pipe = pipe.to("cuda")
                    
                    # Apply optimizations
                    pipe.enable_attention_slicing()
                    pipe.enable_model_cpu_offload()
                    
                    gpu_memory = get_memory_usage()
                    print(f"GPU memory after optimization: {gpu_memory['gpu_allocated']:.2f}GB")
                
                # Quick generation test
                print("Testing quick generation...")
                test_prompt = "A calm ocean with gentle waves"
                
                try:
                    gen_start = time.time()
                    
                    video_frames = pipe(
                        prompt=test_prompt,
                        width=512,
                        height=512,
                        num_frames=8,
                        num_inference_steps=10,  # Very low for quick test
                        guidance_scale=7.5,
                        generator=torch.Generator().manual_seed(42)
                    ).frames[0]
                    
                    gen_time = time.time() - gen_start
                    final_memory = get_memory_usage()
                    
                    print(f"✅ Generation completed in {gen_time:.2f}s")
                    print(f"Generated {len(video_frames)} frames")
                    print(f"Peak GPU memory: {final_memory['gpu_allocated']:.2f}GB")
                    
                    # Save sample frame
                    if video_frames:
                        output_dir = Path("outputs/model_test")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        middle_frame = video_frames[len(video_frames)//2]
                        if isinstance(middle_frame, torch.Tensor):
                            middle_frame = middle_frame.cpu().numpy()
                        if isinstance(middle_frame, np.ndarray):
                            if middle_frame.max() <= 1.0:
                                middle_frame = (middle_frame * 255).astype(np.uint8)
                            middle_frame = Image.fromarray(middle_frame)
                        
                        sample_path = output_dir / f"{config['name'].lower().replace(' ', '_')}_sample.png"
                        middle_frame.save(sample_path)
                        print(f"Sample saved: {sample_path}")
                    
                    print(f"✅ {config['name']} test PASSED")
                    
                    # If this is the 5B model and it worked, we're good!
                    if '5b' in config['model_id'].lower():
                        print(f"\n🎉 SUCCESS: {config['name']} is working on your RTX 4080!")
                        return True
                        
                except torch.cuda.OutOfMemoryError as e:
                    print(f"❌ GPU Out of Memory during generation: {e}")
                    
                except Exception as e:
                    print(f"❌ Generation error: {e}")
                
                # Cleanup
                del pipe
                clear_memory()
                
            except Exception as e:
                print(f"❌ Failed to load {config['name']}: {e}")
                if "component" in str(e) and "cannot be loaded" in str(e):
                    print("This appears to be a dependency/compatibility issue.")
                    print("Try updating your packages:")
                    print("pip install --upgrade transformers diffusers torch")
                clear_memory()
                continue
        
        return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main test function"""
    print("TI2V-5B Model Compatibility Test")
    print("RTX 4080 16GB VRAM")
    print("=" * 60)
    
    # System info
    print("System Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"CPU Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    
    # Run tests
    success = test_model_loading_step_by_step()
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    if success:
        print("✅ TI2V-5B model is compatible with your system!")
        print("Recommendations:")
        print("  - Use 512x512 or 720x480 resolution for best performance")
        print("  - Keep frame count between 8-24 frames")
        print("  - Use 15-25 inference steps for good quality/speed balance")
        print("  - Enable attention slicing and CPU offload for memory efficiency")
    else:
        print("❌ TI2V-5B model compatibility issues detected")
        print("Troubleshooting steps:")
        print("  1. Update packages: pip install --upgrade transformers diffusers torch")
        print("  2. Clear cache: pip cache purge")
        print("  3. Restart Python environment")
        print("  4. Try CogVideoX-2b model as alternative")

if __name__ == "__main__":
    main()