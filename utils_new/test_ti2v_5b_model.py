#!/usr/bin/env python3
"""
Test script for TI2V-5B model - optimized for consumer GPUs
This model requires ~8-10 GB VRAM and should work well on RTX 4080 16GB
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
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
    else:
        gpu_memory = gpu_memory_cached = 0
    
    cpu_memory = psutil.Process().memory_info().rss / 1024**3  # GB
    
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

def test_ti2v_5b_model():
    """Test TI2V-5B model for video generation"""
    print("=" * 60)
    print("Testing TI2V-5B Model (5B parameters)")
    print("Optimized for consumer GPUs (RTX 4080)")
    print("=" * 60)
    
    # Clear memory before starting
    clear_memory()
    initial_memory = get_memory_usage()
    print(f"Initial Memory - GPU: {initial_memory['gpu_allocated']:.2f}GB, CPU: {initial_memory['cpu_memory']:.2f}GB")
    
    try:
        from diffusers import CogVideoXPipeline
        import numpy as np
from PIL import Image
        
        # Model configuration for TI2V-5B
        model_id = "THUDM/CogVideoX-5b"  # 5B parameter model
        
        print(f"\nLoading model: {model_id}")
        print("This model is optimized for consumer GPUs...")
        
        start_time = time.time()
        
        # Load pipeline with optimizations for consumer GPU
        pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            
        # Enable memory efficient attention
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()  # Offload to CPU when not in use
        
        load_time = time.time() - start_time
        load_memory = get_memory_usage()
        
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Memory after loading - GPU: {load_memory['gpu_allocated']:.2f}GB, CPU: {load_memory['cpu_memory']:.2f}GB")
        
        # Test parameters optimized for RTX 4080
        test_configs = [
            {
                "name": "Low Resource (512x512, 8 frames)",
                "width": 512,
                "height": 512,
                "num_frames": 8,
                "num_inference_steps": 20
            },
            {
                "name": "Medium Quality (720x480, 16 frames)",
                "width": 720,
                "height": 480,
                "num_frames": 16,
                "num_inference_steps": 25
            },
            {
                "name": "High Quality (1024x576, 24 frames)",
                "width": 1024,
                "height": 576,
                "num_frames": 24,
                "num_inference_steps": 30
            }
        ]
        
        # Test prompt
        prompt = "A serene lake with gentle ripples, surrounded by autumn trees with golden leaves swaying in the breeze"
        
        for i, config in enumerate(test_configs):
            print(f"\n--- Test {i+1}: {config['name']} ---")
            
            try:
                clear_memory()
                pre_gen_memory = get_memory_usage()
                
                print(f"Generating {config['num_frames']} frames at {config['width']}x{config['height']}")
                print(f"Pre-generation GPU memory: {pre_gen_memory['gpu_allocated']:.2f}GB")
                
                gen_start = time.time()
                
                # Generate video with progress callback
                def progress_callback(step, timestep, latents):
                    if step % 5 == 0:
                        current_memory = get_memory_usage()
                        print(f"Step {step}: GPU {current_memory['gpu_allocated']:.2f}GB")
                
                video_frames = pipe(
                    prompt=prompt,
                    width=config['width'],
                    height=config['height'],
                    num_frames=config['num_frames'],
                    num_inference_steps=config['num_inference_steps'],
                    guidance_scale=7.5,
                    generator=torch.Generator().manual_seed(42),
                    callback=progress_callback,
                    callback_steps=1
                ).frames[0]
                
                gen_time = time.time() - gen_start
                post_gen_memory = get_memory_usage()
                
                print(f"Generation completed in {gen_time:.2f} seconds")
                print(f"Peak GPU memory: {post_gen_memory['gpu_allocated']:.2f}GB")
                print(f"Frames per second: {config['num_frames']/gen_time:.2f}")
                
                # Save sample frame
                if video_frames:
                    output_dir = Path("outputs/ti2v_5b_test")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    sample_frame = video_frames[len(video_frames)//2]  # Middle frame
                    if isinstance(sample_frame, torch.Tensor):
                        sample_frame = sample_frame.cpu().numpy()
                    if isinstance(sample_frame, np.ndarray):
                        if sample_frame.max() <= 1.0:
                            sample_frame = (sample_frame * 255).astype(np.uint8)
                        sample_frame = Image.fromarray(sample_frame)
                    
                    sample_path = output_dir / f"sample_frame_test_{i+1}.png"
                    sample_frame.save(sample_path)
                    print(f"Sample frame saved: {sample_path}")
                
                print(f"✅ Test {i+1} completed successfully")
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ GPU Out of Memory for {config['name']}")
                print(f"Error: {e}")
                clear_memory()
                continue
                
            except Exception as e:
                print(f"❌ Error in {config['name']}: {e}")
                clear_memory()
                continue
        
        print(f"\n{'='*60}")
        print("TI2V-5B Model Test Summary")
        print(f"{'='*60}")
        print(f"Model: {model_id}")
        print(f"Load time: {load_time:.2f} seconds")
        print(f"Peak GPU memory: {max(load_memory['gpu_allocated'], post_gen_memory['gpu_allocated']):.2f}GB")
        print("Recommendation: TI2V-5B is well-suited for RTX 4080 16GB")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure diffusers and required dependencies are installed")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
traceback.print_exc()
        
    finally:
        clear_memory()
        final_memory = get_memory_usage()
        print(f"Final memory - GPU: {final_memory['gpu_allocated']:.2f}GB, CPU: {final_memory['cpu_memory']:.2f}GB")

if __name__ == "__main__":
    # Check system requirements
    print("System Check:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print(f"CPU Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print()
    
    test_ti2v_5b_model()