#!/usr/bin/env python3
"""
Compare actual WAN models for RTX 4080 16GB:
- Wan-AI/Wan2.2-TI2V-5B-Diffusers (Text+Image to Video, 5B params)
- Wan-AI/Wan2.2-T2V-A14B-Diffusers (Text to Video, 14B params)
- Wan-AI/Wan2.2-I2V-A14B-Diffusers (Image to Video, 14B params)
"""

import torch
import gc
import time
import psutil
import json
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np

# Add current directory to path
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

def create_test_image():
    """Create a simple test image for I2V and TI2V models"""
    # Create a simple gradient image
    img = Image.new('RGB', (512, 512))
    pixels = img.load()
    
    for i in range(512):
        for j in range(512):
            # Create a blue to white gradient
            intensity = int((i + j) / 2 * 255 / 512)
            pixels[i, j] = (100, 150, 255 - intensity//2)
    
    return img

def test_wan_model(model_config):
    """Test a specific WAN model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_config['name']}")
    print(f"Model ID: {model_config['model_id']}")
    print(f"Type: {model_config['type']}")
    print(f"{'='*60}")
    
    results = {
        'model_name': model_config['name'],
        'model_id': model_config['model_id'],
        'model_type': model_config['type'],
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    try:
        # Apply WAN compatibility if available
        try:
            from wan22_compatibility_clean import apply_wan22_compatibility
            apply_wan22_compatibility()
            print("✅ WAN compatibility layer applied")
        except ImportError:
            print("⚠️ WAN compatibility layer not available")
        
        from diffusers import DiffusionPipeline
        
        clear_memory()
        initial_memory = get_memory_usage()
        
        print(f"Loading {model_config['name']}...")
        start_time = time.time()
        
        # Load WAN model with trust_remote_code
        pipe = DiffusionPipeline.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch.float16,
            trust_remote_code=True,  # Required for WAN models
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            
        # Apply optimizations
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
        
        load_time = time.time() - start_time
        load_memory = get_memory_usage()
        
        results['load_time'] = load_time
        results['load_memory'] = load_memory
        results['pipeline_class'] = type(pipe).__name__
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        print(f"Pipeline class: {type(pipe).__name__}")
        print(f"Memory usage: {load_memory['gpu_allocated']:.2f}GB GPU, {load_memory['cpu_memory']:.2f}GB CPU")
        
        # Test configurations based on model type
        if model_config['type'] == 'TI2V':
            # Text + Image to Video
            test_configs = [
                {
                    "name": "TI2V Quick Test (512x512, 8 frames)",
                    "width": 512,
                    "height": 512,
                    "num_frames": 8,
                    "num_inference_steps": 15,
                    "requires_image": True
                },
                {
                    "name": "TI2V Standard Test (720x480, 16 frames)",
                    "width": 720,
                    "height": 480,
                    "num_frames": 16,
                    "num_inference_steps": 20,
                    "requires_image": True
                }
            ]
        elif model_config['type'] == 'I2V':
            # Image to Video
            test_configs = [
                {
                    "name": "I2V Quick Test (512x512, 8 frames)",
                    "width": 512,
                    "height": 512,
                    "num_frames": 8,
                    "num_inference_steps": 15,
                    "requires_image": True
                }
            ]
        else:  # T2V
            # Text to Video
            test_configs = [
                {
                    "name": "T2V Quick Test (512x512, 8 frames)",
                    "width": 512,
                    "height": 512,
                    "num_frames": 8,
                    "num_inference_steps": 15,
                    "requires_image": False
                }
            ]
        
        # Test prompts
        test_prompt = "A serene mountain lake with gentle ripples, surrounded by pine trees swaying in the breeze"
        test_image = create_test_image() if any(config.get('requires_image') for config in test_configs) else None
        
        for test_config in test_configs:
            print(f"\n--- {test_config['name']} ---")
            
            test_result = {
                'config': test_config,
                'success': False,
                'error': None
            }
            
            try:
                clear_memory()
                pre_memory = get_memory_usage()
                
                gen_start = time.time()
                peak_memory = pre_memory['gpu_allocated']
                
                def memory_callback(step, timestep, latents):
                    nonlocal peak_memory
                    current_memory = get_memory_usage()
                    peak_memory = max(peak_memory, current_memory['gpu_allocated'])
                    if step % 5 == 0:
                        print(f"  Step {step}: {current_memory['gpu_allocated']:.2f}GB")
                
                # Prepare generation arguments
                gen_kwargs = {
                    'prompt': test_prompt,
                    'width': test_config['width'],
                    'height': test_config['height'],
                    'num_frames': test_config['num_frames'],
                    'num_inference_steps': test_config['num_inference_steps'],
                    'guidance_scale': 7.5,
                    'generator': torch.Generator().manual_seed(42)
                }
                
                # Add image if required
                if test_config.get('requires_image') and test_image:
                    gen_kwargs['image'] = test_image
                
                # Add callback if supported
                try:
                    gen_kwargs['callback'] = memory_callback
                    gen_kwargs['callback_steps'] = 1
                    result = pipe(**gen_kwargs)
                except TypeError:
                    # Remove callback and try again
                    gen_kwargs.pop('callback', None)
                    gen_kwargs.pop('callback_steps', None)
                    result = pipe(**gen_kwargs)
                
                # Extract video frames
                if hasattr(result, 'frames'):
                    video_frames = result.frames[0]
                elif hasattr(result, 'videos'):
                    video_frames = result.videos[0]
                else:
                    video_frames = result
                
                gen_time = time.time() - gen_start
                
                test_result.update({
                    'success': True,
                    'generation_time': gen_time,
                    'peak_memory': peak_memory,
                    'fps': test_config['num_frames'] / gen_time,
                    'frames_generated': len(video_frames) if video_frames else 0
                })
                
                print(f"✅ Generated in {gen_time:.2f}s ({test_result['fps']:.2f} fps)")
                print(f"Peak memory: {peak_memory:.2f}GB")
                print(f"Frames generated: {len(video_frames) if video_frames else 0}")
                
                # Save sample frame
                if video_frames and len(video_frames) > 0:
                    output_dir = Path("outputs/wan_model_comparison")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    middle_frame = video_frames[len(video_frames)//2]
                    
                    # Convert to PIL Image
                    if isinstance(middle_frame, torch.Tensor):
                        middle_frame = middle_frame.cpu().numpy()
                    if isinstance(middle_frame, np.ndarray):
                        if middle_frame.max() <= 1.0:
                            middle_frame = (middle_frame * 255).astype(np.uint8)
                        if len(middle_frame.shape) == 3 and middle_frame.shape[0] == 3:
                            middle_frame = middle_frame.transpose(1, 2, 0)
                        middle_frame = Image.fromarray(middle_frame)
                    
                    frame_path = output_dir / f"{model_config['name'].lower().replace(' ', '_')}_{test_config['name'].lower().replace(' ', '_')}.png"
                    middle_frame.save(frame_path)
                    test_result['sample_frame'] = str(frame_path)
                    print(f"Sample frame saved: {frame_path}")
                
            except torch.cuda.OutOfMemoryError as e:
                test_result.update({
                    'success': False,
                    'error': f"GPU Out of Memory: {str(e)}"
                })
                print(f"❌ GPU OOM: {e}")
                
            except Exception as e:
                test_result.update({
                    'success': False,
                    'error': str(e)
                })
                print(f"❌ Error: {e}")
                
            finally:
                clear_memory()
                
            results['tests'].append(test_result)
        
        # Cleanup
        del pipe
        clear_memory()
        
    except Exception as e:
        results['load_error'] = str(e)
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def main():
    """Main comparison function"""
    print("WAN Model Comparison for RTX 4080 16GB")
    print("Testing actual Wan-AI models")
    print("=" * 60)
    
    # WAN model configurations
    wan_models = [
        {
            'name': 'WAN TI2V-5B',
            'model_id': 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
            'type': 'TI2V',
            'description': 'Text+Image to Video (5B params, optimized for consumer GPUs)'
        },
        {
            'name': 'WAN T2V-A14B',
            'model_id': 'Wan-AI/Wan2.2-T2V-A14B-Diffusers',
            'type': 'T2V',
            'description': 'Text to Video (14B params, high-end model)'
        },
        {
            'name': 'WAN I2V-A14B',
            'model_id': 'Wan-AI/Wan2.2-I2V-A14B-Diffusers',
            'type': 'I2V',
            'description': 'Image to Video (14B params, high-end model)'
        }
    ]
    
    all_results = []
    
    for model_config in wan_models:
        try:
            results = test_wan_model(model_config)
            all_results.append(results)
            
            # Brief summary
            print(f"\n📊 {model_config['name']} Summary:")
            if 'load_time' in results:
                print(f"  Load time: {results['load_time']:.2f}s")
                print(f"  Load memory: {results['load_memory']['gpu_allocated']:.2f}GB")
                print(f"  Pipeline: {results.get('pipeline_class', 'Unknown')}")
                
                successful_tests = [t for t in results['tests'] if t['success']]
                if successful_tests:
                    avg_fps = sum(t['fps'] for t in successful_tests) / len(successful_tests)
                    max_memory = max(t['peak_memory'] for t in successful_tests)
                    print(f"  Average FPS: {avg_fps:.2f}")
                    print(f"  Peak memory: {max_memory:.2f}GB")
                    print(f"  Success rate: {len(successful_tests)}/{len(results['tests'])}")
                else:
                    print("  ❌ No successful generations")
            else:
                print("  ❌ Failed to load")
                
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error testing {model_config['name']}: {e}")
    
    # Save detailed results
    results_file = Path("outputs/wan_model_comparison_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Final recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR RTX 4080 16GB")
    print(f"{'='*60}")
    
    if all_results:
        ti2v_results = next((r for r in all_results if 'TI2V-5B' in r['model_name']), None)
        t2v_results = next((r for r in all_results if 'T2V-A14B' in r['model_name']), None)
        i2v_results = next((r for r in all_results if 'I2V-A14B' in r['model_name']), None)
        
        print("Model Performance Summary:")
        
        if ti2v_results and 'load_time' in ti2v_results:
            ti2v_success = len([t for t in ti2v_results['tests'] if t['success']])
            print(f"✅ WAN TI2V-5B: {ti2v_success}/{len(ti2v_results['tests'])} tests successful")
            print("   - Best choice for RTX 4080 (5B parameters)")
            print("   - Supports both text and image input")
            print("   - Good balance of quality and performance")
        
        if t2v_results and 'load_time' in t2v_results:
            t2v_success = len([t for t in t2v_results['tests'] if t['success']])
            print(f"⚠️  WAN T2V-A14B: {t2v_success}/{len(t2v_results['tests'])} tests successful")
            print("   - High memory usage (14B parameters)")
            print("   - Use for high-quality text-only generations")
        
        if i2v_results and 'load_time' in i2v_results:
            i2v_success = len([t for t in i2v_results['tests'] if t['success']])
            print(f"⚠️  WAN I2V-A14B: {i2v_success}/{len(i2v_results['tests'])} tests successful")
            print("   - High memory usage (14B parameters)")
            print("   - Use for high-quality image-to-video")
    
    print("\n🎯 Best practices for WAN models:")
    print("   - Start with TI2V-5B for development and testing")
    print("   - Use A14B models for final high-quality renders")
    print("   - Always use trust_remote_code=True")
    print("   - Enable attention slicing and CPU offload")
    print("   - Avoid quantization (causes hanging)")
    print("   - Monitor GPU memory usage closely")

if __name__ == "__main__":
    # System check
    print("System Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"CPU Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print()
    
    main()
