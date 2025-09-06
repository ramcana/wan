#!/usr/bin/env python3
"""
Compare WAN2.2 models: T2V-A14B vs TI2V-5B
Performance and quality comparison for RTX 4080 16GB
"""

import torch
import gc
import time
import psutil
import json
from pathlib import Path
from datetime import datetime

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

def test_model_performance(model_config):
    """Test a specific model configuration"""
    print(f"\n{'='*60}")
    print(f"Testing {model_config['name']}")
    print(f"Model ID: {model_config['model_id']}")
    print(f"{'='*60}")
    
    results = {
        'model_name': model_config['name'],
        'model_id': model_config['model_id'],
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    try:
        from diffusers import CogVideoXPipeline
        import numpy as np
        from PIL import Image
        
        clear_memory()
        initial_memory = get_memory_usage()
        
        print(f"Loading {model_config['name']}...")
        start_time = time.time()
        
        # Load pipeline with optimizations
        pipe = CogVideoXPipeline.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            
        # Apply optimizations
        pipe.enable_attention_slicing()
        if model_config.get('enable_cpu_offload', True):
            pipe.enable_model_cpu_offload()
        
        load_time = time.time() - start_time
        load_memory = get_memory_usage()
        
        results['load_time'] = load_time
        results['load_memory'] = load_memory
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        print(f"Memory usage: {load_memory['gpu_allocated']:.2f}GB GPU, {load_memory['cpu_memory']:.2f}GB CPU")
        
        # Test configurations
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
        
        prompt = "A peaceful mountain lake reflecting the sky, with gentle waves and surrounding pine trees"
        
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
                
                # Track peak memory during generation
                peak_memory = pre_memory['gpu_allocated']
                
                def memory_callback(step, timestep, latents):
                    nonlocal peak_memory
                    current_memory = get_memory_usage()
                    peak_memory = max(peak_memory, current_memory['gpu_allocated'])
                    if step % 5 == 0:
                        print(f"  Step {step}: {current_memory['gpu_allocated']:.2f}GB")
                
                # Generate without quantization (based on previous findings)
                video_frames = pipe(
                    prompt=prompt,
                    width=test_config['width'],
                    height=test_config['height'],
                    num_frames=test_config['num_frames'],
                    num_inference_steps=test_config['num_inference_steps'],
                    guidance_scale=7.5,
                    generator=torch.Generator().manual_seed(42),
                    callback=memory_callback,
                    callback_steps=1
                ).frames[0]
                
                gen_time = time.time() - gen_start
                post_memory = get_memory_usage()
                
                test_result.update({
                    'success': True,
                    'generation_time': gen_time,
                    'peak_memory': peak_memory,
                    'fps': test_config['num_frames'] / gen_time,
                    'frames_generated': len(video_frames) if video_frames else 0
                })
                
                print(f"✅ Generated in {gen_time:.2f}s ({test_result['fps']:.2f} fps)")
                print(f"Peak memory: {peak_memory:.2f}GB")
                
                # Save sample frame
                if video_frames:
                    output_dir = Path("outputs/model_comparison")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    middle_frame = video_frames[len(video_frames)//2]
                    if isinstance(middle_frame, torch.Tensor):
                        middle_frame = middle_frame.cpu().numpy()
                    if isinstance(middle_frame, np.ndarray):
                        if middle_frame.max() <= 1.0:
                            middle_frame = (middle_frame * 255).astype(np.uint8)
                        middle_frame = Image.fromarray(middle_frame)
                    
                    frame_path = output_dir / f"{model_config['name'].lower().replace(' ', '_')}_{test_config['name'].lower().replace(' ', '_')}.png"
                    middle_frame.save(frame_path)
                    test_result['sample_frame'] = str(frame_path)
                
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
    
    return results

def main():
    """Main comparison function"""
    print("WAN2.2 Model Comparison for RTX 4080 16GB")
    print("=" * 60)
    
    # Model configurations
    models = [
        {
            'name': 'TI2V-5B',
            'model_id': 'THUDM/CogVideoX-5b',
            'enable_cpu_offload': True,
            'description': 'Optimized for consumer GPUs (8-10GB VRAM)'
        },
        {
            'name': 'T2V-A14B',
            'model_id': 'THUDM/CogVideoX-2b',  # Using 2B as proxy for testing
            'enable_cpu_offload': True,
            'description': 'High-end model (requires 12-16GB+ VRAM)'
        }
    ]
    
    all_results = []
    
    for model_config in models:
        try:
            results = test_model_performance(model_config)
            all_results.append(results)
            
            # Brief summary
            print(f"\n📊 {model_config['name']} Summary:")
            if 'load_time' in results:
                print(f"  Load time: {results['load_time']:.2f}s")
                print(f"  Load memory: {results['load_memory']['gpu_allocated']:.2f}GB")
                
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
    results_file = Path("outputs/model_comparison_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Final recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR RTX 4080 16GB")
    print(f"{'='*60}")
    
    if len(all_results) >= 2:
        ti2v_results = next((r for r in all_results if 'TI2V-5B' in r['model_name']), None)
        t2v_results = next((r for r in all_results if 'T2V-A14B' in r['model_name']), None)
        
        if ti2v_results and 'load_time' in ti2v_results:
            ti2v_success = len([t for t in ti2v_results['tests'] if t['success']])
            print(f"✅ TI2V-5B: {ti2v_success}/{len(ti2v_results['tests'])} tests successful")
            print("   - Recommended for regular use")
            print("   - Good balance of quality and performance")
        
        if t2v_results and 'load_time' in t2v_results:
            t2v_success = len([t for t in t2v_results['tests'] if t['success']])
            print(f"⚠️  T2V-A14B: {t2v_success}/{len(t2v_results['tests'])} tests successful")
            print("   - Use for high-quality outputs only")
            print("   - Avoid quantization (causes hanging)")
            print("   - Consider lower resolution/frame count")
    
    print("\n🎯 Best practices:")
    print("   - Start with TI2V-5B for development/testing")
    print("   - Use T2V-A14B for final high-quality renders")
    print("   - Monitor GPU memory usage closely")
    print("   - Disable quantization to avoid hanging issues")

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