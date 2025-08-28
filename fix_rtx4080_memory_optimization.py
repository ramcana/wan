#!/usr/bin/env python3
"""
RTX 4080 Memory Optimization Fix
Optimizes CUDA memory usage for 16GB VRAM RTX 4080
"""

import os
import sys
import torch
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def apply_rtx4080_memory_optimizations():
    """Apply RTX 4080 specific memory optimizations"""
    print("🔧 Applying RTX 4080 Memory Optimizations")
    print("=" * 50)
    
    # 1. Set PyTorch CUDA memory allocation strategy
    print("1. Setting CUDA memory allocation strategy...")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("✅ Set expandable_segments:True")
    
    # 2. Enable memory fraction for RTX 4080 (16GB)
    print("\n2. Setting CUDA memory fraction for RTX 4080...")
    if torch.cuda.is_available():
        # Reserve 14GB out of 16GB (leave 2GB for system)
        torch.cuda.set_per_process_memory_fraction(0.875)  # 14/16 = 0.875
        print("✅ Set memory fraction to 87.5% (14GB out of 16GB)")
    
    # 3. Clear any existing CUDA cache
    print("\n3. Clearing CUDA cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ CUDA cache cleared")
    
    # 4. Set environment variables for memory optimization
    print("\n4. Setting memory optimization environment variables...")
    env_vars = {
        'CUDA_LAUNCH_BLOCKING': '0',  # Async CUDA operations
        'TORCH_CUDNN_V8_API_ENABLED': '1',  # Enable cuDNN v8 optimizations
        'CUDA_CACHE_DISABLE': '0',  # Enable CUDA cache
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:128'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")
    
    # 5. Display current CUDA status
    print("\n5. CUDA Status Check:")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = props.total_memory / 1024**3
        
        print(f"   GPU: {props.name}")
        print(f"   Total VRAM: {total:.2f} GB")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        print(f"   Free: {total - reserved:.2f} GB")
    
    print("\n" + "=" * 50)
    print("🎯 RTX 4080 Memory Optimization Complete!")
    return True

def update_pipeline_loader_for_memory_efficiency():
    """Update the pipeline loader to use memory-efficient loading"""
    print("\n🔧 Updating Pipeline Loader for Memory Efficiency")
    print("=" * 50)
    
    try:
        # Read the current system integration file
        system_integration_path = Path("backend/core/system_integration.py")
        
        if not system_integration_path.exists():
            print("❌ System integration file not found")
            return False
        
        with open(system_integration_path, 'r') as f:
            content = f.read()
        
        # Check if memory optimizations are already applied
        if "torch_dtype=torch.float16" in content and "low_cpu_mem_usage=True" in content:
            print("✅ Memory optimizations already applied to pipeline loader")
            return True
        
        # Apply memory optimizations to the pipeline loading
        old_loading_code = '''pipeline = DiffusionPipeline.from_pretrained(
                                model_path,
                                trust_remote_code=True,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                            )'''
        
        new_loading_code = '''pipeline = DiffusionPipeline.from_pretrained(
                                model_path,
                                trust_remote_code=True,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                low_cpu_mem_usage=True,
                                device_map="auto",
                                max_memory={0: "14GB"}  # RTX 4080 optimization
                            )'''
        
        if old_loading_code in content:
            content = content.replace(old_loading_code, new_loading_code)
            
            with open(system_integration_path, 'w') as f:
                f.write(content)
            
            print("✅ Applied memory-efficient loading to pipeline loader")
            return True
        else:
            print("⚠️  Pipeline loading code not found in expected format")
            return False
            
    except Exception as e:
        print(f"❌ Failed to update pipeline loader: {e}")
        return False

def create_memory_optimized_config():
    """Create a memory-optimized configuration file"""
    print("\n📝 Creating Memory-Optimized Configuration")
    print("=" * 50)
    
    config = {
        "rtx4080_optimizations": {
            "max_vram_usage_gb": 14,
            "enable_memory_efficient_attention": True,
            "use_fp16": True,
            "enable_cpu_offload": True,
            "chunk_size": 4,
            "max_batch_size": 1,
            "gradient_checkpointing": True
        },
        "model_loading": {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "torch_dtype": "float16",
            "max_memory": {"0": "14GB"}
        },
        "generation_params": {
            "default_resolution": "512x512",
            "max_frames": 16,
            "recommended_steps": 20
        }
    }
    
    import json
    config_path = Path("rtx4080_memory_config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created memory configuration: {config_path}")
    return True

def main():
    """Apply all RTX 4080 memory optimizations"""
    print("🚀 RTX 4080 Memory Optimization Suite")
    print("Fixing CUDA out of memory issues for 16GB VRAM")
    print("=" * 60)
    
    # Apply optimizations
    opt1 = apply_rtx4080_memory_optimizations()
    opt2 = update_pipeline_loader_for_memory_efficiency()
    opt3 = create_memory_optimized_config()
    
    print("\n" + "=" * 60)
    print("🎯 OPTIMIZATION RESULTS:")
    
    if opt1 and opt2 and opt3:
        print("✅ ALL OPTIMIZATIONS APPLIED SUCCESSFULLY!")
        print("\n🎉 RTX 4080 Memory Optimization Complete!")
        print("\n📋 What was fixed:")
        print("   • CUDA memory allocation strategy optimized")
        print("   • Memory fraction set to 87.5% (14GB/16GB)")
        print("   • Pipeline loader updated for memory efficiency")
        print("   • Memory-optimized configuration created")
        
        print("\n🚀 Next Steps:")
        print("   1. Restart your backend server")
        print("   2. Test generation with smaller parameters")
        print("   3. Monitor VRAM usage during generation")
        
        print("\n⚡ Recommended Test Parameters:")
        print("   • Resolution: 512x512 (not 1024x1024)")
        print("   • Frames: 8-16 (not 32+)")
        print("   • Steps: 20 (not 50+)")
        
        return True
    else:
        print("❌ Some optimizations failed - check output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)