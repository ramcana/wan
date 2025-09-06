#!/usr/bin/env python3
"""
RTX 4080 Model Loading Optimization
Optimize model loading speed for RTX 4080 with 16GB VRAM
"""

import json
import os
import sys
import torch
import time
from pathlib import Path

def optimize_loading_configuration():
    """Optimize backend configuration for faster model loading"""
    
    print("🚀 RTX 4080 Model Loading Optimization")
    print("=" * 50)
    
    # Backend config path
    backend_config_path = Path("backend/config.json")
    
    if not backend_config_path.exists():
        print("❌ Backend config not found")
        return False
    
    # Load current config
    with open(backend_config_path, 'r') as f:
        config = json.load(f)
    
    print("📝 Current Configuration:")
    print(f"   Preload models: {config['models'].get('preload_models', False)}")
    print(f"   Model cache size: {config['models'].get('model_cache_size', 2)}")
    print(f"   VRAM limit: {config['hardware'].get('vram_limit_gb', 14)}GB")
    print(f"   Quantization: {config['models'].get('quantization_level', 'bf16')}")
    
    # Backup current config
    backup_path = backend_config_path.with_suffix('.backup.json')
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Backup saved: {backup_path}")
    
    # Apply RTX 4080 optimizations for faster loading
    optimizations = {
        # Model loading optimizations
        "models": {
            **config["models"],
            "preload_models": True,  # Preload the default model
            "model_cache_size": 1,   # Keep only one model in memory
            "auto_optimize": True,
            "enable_offloading": False,  # Disable offloading with 16GB VRAM
            "quantization_enabled": True,
            "quantization_level": "bf16"  # Best balance for RTX 4080
        },
        
        # Hardware optimizations for RTX 4080
        "hardware": {
            **config["hardware"],
            "vram_limit_gb": 15,  # Use more VRAM for faster loading
            "cpu_threads": 16,    # Optimal for model loading
            "enable_mixed_precision": True,
            "enable_attention_slicing": False,  # Disable for speed with enough VRAM
            "enable_memory_efficient_attention": True,
            "enable_xformers": True,
            "vae_tile_size": 1024,  # Larger tiles for RTX 4080
            "attention_slice_size": "auto"
        },
        
        # Performance optimizations
        "performance": {
            **config.get("performance", {}),
            "enable_torch_compile": False,  # Disable during loading for speed
            "torch_compile_mode": "default",
            "enable_cuda_graphs": False,
            "memory_format": "channels_last",
            "enable_flash_attention": False  # Can slow down loading
        },
        
        # Loading-specific optimizations
        "loading": {
            "use_fast_loading": True,
            "skip_safety_checker": True,  # Faster loading
            "use_safetensors": True,      # Faster format
            "torch_dtype": "bfloat16",    # Consistent with quantization
            "device_map": None,           # Don't use device_map for RTX 4080
            "low_cpu_mem_usage": True,    # Load directly to GPU
            "use_auth_token": False,
            "cache_dir": "models/.cache"
        }
    }
    
    # Update config with optimizations
    for key, value in optimizations.items():
        config[key] = value
    
    # Save optimized config
    with open(backend_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Applied RTX 4080 Loading Optimizations:")
    print("   • Enabled model preloading")
    print("   • Increased VRAM limit to 15GB")
    print("   • Disabled attention slicing (not needed)")
    print("   • Optimized for bf16 quantization")
    print("   • Enabled fast loading options")
    print("   • Disabled CPU offloading")
    
    return True

def create_fast_loading_script():
    """Create a script to preload models quickly"""
    
    script_content = '''#!/usr/bin/env python3
"""
Fast Model Preloader for RTX 4080
Preload models with optimized settings
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def preload_models():
    """Preload models with RTX 4080 optimizations"""
    
    print("🚀 Fast Model Preloader - RTX 4080")
    print("=" * 40)
    
    # Set optimal environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        # Import generation service
        from services.generation_service import GenerationService
        
        print("\\n📦 Initializing Generation Service...")
        start_time = time.time()
        
        # Create service with optimized settings
        service = GenerationService()
        
        # Initialize with preloading
        print("🔄 Preloading default model (T2V-A14B)...")
        service.initialize()
        
        load_time = time.time() - start_time
        print(f"✅ Model preloaded in {load_time:.1f} seconds")
        
        # Check VRAM usage
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_cached = torch.cuda.memory_reserved() / 1e9
        
        print(f"\\n📊 VRAM Usage:")
        print(f"   Allocated: {vram_used:.1f}GB")
        print(f"   Cached: {vram_cached:.1f}GB")
        print(f"   Available: {16 - vram_cached:.1f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Preloading failed: {e}")
        return False

if __name__ == "__main__":
    success = preload_models()
    exit(0 if success else 1)
'''
    
    script_path = Path("preload_models_rtx4080.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created fast loading script: {script_path}")
    return script_path

def create_optimized_startup_batch():
    """Create optimized startup batch file"""
    
    batch_content = '''@echo off
echo RTX 4080 Optimized Startup for WAN2.2
echo =====================================

echo.
echo 1. Optimizing model loading configuration...
python optimize_model_loading_rtx4080.py

echo.
echo 2. Preloading models (this may take a few minutes first time)...
python preload_models_rtx4080.py

echo.
echo 3. Starting backend server...
cd backend
start "WAN Backend" python app.py

echo.
echo 4. Waiting for backend to initialize...
timeout /t 10 /nobreak > nul

echo.
echo 5. Starting frontend...
cd ../frontend
start "WAN Frontend" npm run dev

echo.
echo ✅ Both servers starting with RTX 4080 optimizations!
echo Backend: http://localhost:9000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul
'''
    
    batch_path = Path("start_optimized_rtx4080.bat")
    with open(batch_path, 'w') as f:
        f.write(batch_content)
    
    print(f"✅ Created optimized startup script: {batch_path}")
    return batch_path

def main():
    print("🎯 RTX 4080 Model Loading Optimization Suite")
    print("=" * 60)
    
    # 1. Optimize configuration
    if not optimize_loading_configuration():
        print("❌ Configuration optimization failed")
        return False
    
    # 2. Create fast loading script
    preload_script = create_fast_loading_script()
    
    # 3. Create optimized startup
    startup_script = create_optimized_startup_batch()
    
    print(f"\\n🎉 RTX 4080 Optimization Complete!")
    print("=" * 40)
    print("\\n📋 Next Steps:")
    print("1. Restart your backend server")
    print("2. Run the optimized startup:")
    print(f"   {startup_script}")
    print("\\n💡 Expected Improvements:")
    print("   • Faster model loading (preloaded)")
    print("   • Better VRAM utilization (15GB limit)")
    print("   • Optimized for bf16 quantization")
    print("   • No CPU offloading overhead")
    
    print("\\n⚠️  Note: First model load will still take time,")
    print("   but subsequent generations will be much faster!")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)