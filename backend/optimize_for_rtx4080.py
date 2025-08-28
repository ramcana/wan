#!/usr/bin/env python3
"""
RTX 4080 Optimization Script
Optimizes Wan2.2 for RTX 4080 16GB VRAM with Threadripper PRO 5995WX
"""

import json
import torch
import psutil
from pathlib import Path

def detect_system_specs():
    """Detect and return system specifications"""
    specs = {}
    
    # GPU Detection
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        specs['gpu'] = {
            'name': gpu_props.name,
            'vram_gb': gpu_props.total_memory / (1024**3),
            'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
            'multiprocessors': gpu_props.multi_processor_count
        }
    else:
        specs['gpu'] = None
    
    # CPU Detection
    specs['cpu'] = {
        'cores': psutil.cpu_count(logical=False),
        'threads': psutil.cpu_count(logical=True),
        'frequency_ghz': psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else None
    }
    
    # RAM Detection
    memory = psutil.virtual_memory()
    specs['ram'] = {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3)
    }
    
    return specs

def create_optimized_config(specs):
    """Create optimized configuration for RTX 4080 system"""
    
    # Base configuration optimized for RTX 4080
    config = {
        "generation": {
            "mode": "real",
            "enable_real_models": True,
            "fallback_to_mock": False,  # Disable fallback since we have good hardware
            "auto_download_models": True,
            "max_concurrent_generations": 1,  # Conservative for VRAM management
            "generation_timeout_minutes": 45,  # Longer timeout for complex generations
            "enable_progress_tracking": True
        },
        "models": {
            "base_path": "models",
            "auto_optimize": True,
            "enable_offloading": False,  # Disable with 16GB VRAM
            "vram_management": True,
            "quantization_enabled": True,
            "quantization_level": "bf16",  # Good balance for RTX 4080
            "supported_types": ["t2v-A14B", "i2v-A14B", "ti2v-5B"],
            "default_model": "t2v-A14B",
            "preload_models": False,  # Load on demand to save VRAM
            "model_cache_size": 2  # Cache up to 2 models
        },
        "hardware": {
            "auto_detect": True,
            "optimize_for_hardware": True,
            "vram_limit_gb": 14,  # Leave 2GB buffer for system
            "cpu_threads": min(32, specs['cpu']['threads']),  # Use up to 32 threads
            "enable_mixed_precision": True,
            "enable_attention_slicing": True,
            "enable_memory_efficient_attention": True,
            "enable_xformers": True,
            "vae_tile_size": 512,  # Larger tiles for better performance
            "attention_slice_size": "auto"
        },
        "optimization": {
            "max_vram_usage_gb": 14,
            "enable_offload": False,
            "vae_tile_size": 512,
            "quantization_level": "bf16",
            "enable_sequential_offload": False,
            "attention_slice_size": "auto",
            "enable_xformers": True,
            "vae_tile_size_range": [256, 1024]
        },
        "websocket": {
            "enable_progress_updates": True,
            "detailed_progress": True,
            "resource_monitoring": True,
            "update_interval_seconds": 0.5,  # Faster updates
            "vram_monitoring": True
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
            "cors_origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 120  # Higher limit for powerful system
            },
            "max_request_size": "100MB"
        },
        "database": {
            "url": "sqlite:///./generation_tasks.db",
            "echo": False,
            "pool_size": 20,  # Higher pool for better concurrency
            "max_overflow": 30
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/fastapi_backend.log",
            "max_file_size": "50MB",
            "backup_count": 5
        },
        "security": {
            "secret_key": "your-secret-key-change-in-production",
            "algorithm": "HS256",
            "access_token_expire_minutes": 60
        },
        "performance": {
            "enable_torch_compile": True,  # RTX 4080 supports this well
            "torch_compile_mode": "default",
            "enable_cuda_graphs": False,  # Can cause issues with dynamic shapes
            "memory_format": "channels_last",
            "enable_flash_attention": True
        }
    }
    
    # Adjust based on detected specs
    if specs['gpu']:
        vram_gb = specs['gpu']['vram_gb']
        if vram_gb >= 16:
            # RTX 4080 optimizations
            config['hardware']['vram_limit_gb'] = int(vram_gb - 2)  # Leave 2GB buffer
            config['optimization']['max_vram_usage_gb'] = int(vram_gb - 2)
            config['models']['enable_offloading'] = False
            config['optimization']['enable_offload'] = False
            config['generation']['max_concurrent_generations'] = 1
        elif vram_gb >= 12:
            # RTX 4070 Ti / 3080 Ti optimizations
            config['hardware']['vram_limit_gb'] = int(vram_gb - 1)
            config['optimization']['max_vram_usage_gb'] = int(vram_gb - 1)
            config['models']['enable_offloading'] = True
            config['optimization']['enable_offload'] = True
        else:
            # Lower VRAM cards
            config['hardware']['vram_limit_gb'] = int(vram_gb - 1)
            config['optimization']['max_vram_usage_gb'] = int(vram_gb - 1)
            config['models']['enable_offloading'] = True
            config['optimization']['enable_offload'] = True
            config['optimization']['quantization_level'] = "int8"
    
    # Adjust for high core count CPU
    if specs['cpu']['threads'] >= 64:
        config['hardware']['cpu_threads'] = 32  # Don't use all threads
        config['api']['workers'] = 4
    elif specs['cpu']['threads'] >= 32:
        config['hardware']['cpu_threads'] = 16
        config['api']['workers'] = 2
    
    return config

def backup_existing_config():
    """Backup existing configuration"""
    config_path = Path("config.json")
    if config_path.exists():
        backup_path = Path("config_backup_rtx4080.json")
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"✅ Backed up existing config to {backup_path}")
        return True
    return False

def apply_optimizations():
    """Apply RTX 4080 optimizations"""
    print("🚀 RTX 4080 OPTIMIZATION SCRIPT")
    print("=" * 50)
    
    # Detect system
    print("\n1. 🔍 Detecting System Specifications...")
    specs = detect_system_specs()
    
    if specs['gpu']:
        gpu = specs['gpu']
        print(f"   GPU: {gpu['name']}")
        print(f"   VRAM: {gpu['vram_gb']:.1f}GB")
        print(f"   Compute: {gpu['compute_capability']}")
    else:
        print("   ❌ No CUDA GPU detected!")
        return False
    
    print(f"   CPU: {specs['cpu']['threads']} threads")
    print(f"   RAM: {specs['ram']['total_gb']:.1f}GB")
    
    # Backup existing config
    print("\n2. 💾 Backing up existing configuration...")
    backup_existing_config()
    
    # Create optimized config
    print("\n3. ⚙️ Creating optimized configuration...")
    config = create_optimized_config(specs)
    
    # Write new config
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("   ✅ Optimized configuration saved to config.json")
    
    # Create VRAM validation fix
    print("\n4. 🔧 Creating VRAM validation fix...")
    create_vram_validation_fix(specs)
    
    # Performance recommendations
    print("\n5. 📈 Performance Recommendations:")
    print("   • Use resolutions up to 1920x1080 comfortably")
    print("   • Video duration up to 8 seconds should work well")
    print("   • Consider bf16 quantization for best quality/performance balance")
    print("   • Monitor VRAM usage via /api/v1/performance/status")
    
    print(f"\n🎉 RTX 4080 optimization complete!")
    print("   Restart the backend server to apply changes")
    
    return True

def create_vram_validation_fix(specs):
    """Create a VRAM validation fix for the frontend"""
    
    vram_fix_script = f'''#!/usr/bin/env python3
"""
VRAM Validation Fix for RTX 4080
Ensures proper VRAM detection and validation
"""

import torch
import json
from pathlib import Path

def get_actual_vram():
    """Get actual VRAM from GPU"""
    if not torch.cuda.is_available():
        return 0
    
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)  # Convert to GB
    except:
        return 0

def update_frontend_vram_config():
    """Update frontend VRAM configuration"""
    actual_vram = get_actual_vram()
    
    if actual_vram >= 16:
        # RTX 4080 configuration
        vram_config = {{
            "total_vram_gb": {specs['gpu']['vram_gb']:.1f},
            "usable_vram_gb": {specs['gpu']['vram_gb'] - 2:.1f},
            "vram_estimates": {{
                "1280x720": 6,
                "1280x704": 6,
                "1920x1080": 10,
                "1920x1088": 10
            }},
            "duration_multiplier": 0.15,  # VRAM per additional second
            "max_safe_duration": 10,
            "warnings": {{
                "high_vram_threshold": 12,
                "very_high_vram_threshold": 14
            }}
        }}
    else:
        # Fallback for other GPUs
        vram_config = {{
            "total_vram_gb": actual_vram,
            "usable_vram_gb": max(0, actual_vram - 1),
            "vram_estimates": {{
                "1280x720": 8,
                "1280x704": 8,
                "1920x1080": 12,
                "1920x1088": 12
            }},
            "duration_multiplier": 0.2,
            "max_safe_duration": 6,
            "warnings": {{
                "high_vram_threshold": actual_vram * 0.8,
                "very_high_vram_threshold": actual_vram * 0.9
            }}
        }}
    
    # Save VRAM config
    config_path = Path("frontend_vram_config.json")
    with open(config_path, 'w') as f:
        json.dump(vram_config, f, indent=2)
    
    print(f"VRAM config saved: {actual_vram:.1f}GB detected")
    return vram_config

if __name__ == "__main__":
    update_frontend_vram_config()
'''
    
    with open("fix_vram_validation.py", 'w', encoding='utf-8') as f:
        f.write(vram_fix_script)
    
    print("   VRAM validation fix created: fix_vram_validation.py")

if __name__ == "__main__":
    apply_optimizations()