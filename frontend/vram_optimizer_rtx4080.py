#!/usr/bin/env python3
"""
Frontend VRAM Optimizer for RTX 4080
Patches the frontend UI to properly handle RTX 4080's 16GB VRAM
"""

import re
import sys
from pathlib import Path

def patch_ui_vram_validation():
    """Patch the UI file to fix VRAM validation for RTX 4080"""
    
    ui_path = Path("ui.py")
    if not ui_path.exists():
        print("❌ ui.py not found in current directory")
        return False
    
    print("🔧 Patching UI VRAM validation for RTX 4080...")
    
    # Read the current UI file
    with open(ui_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original
    backup_path = Path("ui_backup_rtx4080.py")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Backed up original UI to {backup_path}")
    
    # Patch 1: Update VRAM estimates for RTX 4080
    vram_estimates_pattern = r'vram_estimates\s*=\s*\{[^}]+\}'
    new_vram_estimates = '''vram_estimates = {
                "1280x720": 6,   # Optimized for RTX 4080
                "1280x704": 6,
                "1920x1080": 10,
                "1920x1088": 10,
                "2560x1440": 14  # High-res option for RTX 4080
            }'''
    
    if re.search(vram_estimates_pattern, content):
        content = re.sub(vram_estimates_pattern, new_vram_estimates, content)
        print("✅ Updated VRAM estimates for RTX 4080")
    
    # Patch 2: Update max VRAM usage check
    max_vram_pattern = r'max_vram_usage_gb["\']:\s*\d+'
    content = re.sub(max_vram_pattern, '"max_vram_usage_gb": 14', content)
    
    # Patch 3: Update VRAM warnings
    vram_warning_pattern = r'estimated_vram\s*>\s*\d+'
    content = re.sub(vram_warning_pattern, 'estimated_vram > 14', content)
    
    # Patch 4: Add RTX 4080 specific optimizations
    optimization_section = '''
                # RTX 4080 Optimizations
                "rtx4080_optimizations": {
                    "enable_bf16": True,
                    "disable_cpu_offload": True,
                    "vae_tile_size": 512,
                    "max_concurrent_generations": 1,
                    "enable_attention_slicing": True
                },'''
    
    # Insert after the optimization config
    config_pattern = r'("optimization":\s*\{[^}]+\})'
    if re.search(config_pattern, content):
        content = re.sub(config_pattern, r'\1,' + optimization_section, content)
    
    # Write the patched file
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ UI VRAM validation patched for RTX 4080")
    return True

def patch_event_handlers():
    """Patch event handlers for better VRAM estimates"""
    
    handlers_path = Path("ui_event_handlers_enhanced.py")
    if not handlers_path.exists():
        print("⚠️ ui_event_handlers_enhanced.py not found, skipping")
        return False
    
    print("🔧 Patching event handlers for RTX 4080...")
    
    with open(handlers_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup
    backup_path = Path("ui_event_handlers_enhanced_backup_rtx4080.py")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Update base VRAM estimates
    base_vram_pattern = r'base_vram\s*=\s*\{[^}]+\}'
    new_base_vram = '''base_vram = {
                "1280x720": 6,   # RTX 4080 optimized
                "1280x704": 6,
                "1920x1080": 10,
                "1920x1088": 10,
                "2560x1440": 14
            }'''
    
    content = re.sub(base_vram_pattern, new_base_vram, content)
    
    # Update VRAM per second multiplier for RTX 4080
    vram_per_second_pattern = r'vram_per_second\s*=\s*[^*]+\*\s*[\d.]+\s*#'
    new_vram_per_second = 'vram_per_second = base_vram.get(resolution, 6) * 0.15  #'
    content = re.sub(vram_per_second_pattern, new_vram_per_second, content)
    
    # Update duration warnings for RTX 4080
    duration_warning_pattern = r'duration\s*>\s*6:'
    content = re.sub(duration_warning_pattern, 'duration > 10:', content)
    
    with open(handlers_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Event handlers patched for RTX 4080")
    return True

def create_vram_detection_override():
    """Create a VRAM detection override for the frontend"""
    
    override_script = '''#!/usr/bin/env python3
"""
VRAM Detection Override for RTX 4080
Forces proper VRAM detection in the frontend
"""

import torch
import json
import os

def get_gpu_info():
    """Get actual GPU information"""
    if not torch.cuda.is_available():
        return None
    
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "vram_gb": props.total_memory / (1024**3),
        "compute_capability": f"{props.major}.{props.minor}"
    }

def create_vram_override():
    """Create VRAM override configuration"""
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        print("❌ No CUDA GPU detected")
        return False
    
    # RTX 4080 specific configuration
    if "RTX 4080" in gpu_info["name"]:
        vram_config = {
            "gpu_detected": True,
            "gpu_name": gpu_info["name"],
            "total_vram_gb": gpu_info["vram_gb"],
            "usable_vram_gb": gpu_info["vram_gb"] - 2,  # 2GB buffer
            "optimizations": {
                "quantization": "bf16",
                "cpu_offload": False,
                "vae_tiling": True,
                "attention_slicing": True,
                "max_resolution": "2560x1440",
                "max_duration": 10
            },
            "vram_estimates": {
                "1280x720": {"base": 6, "per_second": 0.9},
                "1280x704": {"base": 6, "per_second": 0.9},
                "1920x1080": {"base": 10, "per_second": 1.5},
                "1920x1088": {"base": 10, "per_second": 1.5},
                "2560x1440": {"base": 14, "per_second": 2.1}
            }
        }
    else:
        # Generic high-VRAM GPU
        vram_config = {
            "gpu_detected": True,
            "gpu_name": gpu_info["name"],
            "total_vram_gb": gpu_info["vram_gb"],
            "usable_vram_gb": max(4, gpu_info["vram_gb"] - 1),
            "optimizations": {
                "quantization": "bf16" if gpu_info["vram_gb"] >= 12 else "int8",
                "cpu_offload": gpu_info["vram_gb"] < 12,
                "vae_tiling": True,
                "attention_slicing": True,
                "max_resolution": "1920x1080" if gpu_info["vram_gb"] >= 12 else "1280x720",
                "max_duration": 8 if gpu_info["vram_gb"] >= 12 else 6
            }
        }
    
    # Save configuration
    with open("gpu_override.json", "w") as f:
        json.dump(vram_config, f, indent=2)
    
    print(f"✅ GPU override created for {gpu_info['name']} ({gpu_info['vram_gb']:.1f}GB)")
    return True

if __name__ == "__main__":
    create_vram_override()
'''
    
    with open("vram_detection_override.py", 'w', encoding='utf-8') as f:
        f.write(override_script)
    
    print("✅ VRAM detection override created")

def main():
    """Main optimization function"""
    print("🚀 RTX 4080 FRONTEND VRAM OPTIMIZER")
    print("=" * 50)
    
    success_count = 0
    
    # Patch UI VRAM validation
    if patch_ui_vram_validation():
        success_count += 1
    
    # Patch event handlers
    if patch_event_handlers():
        success_count += 1
    
    # Create VRAM detection override
    create_vram_detection_override()
    success_count += 1
    
    print(f"\n✅ Frontend optimization complete! ({success_count}/3 components updated)")
    print("\n📋 Next Steps:")
    print("1. Run: python vram_detection_override.py")
    print("2. Restart the frontend server")
    print("3. Test VRAM validation with different resolutions")
    
    return success_count > 0

if __name__ == "__main__":
    main()
