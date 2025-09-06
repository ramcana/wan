#!/usr/bin/env python3
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
