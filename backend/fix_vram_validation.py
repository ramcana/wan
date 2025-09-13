#!/usr/bin/env python3
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
        vram_config = {
            "total_vram_gb": actual_vram,
            "usable_vram_gb": actual_vram - 2,
            "vram_estimates": {
                "1280x720": 6,
                "1280x704": 6,
                "1920x1080": 10,
                "1920x1088": 10
            },
            "duration_multiplier": 0.15,  # VRAM per additional second
            "max_safe_duration": 10,
            "warnings": {
                "high_vram_threshold": 12,
                "very_high_vram_threshold": 14
            }
        }
    else:
        # Fallback for other GPUs
        vram_config = {
            "total_vram_gb": actual_vram,
            "usable_vram_gb": max(0, actual_vram - 1),
            "vram_estimates": {
                "1280x720": 8,
                "1280x704": 8,
                "1920x1080": 12,
                "1920x1088": 12
            },
            "duration_multiplier": 0.2,
            "max_safe_duration": 6,
            "warnings": {
                "high_vram_threshold": actual_vram * 0.8,
                "very_high_vram_threshold": actual_vram * 0.9
            }
        }
    
    # Save VRAM config
    config_path = Path("frontend_vram_config.json")
    with open(config_path, 'w') as f:
        json.dump(vram_config, f, indent=2)
    
    print(f"VRAM config saved: {actual_vram:.1f}GB detected")
    return vram_config

if __name__ == "__main__":
    update_frontend_vram_config()
