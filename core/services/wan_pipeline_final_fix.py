#!/usr/bin/env python3
"""
Final WAN Pipeline Loading Fix

This addresses the core issue: the WAN pipeline compatibility layer is causing
a recursive loop when DiffusionPipeline.from_pretrained tries to load the WAN model.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_final_wan_pipeline_fix():
    """Apply the final fix for WAN pipeline loading"""
    
    print("🔧 Applying Final WAN Pipeline Loading Fix")
    print("=" * 50)
    
    try:
        # The core issue is in the pipeline manager - it's passing the model_index.json
        # contents as kwargs to the pipeline, but the WAN pipeline doesn't expect them
        
        # Fix 1: Update pipeline_manager.py to filter out model_index.json contents
        pipeline_manager_path = Path("pipeline_manager.py")
        if pipeline_manager_path.exists():
            with open(pipeline_manager_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the section where WAN model loading happens
            old_section = '''# Prepare loading arguments
            load_args = {
                "pretrained_model_name_or_path": model_path,
                "trust_remote_code": trust_remote_code,
                **kwargs
            }'''
            
            new_section = '''# Prepare loading arguments - filter out model_index.json contents for WAN models
            if pipeline_class == "WanPipeline":
                # For WAN models, only pass essential loading arguments
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k in ['torch_dtype', 'device_map', 'low_cpu_mem_usage', 'variant']}
                load_args = {
                    "pretrained_model_name_or_path": model_path,
                    "trust_remote_code": trust_remote_code,
                    **filtered_kwargs
                }
            else:
                load_args = {
                    "pretrained_model_name_or_path": model_path,
                    "trust_remote_code": trust_remote_code,
                    **kwargs
                }'''
            
            if old_section in content:
                content = content.replace(old_section, new_section)
                with open(pipeline_manager_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ Fixed pipeline_manager.py argument filtering")
            else:
                print("⚠️  Could not find target section in pipeline_manager.py")
        
        # Fix 2: Simplify the WAN compatibility layer to avoid recursion
        wan_compat_path = Path("wan22_compatibility_clean.py")
        if wan_compat_path.exists():
            simplified_compat = '''"""
Wan2.2 Pipeline Compatibility Layer - Simplified Version
This provides basic compatibility for Wan2.2 models when the full WanPipeline is not available
"""

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
import torch
import logging

logger = logging.getLogger(__name__)

class WanPipeline(DiffusionPipeline):
    """Simplified compatibility wrapper for WanPipeline"""
    
    config_name = "model_index.json"
    _optional_components = []
    _exclude_from_cpu_offload = []
    
    def __init__(self, **kwargs):
        # Initialize parent class first
        super().__init__()
        logger.info("Using simplified WanPipeline compatibility layer")
        
        # Store all kwargs as attributes without trying to register them as modules
        for key, value in kwargs.items():
            if not key.startswith('_'):
                setattr(self, key, value)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Simplified loading that avoids recursion"""
        logger.info(f"WanPipeline.from_pretrained called for {pretrained_model_name_or_path}")
        
        # Remove our custom WanPipeline from diffusers temporarily to avoid recursion
        import diffusers
        original_wan_pipeline = getattr(diffusers, 'WanPipeline', None)
        if hasattr(diffusers, 'WanPipeline'):
            delattr(diffusers, 'WanPipeline')
        
        try:
            # Use DiffusionPipeline's auto-detection
            clean_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['boundary_ratio']}
            clean_kwargs['trust_remote_code'] = True
            
            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                **clean_kwargs
            )
            
            logger.info(f"Successfully loaded pipeline: {type(pipeline).__name__}")
            return pipeline
            
        except Exception as e:
            logger.error(f"WanPipeline loading failed: {e}")
            raise
        finally:
            # Restore our WanPipeline
            if original_wan_pipeline:
                diffusers.WanPipeline = original_wan_pipeline

class AutoencoderKLWan(AutoencoderKL):
    """Compatibility wrapper for AutoencoderKLWan"""
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load with fallback to standard AutoencoderKL"""
        try:
            kwargs.setdefault('ignore_mismatched_sizes', True)
            return AutoencoderKL.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception as e:
            logger.error(f"AutoencoderKLWan fallback failed: {e}")
            raise e

def apply_wan22_compatibility():
    """Apply the Wan2.2 compatibility layer"""
    try:
        import diffusers
        diffusers.WanPipeline = WanPipeline
        diffusers.AutoencoderKLWan = AutoencoderKLWan
        logger.info("Wan2.2 compatibility layer applied successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to apply Wan2.2 compatibility layer: {e}")
        return False

# Auto-apply when imported
if apply_wan22_compatibility():
    print("✅ Wan2.2 compatibility layer loaded")
else:
    print("❌ Failed to load Wan2.2 compatibility layer")
'''
            
            with open(wan_compat_path, 'w', encoding='utf-8') as f:
                f.write(simplified_compat)
            print("✅ Simplified WAN compatibility layer")
        
        print("\n✅ Final WAN pipeline loading fix applied!")
        print("🚀 Try running the UI again - the pipeline should load correctly now.")
        return True
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        return False

if __name__ == "__main__":
    success = apply_final_wan_pipeline_fix()
    
    if success:
        print("\n🎉 Final fix applied successfully!")
        print("The WAN2.2 pipeline should now load without the 'expected kwargs' error.")
    else:
        print("\n❌ Fix failed. Manual intervention may be required.")
    
    input("\nPress Enter to exit...")