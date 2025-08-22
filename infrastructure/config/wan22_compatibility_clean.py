"""
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
    
    def __init__(self, 
                 transformer=None,
                 transformer_2=None, 
                 scheduler=None,
                 vae=None,
                 text_encoder=None,
                 tokenizer=None,
                 safety_checker=None,
                 feature_extractor=None,
                 **kwargs):
        # Initialize parent class first
        super().__init__()
        logger.info("Using simplified WanPipeline compatibility layer")
        
        # Register components that are provided
        if transformer is not None:
            self.register_modules(transformer=transformer)
        if transformer_2 is not None:
            self.register_modules(transformer_2=transformer_2)
        if scheduler is not None:
            self.register_modules(scheduler=scheduler)
        if vae is not None:
            self.register_modules(vae=vae)
        if text_encoder is not None:
            self.register_modules(text_encoder=text_encoder)
        if tokenizer is not None:
            self.register_modules(tokenizer=tokenizer)
        if safety_checker is not None:
            self.register_modules(safety_checker=safety_checker)
        if feature_extractor is not None:
            self.register_modules(feature_extractor=feature_extractor)
        
        # Store other kwargs as attributes
        for key, value in kwargs.items():
            if not key.startswith('_') and not hasattr(self, key):
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
            # Use DiffusionPipeline's auto-detection with trust_remote_code
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
        # Only apply if WanPipeline doesn't already exist
        if not hasattr(diffusers, 'WanPipeline'):
            diffusers.WanPipeline = WanPipeline
        if not hasattr(diffusers, 'AutoencoderKLWan'):
            diffusers.AutoencoderKLWan = AutoencoderKLWan
        logger.info("Wan2.2 compatibility layer applied successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to apply Wan2.2 compatibility layer: {e}")
        return False

def remove_wan22_compatibility():
    """Remove the Wan2.2 compatibility layer to allow remote code loading"""
    try:
        import diffusers
        if hasattr(diffusers, 'WanPipeline') and diffusers.WanPipeline == WanPipeline:
            delattr(diffusers, 'WanPipeline')
        if hasattr(diffusers, 'AutoencoderKLWan') and diffusers.AutoencoderKLWan == AutoencoderKLWan:
            delattr(diffusers, 'AutoencoderKLWan')
        logger.info("Wan2.2 compatibility layer removed for remote code loading")
        return True
    except Exception as e:
        logger.error(f"Failed to remove Wan2.2 compatibility layer: {e}")
        return False

# Auto-apply when imported
if apply_wan22_compatibility():
    print("✅ Wan2.2 compatibility layer loaded")
else:
    print("❌ Failed to load Wan2.2 compatibility layer")
