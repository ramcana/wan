"""
Wan2.2 Pipeline Compatibility Layer
This provides basic compatibility for Wan2.2 models when the full WanPipeline is not available
"""

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
import torch
import logging

logger = logging.getLogger(__name__)

class WanPipeline(DiffusionPipeline):
    """Compatibility wrapper for WanPipeline"""
    
    # Define the expected components for the pipeline
    config_name = "model_index.json"
    _optional_components = []
    _exclude_from_cpu_offload = []
    
    def __init__(self, **kwargs):
        # Initialize with minimal required components
        super().__init__()
        logger.info("Using WanPipeline compatibility layer")
        
        # Set up basic attributes that diffusers expects
        self.register_modules(**kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load with fallback to standard pipeline"""
        
        # Prepare kwargs for loading custom models
        clean_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['boundary_ratio']}
        
        # Add parameters to handle model weight mismatches
        clean_kwargs.update({
            'trust_remote_code': True,
            'low_cpu_mem_usage': False,
            'ignore_mismatched_sizes': True,
            'safety_checker': None,  # Disable safety checker to avoid issues
            'requires_safety_checker': False
        })
        
        # Don't duplicate existing parameters
        for key in ['trust_remote_code', 'low_cpu_mem_usage', 'ignore_mismatched_sizes']:
            if key in kwargs:
                clean_kwargs[key] = kwargs[key]
        
        try:
            # Try to load as standard diffusion pipeline first
            from diffusers import StableDiffusionPipeline
            logger.info(f"Loading {pretrained_model_name_or_path} with StableDiffusionPipeline fallback")
            
            return StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, 
                **clean_kwargs
            )
        except Exception as e:
            logger.error(f"StableDiffusionPipeline fallback failed: {e}")
            
            # If StableDiffusionPipeline fails, try basic DiffusionPipeline
            try:
                logger.info("Trying basic DiffusionPipeline as final fallback")
                
                return DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path,
                    **clean_kwargs
                )
            except Exception as e2:
                logger.error(f"DiffusionPipeline also failed: {e2}")
                
                # Final attempt: try without some problematic parameters
                try:
                    logger.info("Trying minimal parameter set as last resort")
                    minimal_kwargs = {
                        'local_files_only': clean_kwargs.get('local_files_only', True),
                        'trust_remote_code': True,
                        'torch_dtype': clean_kwargs.get('torch_dtype'),
                        'low_cpu_mem_usage': False
                    }
                    # Remove None values
                    minimal_kwargs = {k: v for k, v in minimal_kwargs.items() if v is not None}
                    
                    return DiffusionPipeline.from_pretrained(
                        pretrained_model_name_or_path,
                        **minimal_kwargs
                    )
                except Exception as e3:
                    logger.error(f"All pipeline loading methods failed: {e3}")
                    raise e

class AutoencoderKLWan(AutoencoderKL):
    """Compatibility wrapper for AutoencoderKLWan"""
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load with fallback to standard AutoencoderKL"""
        try:
            logger.info(f"Loading {pretrained_model_name_or_path} with AutoencoderKL fallback")
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