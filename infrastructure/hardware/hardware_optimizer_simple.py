"""
Hardware Optimizer for WAN22 System Optimization - RTX 4080 Implementation
"""

import os
import platform
import psutil
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class HardwareProfile:
    """Hardware profile containing system specifications"""
    cpu_model: str
    cpu_cores: int
    total_memory_gb: int
    gpu_model: str
    vram_gb: int
    cuda_version: str = "Unknown"
    driver_version: str = "Unknown"
    is_rtx_4080: bool = False
    is_threadripper_pro: bool = False

@dataclass
class OptimalSettings:
    """Optimal settings for hardware configuration"""
    tile_size: Tuple[int, int]
    batch_size: int
    enable_cpu_offload: bool
    enable_tensor_cores: bool
    memory_fraction: float
    num_threads: int
    enable_xformers: bool
    vae_tile_size: Tuple[int, int]
    text_encoder_offload: bool
    vae_offload: bool
    use_fp16: bool
    use_bf16: bool
    gradient_checkpointing: bool

@dataclass
class OptimizationResult:
    """Result of hardware optimization"""
    success: bool
    optimizations_applied: List[str]
    performance_improvement: float
    memory_savings: int
    warnings: List[str]
    errors: List[str]
    settings: Optional[OptimalSettings] = None

class HardwareOptimizer:
    """Hardware-specific optimizer for WAN22 system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize hardware optimizer"""
        self.config_path = config_path or "config.json"
        self.logger = logging.getLogger(__name__)
        self.hardware_profile: Optional[HardwareProfile] = None
        self.optimal_settings: Optional[OptimalSettings] = None
    
    def generate_rtx_4080_settings(self, profile: HardwareProfile) -> OptimalSettings:
        """Generate optimal settings for RTX 4080"""
        self.logger.info("Generating RTX 4080 specific optimizations")
        
        # RTX 4080 specific optimizations
        settings = OptimalSettings(
            # Optimal tile size for RTX 4080 (256x256 for VAE as specified)
            tile_size=(512, 512),  # General tile size
            vae_tile_size=(256, 256),  # VAE specific tile size
            
            # Batch size optimization for 16GB VRAM
            batch_size=2 if profile.vram_gb >= 16 else 1,
            
            # Enable CPU offloading for text encoder and VAE
            enable_cpu_offload=True,
            text_encoder_offload=True,
            vae_offload=True,
            
            # Enable tensor cores for mixed precision
            enable_tensor_cores=True,
            use_fp16=True,
            use_bf16=True,  # RTX 4080 supports BF16
            
            # Memory optimization
            memory_fraction=0.9,  # Use 90% of VRAM
            gradient_checkpointing=True,
            
            # Threading optimization
            num_threads=min(profile.cpu_cores, 8),  # Optimal for most workloads
            
            # Enable xFormers for memory efficiency
            enable_xformers=True
        )
        
        return settings
    
    def apply_rtx_4080_optimizations(self, profile: HardwareProfile) -> OptimizationResult:
        """Apply RTX 4080 specific optimizations"""
        optimizations_applied = []
        warnings = []
        errors = []
        
        try:
            # Generate optimal settings
            settings = self.generate_rtx_4080_settings(profile)
            self.optimal_settings = settings
            
            # Apply PyTorch optimizations
            if TORCH_AVAILABLE:
                # Enable tensor cores
                if settings.enable_tensor_cores:
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    optimizations_applied.append("Enabled Tensor Cores (TF32)")
                
                # Set memory fraction
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(settings.memory_fraction)
                    optimizations_applied.append(f"Set CUDA memory fraction to {settings.memory_fraction}")
                
                # Apply threading optimizations
                if settings.num_threads:
                    torch.set_num_threads(settings.num_threads)
                    optimizations_applied.append(f"Set PyTorch threads to {settings.num_threads}")
            
            # Environment variables for optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            optimizations_applied.append("Set CUDA memory allocator optimization")
            
            # Enable mixed precision training
            if settings.use_fp16 or settings.use_bf16:
                os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
                optimizations_applied.append("Enabled cuDNN v8 API for mixed precision")
            
            self.logger.info(f"Applied {len(optimizations_applied)} RTX 4080 optimizations")
            
            return OptimizationResult(
                success=True,
                optimizations_applied=optimizations_applied,
                performance_improvement=0.0,
                memory_savings=0,
                warnings=warnings,
                errors=errors,
                settings=settings
            )
            
        except Exception as e:
            error_msg = f"Failed to apply RTX 4080 optimizations: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return OptimizationResult(
                success=False,
                optimizations_applied=optimizations_applied,
                performance_improvement=0.0,
                memory_savings=0,
                warnings=warnings,
                errors=errors
            )
    
    def configure_vae_tiling(self, tile_size: Tuple[int, int] = (256, 256)) -> bool:
        """Configure VAE tiling for memory efficiency"""
        try:
            self.vae_tile_size = tile_size
            self.logger.info(f"Configured VAE tiling: {tile_size}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to configure VAE tiling: {e}")
            return False
    
    def configure_cpu_offloading(self, text_encoder: bool = True, vae: bool = True) -> Dict[str, bool]:
        """Configure CPU offloading for components"""
        config = {
            'text_encoder_offload': text_encoder,
            'vae_offload': vae
        }
        
        try:
            self.cpu_offload_config = config
            self.logger.info(f"Configured CPU offloading: {config}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to configure CPU offloading: {e}")
            return {'text_encoder_offload': False, 'vae_offload': False}
    
    def get_memory_optimization_settings(self, available_vram_gb: int) -> Dict[str, Any]:
        """Get memory optimization settings based on available VRAM"""
        if available_vram_gb >= 16:  # RTX 4080
            return {
                'enable_attention_slicing': False,
                'enable_vae_slicing': False,
                'enable_cpu_offload': True,
                'batch_size': 2,
                'tile_size': (512, 512),
                'vae_tile_size': (256, 256)
            }
        elif available_vram_gb >= 12:
            return {
                'enable_attention_slicing': True,
                'enable_vae_slicing': True,
                'enable_cpu_offload': True,
                'batch_size': 1,
                'tile_size': (384, 384),
                'vae_tile_size': (192, 192)
            }
        else:
            return {
                'enable_attention_slicing': True,
                'enable_vae_slicing': True,
                'enable_cpu_offload': True,
                'batch_size': 1,
                'tile_size': (256, 256),
                'vae_tile_size': (128, 128)
            }