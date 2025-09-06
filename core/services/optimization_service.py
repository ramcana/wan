"""
Core Optimization Service
Handles VRAM optimization, quantization, and performance tuning
Extracted from utils.py as part of functional organization
"""

import logging
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import psutil

# Import error handling system
from infrastructure.hardware.error_handler import (
    handle_error_with_recovery, 
    log_error_with_context
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VRAMOptimizer:
    """Handles VRAM optimization techniques for efficient model inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_config = config.get("optimization", {})
        self.max_vram_gb = self.optimization_config.get("max_vram_usage_gb", 12)
        
    @handle_error_with_recovery
    def apply_quantization(self, model, quantization_level: str = "bf16", timeout_seconds: int = 300, skip_large_components: bool = False):
        """Apply quantization to reduce model memory usage"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping quantization")
            return model
        
        if quantization_level == "none" or quantization_level is None:
            logger.info("Quantization disabled")
            return model
        
        logger.info(f"Applying {quantization_level} quantization (timeout: {timeout_seconds}s, skip_large: {skip_large_components})")
        
        import threading
        
        # Simple timeout tracking for logging purposes
        start_time = time.time()
        timeout_occurred = threading.Event()
        
        try:
            # Check VRAM before quantization
            initial_vram = self.get_vram_usage()
            
            if quantization_level == "fp16":
                model = model.half()
                logger.info("Applied fp16 quantization")
                
            elif quantization_level == "bf16":
                if hasattr(model, 'to'):
                    model = model.to(dtype=torch.bfloat16)
                else:
                    # For pipeline objects, convert components
                    components_converted = []
                    
                    # Standard diffusion components (safe to convert)
                    if hasattr(model, 'unet') and model.unet is not None:
                        model.unet = model.unet.to(dtype=torch.bfloat16)
                        components_converted.append('unet')
                    if hasattr(model, 'vae') and model.vae is not None:
                        model.vae = model.vae.to(dtype=torch.bfloat16)
                        components_converted.append('vae')
                    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                        model.text_encoder = model.text_encoder.to(dtype=torch.bfloat16)
                        components_converted.append('text_encoder')
                    
                    # WAN-specific components - skip if skip_large_components is True
                    if hasattr(model, 'transformer') and model.transformer is not None:
                        if skip_large_components:
                            logger.info("Skipping transformer quantization (skip_large_components=True)")
                        else:
                            logger.info("Converting transformer to bf16... (this may take several minutes)")
                            try:
                                model.transformer = model.transformer.to(dtype=torch.bfloat16)
                                components_converted.append('transformer')
                                logger.info("Transformer conversion completed")
                            except Exception as e:
                                logger.warning(f"Transformer quantization failed: {e}, skipping")
                    
                    if hasattr(model, 'transformer_2') and model.transformer_2 is not None:
                        if skip_large_components:
                            logger.info("Skipping transformer_2 quantization (skip_large_components=True)")
                        else:
                            logger.info("Converting transformer_2 to bf16... (this may take several minutes)")
                            try:
                                model.transformer_2 = model.transformer_2.to(dtype=torch.bfloat16)
                                components_converted.append('transformer_2')
                                logger.info("Transformer_2 conversion completed")
                            except Exception as e:
                                logger.warning(f"Transformer_2 quantization failed: {e}, skipping")
                    
                    logger.info(f"Applied bf16 quantization to components: {components_converted}")
                logger.info("Applied bf16 quantization")
                
            elif quantization_level == "int8":
                try:
                    import bitsandbytes as bnb
                    
                    # Apply int8 quantization using bitsandbytes
                    if hasattr(model, 'unet') and model.unet is not None:
                        model.unet = self._quantize_model_int8(model.unet)
                    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                        model.text_encoder = self._quantize_model_int8(model.text_encoder)
                    # Note: VAE typically not quantized to int8 as it can hurt quality significantly
                    
                    logger.info("Applied int8 quantization")
                    
                except ImportError:
                    logger.error("bitsandbytes not available for int8 quantization, falling back to bf16")
                    return self.apply_quantization(model, "bf16")
                except Exception as e:
                    log_error_with_context(e, "int8_quantization", {"quantization_level": quantization_level})
                    logger.error(f"Failed to apply int8 quantization: {e}, falling back to bf16")
                    return self.apply_quantization(model, "bf16")
            
            else:
                logger.warning(f"Unknown quantization level: {quantization_level}, using bf16")
                return self.apply_quantization(model, "bf16")
            
            # Log completion time
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Quantization completed in {total_time:.1f} seconds")
            
            # Check VRAM after quantization
            final_vram = self.get_vram_usage()
            vram_saved = initial_vram["used_mb"] - final_vram["used_mb"]
            if vram_saved > 0:
                logger.info(f"Quantization saved {vram_saved:.0f}MB VRAM")
            
            return model
            
        except torch.cuda.OutOfMemoryError as e:
            log_error_with_context(e, "quantization", {"quantization_level": quantization_level, "vram_info": initial_vram if 'initial_vram' in locals() else None})
            raise
        except Exception as e:
            log_error_with_context(e, "quantization", {"quantization_level": quantization_level})
            logger.error(f"Failed to apply quantization: {e}")
            return model
        finally:
            # Signal completion to timeout thread
            timeout_occurred.set()
    
    def _quantize_model_int8(self, model):
        """Apply int8 quantization to a model using bitsandbytes"""
        try:
            import bitsandbytes as bnb
            
            # Replace linear layers with int8 quantized versions
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Create int8 linear layer
                    int8_layer = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False
                    )
                    
                    # Copy weights and bias
                    int8_layer.weight.data = module.weight.data
                    if module.bias is not None:
                        int8_layer.bias.data = module.bias.data
                    
                    # Replace the module
                    parent = model
                    for attr in name.split('.')[:-1]:
                        parent = getattr(parent, attr)
                    setattr(parent, name.split('.')[-1], int8_layer)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to quantize model to int8: {e}")
            return model
    
    @handle_error_with_recovery
    def enable_cpu_offload(self, model, enable_sequential: bool = True):
        """Enable CPU offloading to reduce VRAM usage"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping CPU offload")
            return model
        
        try:
            # Check available system RAM
            ram_info = psutil.virtual_memory()
            if ram_info.percent > 80:  # More than 80% RAM used
                logger.warning(f"High RAM usage ({ram_info.percent:.1f}%), CPU offload may cause issues")
            
            if hasattr(model, 'enable_model_cpu_offload'):
                if enable_sequential:
                    # Sequential CPU offload - moves model components between GPU/CPU as needed
                    model.enable_sequential_cpu_offload()
                    logger.info("Enabled sequential CPU offload")
                else:
                    # Standard CPU offload
                    model.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload")
            else:
                logger.warning("Model does not support CPU offload")
            
            return model
            
        except Exception as e:
            log_error_with_context(e, "cpu_offload", {"enable_sequential": enable_sequential, "ram_percent": ram_info.percent if 'ram_info' in locals() else None})
            logger.error(f"Failed to enable CPU offload: {e}")
            return model
    
    def enable_vae_tiling(self, model, tile_size: int = 256):
        """Enable VAE tiling to reduce memory usage during encoding/decoding"""
        if not hasattr(model, 'vae') or model.vae is None:
            logger.warning("Model does not have VAE, skipping tiling")
            return model
        
        # Validate tile size
        min_size, max_size = self.optimization_config.get("vae_tile_size_range", [128, 512])
        tile_size = max(min_size, min(max_size, tile_size))
        
        try:
            if hasattr(model.vae, 'enable_tiling'):
                model.vae.enable_tiling()
                logger.info(f"Enabled VAE tiling with size {tile_size}")
            elif hasattr(model, 'enable_vae_tiling'):
                model.enable_vae_tiling()
                logger.info(f"Enabled VAE tiling with size {tile_size}")
            else:
                # Manual tiling implementation
                self._apply_manual_vae_tiling(model.vae, tile_size)
                logger.info(f"Applied manual VAE tiling with size {tile_size}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to enable VAE tiling: {e}")
            return model
    
    def _apply_manual_vae_tiling(self, vae, tile_size: int):
        """Apply manual VAE tiling by patching encode/decode methods"""
        original_encode = vae.encode
        original_decode = vae.decode
        
        def tiled_encode(x):
            return self._tiled_operation(x, original_encode, tile_size)
        
        def tiled_decode(x):
            return self._tiled_operation(x, original_decode, tile_size)
        
        # Patch the VAE methods
        vae.encode = tiled_encode
        vae.decode = tiled_decode
    
    def _tiled_operation(self, input_tensor, operation_func, tile_size: int):
        """Perform tiled operation to reduce memory usage"""
        try:
            # For small inputs, use original operation
            if input_tensor.shape[-1] <= tile_size and input_tensor.shape[-2] <= tile_size:
                return operation_func(input_tensor)
            
            # Split input into tiles
            batch_size, channels, height, width = input_tensor.shape
            
            # Calculate number of tiles
            tiles_h = (height + tile_size - 1) // tile_size
            tiles_w = (width + tile_size - 1) // tile_size
            
            # Process tiles
            output_tiles = []
            
            for i in range(tiles_h):
                row_tiles = []
                for j in range(tiles_w):
                    # Calculate tile boundaries
                    start_h = i * tile_size
                    end_h = min((i + 1) * tile_size, height)
                    start_w = j * tile_size
                    end_w = min((j + 1) * tile_size, width)
                    
                    # Extract tile
                    tile = input_tensor[:, :, start_h:end_h, start_w:end_w]
                    
                    # Process tile
                    processed_tile = operation_func(tile)
                    row_tiles.append(processed_tile)
                    
                    # Clear GPU cache after each tile
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Concatenate row tiles
                row_output = torch.cat(row_tiles, dim=-1)
                output_tiles.append(row_output)
            
            # Concatenate all tiles
            final_output = torch.cat(output_tiles, dim=-2)
            
            return final_output
            
        except Exception as e:
            logger.error(f"Tiled operation failed: {e}, falling back to original operation")
            return operation_func(input_tensor)
    
    def apply_advanced_optimizations(self, model, optimization_config: Dict[str, Any]):
        """Apply advanced VRAM optimizations based on configuration"""
        try:
            # Enable attention slicing for memory efficiency
            if hasattr(model, 'enable_attention_slicing'):
                slice_size = optimization_config.get('attention_slice_size', 'auto')
                model.enable_attention_slicing(slice_size)
                logger.info(f"Enabled attention slicing with size: {slice_size}")
            
            # Enable memory efficient attention
            if hasattr(model, 'enable_memory_efficient_attention'):
                model.enable_memory_efficient_attention()
                logger.info("Enabled memory efficient attention")
            
            # Enable xformers if available
            if optimization_config.get('enable_xformers', True):
                try:
                    model.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.debug(f"xformers not available: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply advanced optimizations: {e}")
            return model
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage information"""
        if not torch.cuda.is_available():
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "percent": 0}
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            
            total_mb = total_memory / (1024 * 1024)
            used_mb = allocated_memory / (1024 * 1024)
            free_mb = total_mb - used_mb
            percent = (used_mb / total_mb) * 100
            
            return {
                "total_mb": total_mb,
                "used_mb": used_mb, 
                "free_mb": free_mb,
                "percent": percent
            }
        except Exception:
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "percent": 0}
    
    def optimize_model_for_inference(self, model, target_vram_gb: Optional[float] = None):
        """Apply comprehensive optimization for inference"""
        if target_vram_gb is None:
            target_vram_gb = self.max_vram_gb
        
        logger.info(f"Optimizing model for {target_vram_gb}GB VRAM target")
        
        # Get initial VRAM usage
        initial_vram = self.get_vram_usage()
        
        # Apply optimizations in order of effectiveness
        try:
            # 1. Apply quantization (most effective)
            model = self.apply_quantization(model, "bf16")
            
            # 2. Enable advanced optimizations
            model = self.apply_advanced_optimizations(model, self.optimization_config)
            
            # 3. Enable VAE tiling
            model = self.enable_vae_tiling(model)
            
            # 4. Check if CPU offload is needed
            current_vram = self.get_vram_usage()
            if current_vram["used_mb"] > (target_vram_gb * 1024):
                logger.info("VRAM usage still high, enabling CPU offload")
                model = self.enable_cpu_offload(model)
            
            # Final VRAM check
            final_vram = self.get_vram_usage()
            vram_saved = initial_vram["used_mb"] - final_vram["used_mb"]
            
            logger.info(f"Optimization complete. VRAM saved: {vram_saved:.0f}MB")
            logger.info(f"Final VRAM usage: {final_vram['used_mb']:.0f}MB / {final_vram['total_mb']:.0f}MB ({final_vram['percent']:.1f}%)")
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model

# Global optimizer instance
_vram_optimizer = None

def get_vram_optimizer(config: Dict[str, Any] = None) -> VRAMOptimizer:
    """Get the global VRAM optimizer instance"""
    global _vram_optimizer
    if _vram_optimizer is None or config is not None:
        if config is None:
            # Load default config
            import json
            try:
                with open("config.json", 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                config = {"optimization": {"max_vram_usage_gb": 12}}
        _vram_optimizer = VRAMOptimizer(config)
    return _vram_optimizer

# Convenience functions
def optimize_model(model, config: Dict[str, Any] = None):
    """Optimize a model for inference"""
    optimizer = get_vram_optimizer(config)
    return optimizer.optimize_model_for_inference(model)

def apply_quantization(model, level: str = "bf16", config: Dict[str, Any] = None):
    """Apply quantization to a model"""
    optimizer = get_vram_optimizer(config)
    return optimizer.apply_quantization(model, level)

def enable_cpu_offload(model, config: Dict[str, Any] = None):
    """Enable CPU offload for a model"""
    optimizer = get_vram_optimizer(config)
    return optimizer.enable_cpu_offload(model)
