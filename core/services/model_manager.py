"""
Core Model Manager Service
Handles model loading, caching, and optimization with compatibility detection
Extracted from utils.py as part of functional organization
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModel
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
import psutil
from PIL import Image

# Import error handling system
from infrastructure.hardware.error_handler import (
    handle_error_with_recovery, 
    log_error_with_context, 
    ErrorWithRecoveryInfo,
    get_error_recovery_manager,
    create_error_info,
    ErrorCategory
)

# Import compatibility detection system
from infrastructure.hardware.architecture_detector import ArchitectureDetector, ArchitectureType
from backend.core.services.wan_pipeline_loader import WanPipelineLoader, GenerationConfig
from backend.core.services.optimization_manager import OptimizationManager

# Import WAN model configuration system
try:
    from backend.core.models.wan_models.wan_model_config import (
        get_wan_model_config, 
        get_wan_model_info,
        WAN_MODEL_CONFIGS,
        WANModelConfig,
        validate_wan_model_requirements
    )
    WAN_CONFIG_AVAILABLE = True
except ImportError:
    WAN_CONFIG_AVAILABLE = False
    logger.warning("WAN model configuration system not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a loaded model"""
    model_type: str
    model_path: str
    model_id: str
    loaded_at: datetime
    memory_usage_mb: float
    quantization_level: Optional[str] = None
    is_offloaded: bool = False

class ModelCache:
    """Manages model caching and metadata"""
    
    def __init__(self, cache_dir: str = "models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_info_file = self.cache_dir / "cache_info.json"
        self.cache_info = self._load_cache_info()
    
    def _load_cache_info(self) -> Dict[str, Any]:
        """Load cache information from disk"""
        if self.cache_info_file.exists():
            try:
                with open(self.cache_info_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Failed to load cache info, starting fresh")
        return {}
    
    def _save_cache_info(self):
        """Save cache information to disk"""
        try:
            with open(self.cache_info_file, 'w') as f:
                json.dump(self.cache_info, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save cache info: {e}")
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is cached"""
        model_path = self.get_model_path(model_id)
        return model_path.exists() and any(model_path.iterdir())
    
    def get_model_path(self, model_id: str) -> Path:
        """Get the local path for a model"""
        # Sanitize model ID for filesystem
        safe_id = model_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / safe_id
    
    def validate_cached_model(self, model_id: str) -> bool:
        """Validate that a cached model is complete and usable"""
        model_path = self.get_model_path(model_id)
        
        if not model_path.exists():
            return False
        
        # Check for essential files
        essential_files = ["config.json"]
        for file_name in essential_files:
            if not (model_path / file_name).exists():
                logger.warning(f"Missing essential file {file_name} for model {model_id}")
                return False
        
        return True
    
    def update_cache_info(self, model_id: str, info: Dict[str, Any]):
        """Update cache information for a model"""
        self.cache_info[model_id] = info
        self._save_cache_info()
    
    def get_cache_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get cache information for a model"""
        return self.cache_info.get(model_id)

class ModelManager:
    """Manages model loading, caching, and optimization with compatibility detection"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.cache = ModelCache(self.config["directories"]["models_directory"])
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        
        # Initialize compatibility detection system
        self.architecture_detector = ArchitectureDetector()
        self.wan_pipeline_loader = WanPipelineLoader()
        self.optimization_manager = OptimizationManager()
        
        # Model ID mappings - Updated to use actual WAN implementations
        self.model_mappings = {
            "t2v-A14B": "wan_implementation:t2v-A14B",
            "i2v-A14B": "wan_implementation:i2v-A14B", 
            "ti2v-5B": "wan_implementation:ti2v-5B",
            # Legacy placeholder mappings for backward compatibility
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan_implementation:t2v-A14B",
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "wan_implementation:i2v-A14B",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "wan_implementation:ti2v-5B",
            "Wan2.2-T2V-A14B": "wan_implementation:t2v-A14B",
            "Wan2.2-I2V-A14B": "wan_implementation:i2v-A14B",
            "Wan2.2-TI2V-5B": "wan_implementation:ti2v-5B"
        }
        
        # Compatibility status cache
        self._compatibility_cache: Dict[str, Dict[str, Any]] = {}
    
    @handle_error_with_recovery
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log_error_with_context(e, "config_loading", {"config_path": config_path})
            # Return default config as fallback
            logger.warning("Using default configuration due to config loading error")
            return {
                "directories": {"models_directory": "models", "outputs_directory": "outputs", "loras_directory": "loras"},
                "optimization": {"max_vram_usage_gb": 12}
            }
    
    def get_model_id(self, model_type: str) -> str:
        """Get the Hugging Face model ID for a model type"""
        if model_type in self.model_mappings:
            return self.model_mappings[model_type]
        return model_type  # Assume it's already a full model ID
    
    def detect_model_type(self, model_id: str) -> str:
        """Detect the type of model from its ID or config"""
        model_id_lower = model_id.lower()
        
        if "t2v" in model_id_lower:
            return "text-to-video"
        elif "i2v" in model_id_lower:
            return "image-to-video"
        elif "ti2v" in model_id_lower:
            return "text-image-to-video"
        else:
            # Try to detect from config if model is cached
            if self.cache.is_model_cached(model_id):
                try:
                    config_path = self.cache.get_model_path(model_id) / "config.json"
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Look for model type indicators in config
                    if "text_to_video" in str(config).lower():
                        return "text-to-video"
                    elif "image_to_video" in str(config).lower():
                        return "image-to-video"
                    
                except Exception as e:
                    logger.warning(f"Failed to detect model type from config: {e}")
            
            return "unknown"
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive status information for a model"""
        full_model_id = self.get_model_id(model_id)
        
        status = {
            "model_id": full_model_id,
            "is_cached": self.cache.is_model_cached(full_model_id),
            "is_loaded": full_model_id in self.loaded_models,
            "is_valid": False,
            "cache_info": None,
            "model_info": None,
            "size_mb": 0.0,
            "compatibility_status": None,
            "optimization_recommendations": [],
            "is_wan_model": self.is_wan_model(model_id),
            "wan_capabilities": None,
            "wan_validation": None,
            "hardware_compatibility": None,
            "performance_profile": None
        }
        
        if status["is_cached"]:
            status["is_valid"] = self.cache.validate_cached_model(full_model_id)
            status["cache_info"] = self.cache.get_cache_info(full_model_id)
            status["size_mb"] = self._get_model_size_mb(full_model_id)
        
        if status["is_loaded"]:
            status["model_info"] = self.model_info[full_model_id]
        
        # Add WAN model specific information
        if status["is_wan_model"]:
            # Extract model type from full_model_id
            if full_model_id.startswith("wan_implementation:"):
                wan_model_type = full_model_id.split(":", 1)[1]
            else:
                wan_model_type = model_id
            
            # Get WAN model capabilities
            status["wan_capabilities"] = self.get_wan_model_capabilities(wan_model_type)
            
            # Validate WAN model configuration
            is_valid, errors = self.validate_wan_model_configuration(wan_model_type)
            status["wan_validation"] = {
                "is_valid": is_valid,
                "errors": errors
            }
            
            # Get hardware compatibility assessment
            status["hardware_compatibility"] = self.assess_hardware_compatibility(wan_model_type)
            
            # Get performance profile recommendations
            status["performance_profile"] = self.get_performance_profile(wan_model_type)
            
            # Update overall validity based on WAN validation
            if not is_valid:
                status["is_valid"] = False
        
        return status
    
    def _get_model_size_mb(self, model_id: str) -> float:
        """Get the disk size of a cached model in MB"""
        try:
            model_path = self.cache.get_model_path(model_id)
            total_size = 0
            
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Failed to calculate model size: {e}")
            return 0.0
    
    def validate_wan_model_configuration(self, model_type: str) -> Tuple[bool, List[str]]:
        """
        Validate WAN model configuration and parameters
        
        Args:
            model_type: Model type to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            if not WAN_CONFIG_AVAILABLE:
                return False, ["WAN model configuration system not available"]
            
            # Get WAN model configuration
            wan_config = get_wan_model_config(model_type)
            if not wan_config:
                return False, [f"No WAN configuration found for model type: {model_type}"]
            
            # Validate configuration
            is_valid, errors = wan_config.validate()
            if not is_valid:
                return False, errors
            
            # Validate hardware requirements
            hardware_errors = []
            if hasattr(self, 'hardware_profile') and self.hardware_profile:
                available_vram = self.hardware_profile.available_vram_gb
                required_vram = wan_config.optimization.min_vram_gb
                
                if available_vram < required_vram:
                    hardware_errors.append(
                        f"Insufficient VRAM: {available_vram:.1f}GB available, "
                        f"{required_vram:.1f}GB required"
                    )
            
            return len(hardware_errors) == 0, hardware_errors
            
        except Exception as e:
            logger.error(f"Error validating WAN model configuration: {e}")
            return False, [f"Configuration validation error: {str(e)}"]
    
    def get_wan_model_capabilities(self, model_type: str) -> Dict[str, Any]:
        """
        Get WAN model capabilities and requirements
        
        Args:
            model_type: Model type to get capabilities for
            
        Returns:
            Dictionary with model capabilities and requirements
        """
        try:
            if not WAN_CONFIG_AVAILABLE:
                return {
                    "error": "WAN model configuration system not available",
                    "capabilities": {},
                    "requirements": {}
                }
            
            # Get WAN model configuration
            wan_config = get_wan_model_config(model_type)
            if not wan_config:
                return {
                    "error": f"No WAN configuration found for model type: {model_type}",
                    "capabilities": {},
                    "requirements": {}
                }
            
            # Extract capabilities
            capabilities = {
                "model_type": wan_config.model_type,
                "display_name": wan_config.display_name,
                "description": wan_config.description,
                "architecture_type": wan_config.architecture.architecture_type.value if hasattr(wan_config.architecture.architecture_type, 'value') else str(wan_config.architecture.architecture_type),
                "max_resolution": wan_config.architecture.max_resolution,
                "min_resolution": wan_config.architecture.min_resolution,
                "max_frames": wan_config.architecture.max_frames,
                "supports_variable_length": wan_config.architecture.supports_variable_length,
                "supports_attention_slicing": wan_config.architecture.supports_attention_slicing,
                "supports_gradient_checkpointing": wan_config.architecture.supports_gradient_checkpointing,
                **wan_config.capabilities
            }
            
            # Extract requirements
            supported_precisions = []
            default_precision = wan_config.optimization.default_precision
            if hasattr(default_precision, 'value'):
                supported_precisions.append(default_precision.value)
            else:
                supported_precisions.append(str(default_precision))
            
            # Add other supported precisions
            if wan_config.optimization.supports_fp16:
                supported_precisions.append("fp16")
            if wan_config.optimization.supports_bf16:
                supported_precisions.append("bf16")
            if wan_config.optimization.supports_int8:
                supported_precisions.append("int8")
            
            requirements = {
                "min_vram_gb": wan_config.optimization.min_vram_gb,
                "estimated_vram_gb": wan_config.optimization.vram_estimate_gb,
                "supported_precisions": list(set(supported_precisions)),  # Remove duplicates
                "supports_cpu_offload": wan_config.optimization.cpu_offload_enabled,
                "supports_memory_efficient_attention": wan_config.optimization.memory_efficient_attention,
                "supports_torch_compile": wan_config.optimization.supports_torch_compile,
                "supports_xformers": wan_config.optimization.supports_xformers
            }
            
            # Get hardware profiles
            hardware_profiles = {}
            for profile_name, profile in wan_config.hardware_profiles.items():
                precision_value = profile.precision.value if hasattr(profile.precision, 'value') else str(profile.precision)
                hardware_profiles[profile_name] = {
                    "target_gpu": profile.target_gpu,
                    "vram_requirement_gb": profile.vram_requirement_gb,
                    "batch_size": profile.batch_size,
                    "precision": precision_value,
                    "optimizations": profile.optimization_notes
                }
            
            return {
                "capabilities": capabilities,
                "requirements": requirements,
                "hardware_profiles": hardware_profiles,
                "is_valid": True
            }
            
        except Exception as e:
            logger.error(f"Error getting WAN model capabilities: {e}")
            return {
                "error": f"Failed to get capabilities: {str(e)}",
                "capabilities": {},
                "requirements": {}
            }
    
    def get_wan_model_recommendations(self, available_vram_gb: float) -> Dict[str, Any]:
        """
        Get WAN model recommendations based on available hardware
        
        Args:
            available_vram_gb: Available VRAM in GB
            
        Returns:
            Dictionary with model recommendations
        """
        try:
            if not WAN_CONFIG_AVAILABLE:
                return {"error": "WAN model configuration system not available"}
            
            recommendations = {
                "recommended_models": [],
                "compatible_models": [],
                "incompatible_models": [],
                "optimization_suggestions": []
            }
            
            for model_type in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]:
                wan_config = get_wan_model_config(model_type)
                if not wan_config:
                    continue
                
                min_vram = wan_config.optimization.min_vram_gb
                estimated_vram = wan_config.optimization.vram_estimate_gb
                
                model_info = {
                    "model_type": model_type,
                    "display_name": wan_config.display_name,
                    "min_vram_gb": min_vram,
                    "estimated_vram_gb": estimated_vram,
                    "description": wan_config.description
                }
                
                if available_vram_gb >= estimated_vram:
                    recommendations["recommended_models"].append(model_info)
                elif available_vram_gb >= min_vram:
                    model_info["optimization_required"] = True
                    recommendations["compatible_models"].append(model_info)
                else:
                    model_info["vram_deficit_gb"] = min_vram - available_vram_gb
                    recommendations["incompatible_models"].append(model_info)
            
            # Add optimization suggestions
            if available_vram_gb < 8.0:
                recommendations["optimization_suggestions"].extend([
                    "Enable CPU offloading to reduce VRAM usage",
                    "Use attention slicing for memory efficiency",
                    "Consider using FP16 or INT8 quantization",
                    "Enable VAE tiling with smaller tile sizes"
                ])
            elif available_vram_gb < 12.0:
                recommendations["optimization_suggestions"].extend([
                    "Enable memory efficient attention",
                    "Consider CPU offloading for larger models",
                    "Use FP16 precision for optimal performance"
                ])
            else:
                recommendations["optimization_suggestions"].extend([
                    "All models should run optimally",
                    "Consider enabling torch.compile for faster inference",
                    "Use BF16 precision if supported by hardware"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting WAN model recommendations: {e}")
            return {"error": f"Failed to get recommendations: {str(e)}"}
    
    def assess_hardware_compatibility(self, model_type: str) -> Dict[str, Any]:
        """
        Assess hardware compatibility for a WAN model
        
        Args:
            model_type: WAN model type to assess
            
        Returns:
            Dictionary with compatibility assessment
        """
        try:
            if not WAN_CONFIG_AVAILABLE:
                return {"error": "WAN model configuration system not available"}
            
            wan_config = get_wan_model_config(model_type)
            if not wan_config:
                return {"error": f"No WAN configuration found for model type: {model_type}"}
            
            # Get current hardware profile if available
            hardware_profile = getattr(self, 'hardware_profile', None)
            if not hardware_profile:
                # Try to detect hardware profile
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        hardware_profile = {
                            "gpu_name": gpu_name,
                            "available_vram_gb": available_vram,
                            "architecture_type": "cuda"
                        }
                    else:
                        hardware_profile = {
                            "gpu_name": "CPU Only",
                            "available_vram_gb": 0.0,
                            "architecture_type": "cpu_only"
                        }
                except Exception:
                    hardware_profile = {
                        "gpu_name": "Unknown",
                        "available_vram_gb": 0.0,
                        "architecture_type": "unknown"
                    }
            
            # Assess compatibility
            is_compatible, errors = validate_wan_model_requirements(model_type, hardware_profile)
            
            # Find optimal hardware profile
            optimal_profile = wan_config.get_optimal_profile_for_vram(
                hardware_profile.get("available_vram_gb", 0.0)
            )
            
            # Generate recommendations
            recommendations = []
            if not is_compatible:
                if hardware_profile.get("available_vram_gb", 0.0) < wan_config.optimization.min_vram_gb:
                    recommendations.extend([
                        "Enable CPU offloading to reduce VRAM usage",
                        "Use attention slicing for memory efficiency",
                        "Consider using FP16 or INT8 quantization",
                        "Enable VAE tiling with smaller tile sizes"
                    ])
                else:
                    recommendations.append("Hardware should be compatible with optimizations")
            
            return {
                "is_compatible": is_compatible,
                "compatibility_errors": errors,
                "hardware_detected": hardware_profile,
                "optimal_profile": optimal_profile.profile_name if optimal_profile else None,
                "optimal_profile_config": asdict(optimal_profile) if optimal_profile else None,
                "recommendations": recommendations,
                "vram_utilization": {
                    "available_gb": hardware_profile.get("available_vram_gb", 0.0),
                    "required_gb": wan_config.optimization.min_vram_gb,
                    "estimated_gb": wan_config.optimization.vram_estimate_gb,
                    "utilization_percent": min(100.0, (wan_config.optimization.vram_estimate_gb / 
                                                     max(0.1, hardware_profile.get("available_vram_gb", 0.1))) * 100)
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing hardware compatibility: {e}")
            return {"error": f"Compatibility assessment failed: {str(e)}"}
    
    def get_performance_profile(self, model_type: str) -> Dict[str, Any]:
        """
        Get performance profile and optimization recommendations for a WAN model
        
        Args:
            model_type: WAN model type to profile
            
        Returns:
            Dictionary with performance profile and recommendations
        """
        try:
            if not WAN_CONFIG_AVAILABLE:
                return {"error": "WAN model configuration system not available"}
            
            wan_config = get_wan_model_config(model_type)
            if not wan_config:
                return {"error": f"No WAN configuration found for model type: {model_type}"}
            
            # Get hardware compatibility assessment
            hardware_compat = self.assess_hardware_compatibility(model_type)
            
            # Generate performance recommendations
            performance_recommendations = []
            optimization_settings = {}
            
            if hardware_compat.get("is_compatible", False):
                optimal_profile = hardware_compat.get("optimal_profile_config")
                if optimal_profile:
                    # Use optimal profile settings
                    optimization_settings = {
                        "precision": optimal_profile["precision"],
                        "enable_xformers": optimal_profile["enable_xformers"],
                        "vae_tile_size": optimal_profile["vae_tile_size"],
                        "cpu_offload": optimal_profile["cpu_offload"],
                        "enable_attention_slicing": optimal_profile["enable_attention_slicing"],
                        "enable_sequential_cpu_offload": optimal_profile["enable_sequential_cpu_offload"],
                        "batch_size": optimal_profile["batch_size"]
                    }
                    
                    performance_recommendations.extend([
                        f"Use {optimal_profile['precision']} precision for optimal performance",
                        f"Set batch size to {optimal_profile['batch_size']}",
                        f"Configure VAE tile size to {optimal_profile['vae_tile_size']}"
                    ])
                    
                    if optimal_profile["enable_xformers"]:
                        performance_recommendations.append("Enable xFormers for memory efficiency")
                    
                    if optimal_profile["cpu_offload"]:
                        performance_recommendations.append("Enable CPU offloading to manage VRAM usage")
                    
                    if optimal_profile["enable_attention_slicing"]:
                        performance_recommendations.append("Enable attention slicing for reduced memory usage")
            else:
                # Fallback recommendations for incompatible hardware
                optimization_settings = {
                    "precision": "fp16",
                    "enable_xformers": True,
                    "vae_tile_size": 64,
                    "cpu_offload": True,
                    "enable_attention_slicing": True,
                    "enable_sequential_cpu_offload": True,
                    "batch_size": 1
                }
                
                performance_recommendations.extend([
                    "Use aggressive memory optimizations",
                    "Enable all CPU offloading options",
                    "Use smallest VAE tile size",
                    "Consider upgrading hardware for better performance"
                ])
            
            # Estimate performance metrics
            vram_usage = hardware_compat.get("vram_utilization", {})
            estimated_inference_time = self._estimate_inference_time(model_type, optimization_settings)
            
            return {
                "model_type": model_type,
                "optimization_settings": optimization_settings,
                "performance_recommendations": performance_recommendations,
                "estimated_metrics": {
                    "inference_time_seconds": estimated_inference_time,
                    "vram_usage_gb": vram_usage.get("estimated_gb", 0.0),
                    "vram_utilization_percent": vram_usage.get("utilization_percent", 0.0)
                },
                "hardware_requirements": {
                    "min_vram_gb": wan_config.optimization.min_vram_gb,
                    "recommended_vram_gb": wan_config.optimization.vram_estimate_gb,
                    "supports_cpu_fallback": wan_config.optimization.cpu_offload_enabled
                },
                "optimization_features": {
                    "supports_fp16": wan_config.optimization.supports_fp16,
                    "supports_bf16": wan_config.optimization.supports_bf16,
                    "supports_int8": wan_config.optimization.supports_int8,
                    "supports_torch_compile": wan_config.optimization.supports_torch_compile,
                    "supports_xformers": wan_config.optimization.supports_xformers,
                    "memory_efficient_attention": wan_config.optimization.memory_efficient_attention
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance profile: {e}")
            return {"error": f"Performance profiling failed: {str(e)}"}
    
    def _estimate_inference_time(self, model_type: str, optimization_settings: Dict[str, Any]) -> float:
        """
        Estimate inference time based on model type and optimization settings
        
        Args:
            model_type: WAN model type
            optimization_settings: Optimization configuration
            
        Returns:
            Estimated inference time in seconds
        """
        # Base inference times (rough estimates)
        base_times = {
            "t2v-A14B": 45.0,  # seconds for 16 frames
            "i2v-A14B": 50.0,  # seconds for 16 frames (image conditioning overhead)
            "ti2v-5B": 25.0    # seconds for 16 frames (smaller model)
        }
        
        base_time = base_times.get(model_type, 40.0)
        
        # Apply optimization multipliers
        multiplier = 1.0
        
        # Precision impact
        precision = optimization_settings.get("precision", "fp16")
        if precision == "fp32":
            multiplier *= 1.8
        elif precision == "int8":
            multiplier *= 0.7
        
        # CPU offload impact
        if optimization_settings.get("cpu_offload", False):
            multiplier *= 1.4
        
        # Sequential CPU offload impact
        if optimization_settings.get("enable_sequential_cpu_offload", False):
            multiplier *= 1.8
        
        # Attention slicing impact
        if optimization_settings.get("enable_attention_slicing", False):
            multiplier *= 1.2
        
        # xFormers acceleration
        if optimization_settings.get("enable_xformers", False):
            multiplier *= 0.85
        
        return base_time * multiplier
    
    def is_wan_model(self, model_id: str) -> bool:
        """Check if a model ID refers to a WAN model implementation"""
        model_id = self.get_model_id(model_id)
        return model_id.startswith("wan_implementation:") or model_id in ["t2v-A14B", "i2v-A14B", "ti2v-5B"]

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

# Convenience functions for model operations
def load_wan22_model(model_type: str, **kwargs) -> Tuple[Any, ModelInfo]:
    """Load a Wan2.2 model with caching"""
    manager = get_model_manager()
    return manager.load_model(model_type, **kwargs)

def download_wan22_model(model_type: str, force_download: bool = False) -> str:
    """Download a Wan2.2 model"""
    manager = get_model_manager()
    return manager.download_model(model_type, force_download)

def get_model_status(model_type: str) -> Dict[str, Any]:
    """Get status information for a model"""
    manager = get_model_manager()
    return manager.get_model_status(model_type)

def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available cached models"""
    manager = get_model_manager()
    return manager.list_cached_models()

def get_wan_model_capabilities(model_type: str) -> Dict[str, Any]:
    """Get WAN model capabilities and requirements"""
    manager = get_model_manager()
    return manager.get_wan_model_capabilities(model_type)

def validate_wan_model_configuration(model_type: str) -> Tuple[bool, List[str]]:
    """Validate WAN model configuration"""
    manager = get_model_manager()
    return manager.validate_wan_model_configuration(model_type)

def get_wan_model_recommendations(available_vram_gb: float) -> Dict[str, Any]:
    """Get WAN model recommendations for available hardware"""
    manager = get_model_manager()
    return manager.get_wan_model_recommendations(available_vram_gb)

def is_wan_model(model_id: str) -> bool:
    """Check if model ID refers to a WAN model implementation"""
    manager = get_model_manager()
    return manager.is_wan_model(model_id)

def assess_hardware_compatibility(model_type: str) -> Dict[str, Any]:
    """Assess hardware compatibility for a WAN model"""
    manager = get_model_manager()
    return manager.assess_hardware_compatibility(model_type)

def get_performance_profile(model_type: str) -> Dict[str, Any]:
    """Get performance profile and optimization recommendations for a WAN model"""
    manager = get_model_manager()
    return manager.get_performance_profile(model_type)

def get_all_wan_model_status() -> Dict[str, Dict[str, Any]]:
    """Get comprehensive status for all WAN models"""
    manager = get_model_manager()
    wan_models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
    
    status_report = {}
    for model_type in wan_models:
        status_report[model_type] = manager.get_model_status(model_type)
    
    return status_report

def validate_all_wan_configurations() -> Dict[str, Any]:
    """Validate all WAN model configurations"""
    if not WAN_CONFIG_AVAILABLE:
        return {"error": "WAN model configuration system not available"}
    
    validation_results = {
        "overall_valid": True,
        "model_validations": {},
        "summary": {
            "total_models": 0,
            "valid_models": 0,
            "invalid_models": 0,
            "errors": []
        }
    }
    
    wan_models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
    
    for model_type in wan_models:
        is_valid, errors = validate_wan_model_configuration(model_type)
        validation_results["model_validations"][model_type] = {
            "is_valid": is_valid,
            "errors": errors
        }
        
        validation_results["summary"]["total_models"] += 1
        if is_valid:
            validation_results["summary"]["valid_models"] += 1
        else:
            validation_results["summary"]["invalid_models"] += 1
            validation_results["summary"]["errors"].extend(
                [f"{model_type}: {error}" for error in errors]
            )
            validation_results["overall_valid"] = False
    
    return validation_results
