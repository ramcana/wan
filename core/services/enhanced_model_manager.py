"""
Enhanced Model Management System for Wan2.2 Video Generation
Provides robust model loading, availability validation, compatibility verification, and fallback strategies
"""

import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import shutil

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download, HfApi, model_info
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model availability status"""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    LOADED = "loaded"
    ERROR = "error"
    CORRUPTED = "corrupted"
    MISSING = "missing"

class GenerationMode(Enum):
    """Video generation modes"""
    TEXT_TO_VIDEO = "t2v"
    IMAGE_TO_VIDEO = "i2v"
    TEXT_IMAGE_TO_VIDEO = "ti2v"

class ModelCompatibility(Enum):
    """Model compatibility levels"""
    FULLY_COMPATIBLE = "fully_compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    model_type: str
    generation_modes: List[GenerationMode]
    supported_resolutions: List[str]
    min_vram_mb: float
    recommended_vram_mb: float
    model_size_mb: float
    quantization_support: List[str]
    cpu_offload_support: bool
    vae_tiling_support: bool
    last_validated: Optional[datetime] = None
    validation_hash: Optional[str] = None
    download_url: Optional[str] = None
    config_hash: Optional[str] = None

@dataclass
class ModelLoadingResult:
    """Result of model loading operation"""
    success: bool
    model: Optional[Any] = None
    metadata: Optional[ModelMetadata] = None
    error_message: Optional[str] = None
    fallback_applied: bool = False
    optimization_applied: Dict[str, Any] = field(default_factory=dict)
    loading_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0

@dataclass
class CompatibilityCheck:
    """Model compatibility check result"""
    model_id: str
    generation_mode: GenerationMode
    resolution: str
    compatibility: ModelCompatibility
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    estimated_vram_mb: float = 0.0

class EnhancedModelManager:
    """Enhanced model management with robust error handling and fallback strategies"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.cache_dir = Path(self.config["directories"]["models_directory"])
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model registry and status tracking
        self.model_registry: Dict[str, ModelMetadata] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.loading_locks: Dict[str, threading.Lock] = {}
        
        # Model mappings and fallbacks
        self.model_mappings = {
            "t2v-A14B": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "i2v-A14B": "Wan-AI/Wan2.2-I2V-A14B-Diffusers", 
            "ti2v-5B": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        }
        
        self.fallback_models = {
            "t2v-A14B": ["t2v-A14B-quantized", "t2v-base"],
            "i2v-A14B": ["i2v-A14B-quantized", "i2v-base"],
            "ti2v-5B": ["ti2v-5B-quantized", "ti2v-base"]
        }
        
        # Initialize model registry
        self._initialize_model_registry()
        
        # Background validation thread
        self._start_background_validation()
    
    @handle_error_with_recovery
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration with fallback"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log_error_with_context(e, "config_loading", {"config_path": config_path})
            logger.warning("Using default configuration due to config loading error")
            return {
                "directories": {
                    "models_directory": "models", 
                    "outputs_directory": "outputs", 
                    "loras_directory": "loras"
                },
                "optimization": {"max_vram_usage_gb": 12},
                "model_validation": {
                    "validate_on_startup": True,
                    "validation_interval_hours": 24,
                    "auto_repair_corrupted": True
                }
            }
    
    def _initialize_model_registry(self):
        """Initialize the model registry with known models"""
        # Define known model metadata
        known_models = {
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers": ModelMetadata(
                model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                model_type="text-to-video",
                generation_modes=[GenerationMode.TEXT_TO_VIDEO],
                supported_resolutions=["1280x720", "1920x1080", "512x512"],
                min_vram_mb=6000,
                recommended_vram_mb=8000,
                model_size_mb=7500,
                quantization_support=["bf16", "fp16", "int8"],
                cpu_offload_support=True,
                vae_tiling_support=True
            ),
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers": ModelMetadata(
                model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                model_type="image-to-video",
                generation_modes=[GenerationMode.IMAGE_TO_VIDEO],
                supported_resolutions=["1280x720", "1920x1080", "512x512"],
                min_vram_mb=6500,
                recommended_vram_mb=8500,
                model_size_mb=8000,
                quantization_support=["bf16", "fp16", "int8"],
                cpu_offload_support=True,
                vae_tiling_support=True
            ),
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers": ModelMetadata(
                model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                model_type="text-image-to-video",
                generation_modes=[GenerationMode.TEXT_IMAGE_TO_VIDEO],
                supported_resolutions=["1280x720", "1920x1080", "512x512"],
                min_vram_mb=5000,
                recommended_vram_mb=6000,
                model_size_mb=5500,
                quantization_support=["bf16", "fp16", "int8"],
                cpu_offload_support=True,
                vae_tiling_support=True
            )
        }
        
        for model_id, metadata in known_models.items():
            self.model_registry[model_id] = metadata
            self.model_status[model_id] = ModelStatus.UNKNOWN
            self.loading_locks[model_id] = threading.Lock()
    
    def get_model_id(self, model_type: str) -> str:
        """Get the full Hugging Face model ID for a model type"""
        if model_type in self.model_mappings:
            return self.model_mappings[model_type]
        return model_type
    
    @handle_error_with_recovery
    def validate_model_availability(self, model_id: str, force_check: bool = False) -> ModelStatus:
        """Validate model availability with comprehensive checks"""
        full_model_id = self.get_model_id(model_id)
        
        # Check if we have recent validation results
        if not force_check and full_model_id in self.model_status:
            last_check = self.model_registry.get(full_model_id, {}).last_validated
            if last_check and (datetime.now() - last_check) < timedelta(hours=1):
                return self.model_status[full_model_id]
        
        logger.info(f"Validating model availability: {full_model_id}")
        
        try:
            # Check local cache first
            local_status = self._check_local_model(full_model_id)
            if local_status in [ModelStatus.AVAILABLE, ModelStatus.LOADED]:
                self.model_status[full_model_id] = local_status
                return local_status
            
            # Check remote availability
            remote_status = self._check_remote_model(full_model_id)
            self.model_status[full_model_id] = remote_status
            
            # Update validation timestamp
            if full_model_id in self.model_registry:
                self.model_registry[full_model_id].last_validated = datetime.now()
            
            return remote_status
            
        except Exception as e:
            log_error_with_context(e, "model_availability_check", {"model_id": full_model_id})
            self.model_status[full_model_id] = ModelStatus.ERROR
            return ModelStatus.ERROR
    
    def _check_local_model(self, model_id: str) -> ModelStatus:
        """Check if model is available locally and validate integrity"""
        model_path = self.cache_dir / model_id.replace("/", "_")
        
        if not model_path.exists():
            return ModelStatus.MISSING
        
        # Check if model is currently loaded
        if model_id in self.loaded_models:
            return ModelStatus.LOADED
        
        # Validate model integrity
        required_files = ["config.json"]
        for file in required_files:
            if not (model_path / file).exists():
                logger.warning(f"Missing required file {file} for model {model_id}")
                return ModelStatus.CORRUPTED
        
        # Check for model weights
        weight_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
        if not weight_files:
            logger.warning(f"No model weights found for {model_id}")
            return ModelStatus.CORRUPTED
        
        # Validate config integrity
        try:
            with open(model_path / "config.json", 'r') as f:
                config = json.load(f)
            
            # Calculate and verify config hash if available
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            if model_id in self.model_registry:
                stored_hash = self.model_registry[model_id].config_hash
                if stored_hash and stored_hash != config_hash:
                    logger.warning(f"Config hash mismatch for {model_id}")
                    return ModelStatus.CORRUPTED
                else:
                    self.model_registry[model_id].config_hash = config_hash
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to validate config for {model_id}: {e}")
            return ModelStatus.CORRUPTED
        
        return ModelStatus.AVAILABLE
    
    def _check_remote_model(self, model_id: str) -> ModelStatus:
        """Check if model is available on Hugging Face Hub"""
        try:
            api = HfApi()
            info = api.model_info(model_id)
            
            # Update model metadata if available
            if model_id in self.model_registry:
                # Extract additional metadata from model info
                if hasattr(info, 'tags') and info.tags:
                    # Update supported features based on tags
                    pass
            
            return ModelStatus.AVAILABLE
            
        except RepositoryNotFoundError:
            logger.error(f"Model {model_id} not found on Hugging Face Hub")
            return ModelStatus.MISSING
        except Exception as e:
            log_error_with_context(e, "remote_model_check", {"model_id": model_id})
            return ModelStatus.ERROR
    
    @handle_error_with_recovery
    def check_model_compatibility(self, model_id: str, generation_mode: GenerationMode, 
                                 resolution: str = "1280x720") -> CompatibilityCheck:
        """Check model compatibility for specific generation requirements"""
        full_model_id = self.get_model_id(model_id)
        
        compatibility_check = CompatibilityCheck(
            model_id=full_model_id,
            generation_mode=generation_mode,
            resolution=resolution,
            compatibility=ModelCompatibility.UNKNOWN
        )
        
        try:
            # Get model metadata
            if full_model_id not in self.model_registry:
                compatibility_check.compatibility = ModelCompatibility.UNKNOWN
                compatibility_check.issues.append("Model metadata not available")
                return compatibility_check
            
            metadata = self.model_registry[full_model_id]
            
            # Check generation mode compatibility
            if generation_mode not in metadata.generation_modes:
                compatibility_check.compatibility = ModelCompatibility.INCOMPATIBLE
                compatibility_check.issues.append(f"Model does not support {generation_mode.value} generation")
                return compatibility_check
            
            # Check resolution compatibility
            if resolution not in metadata.supported_resolutions:
                compatibility_check.compatibility = ModelCompatibility.PARTIALLY_COMPATIBLE
                compatibility_check.issues.append(f"Resolution {resolution} not officially supported")
                compatibility_check.recommendations.append(f"Consider using supported resolutions: {', '.join(metadata.supported_resolutions)}")
            
            # Check VRAM requirements
            vram_info = self._get_vram_info()
            compatibility_check.estimated_vram_mb = metadata.min_vram_mb
            
            if vram_info["total_mb"] < metadata.min_vram_mb:
                compatibility_check.compatibility = ModelCompatibility.INCOMPATIBLE
                compatibility_check.issues.append(f"Insufficient VRAM: {vram_info['total_mb']:.0f}MB available, {metadata.min_vram_mb:.0f}MB required")
                compatibility_check.recommendations.append("Consider using quantization or CPU offload")
            elif vram_info["total_mb"] < metadata.recommended_vram_mb:
                if compatibility_check.compatibility != ModelCompatibility.PARTIALLY_COMPATIBLE:
                    compatibility_check.compatibility = ModelCompatibility.PARTIALLY_COMPATIBLE
                compatibility_check.issues.append(f"Below recommended VRAM: {metadata.recommended_vram_mb:.0f}MB recommended")
                compatibility_check.recommendations.append("Performance may be reduced, consider optimizations")
            else:
                if compatibility_check.compatibility == ModelCompatibility.UNKNOWN:
                    compatibility_check.compatibility = ModelCompatibility.FULLY_COMPATIBLE
            
            # Check disk space
            disk_usage = psutil.disk_usage(str(self.cache_dir))
            free_gb = disk_usage.free / (1024**3)
            required_gb = metadata.model_size_mb / 1024
            
            if free_gb < required_gb:
                compatibility_check.compatibility = ModelCompatibility.INCOMPATIBLE
                compatibility_check.issues.append(f"Insufficient disk space: {free_gb:.1f}GB available, {required_gb:.1f}GB required")
            
            return compatibility_check
            
        except Exception as e:
            log_error_with_context(e, "compatibility_check", {
                "model_id": full_model_id,
                "generation_mode": generation_mode.value,
                "resolution": resolution
            })
            compatibility_check.compatibility = ModelCompatibility.UNKNOWN
            compatibility_check.issues.append(f"Compatibility check failed: {str(e)}")
            return compatibility_check
    
    def _get_vram_info(self) -> Dict[str, float]:
        """Get VRAM information"""
        if not torch.cuda.is_available():
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            
            total_mb = total_memory / (1024 * 1024)
            used_mb = allocated_memory / (1024 * 1024)
            free_mb = total_mb - used_mb
            
            return {"total_mb": total_mb, "used_mb": used_mb, "free_mb": free_mb}
        except Exception:
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}
    
    @handle_error_with_recovery
    def load_model_with_fallback(self, model_id: str, **kwargs) -> ModelLoadingResult:
        """Load model with comprehensive fallback strategies"""
        full_model_id = self.get_model_id(model_id)
        start_time = time.time()
        
        # Initialize result
        result = ModelLoadingResult(success=False)
        
        # Check if model is already loaded
        if full_model_id in self.loaded_models:
            logger.info(f"Using already loaded model: {full_model_id}")
            result.success = True
            result.model = self.loaded_models[full_model_id]
            result.metadata = self.model_registry.get(full_model_id)
            result.loading_time_seconds = time.time() - start_time
            return result
        
        # Acquire loading lock to prevent concurrent loading
        with self.loading_locks.get(full_model_id, threading.Lock()):
            # Double-check after acquiring lock
            if full_model_id in self.loaded_models:
                result.success = True
                result.model = self.loaded_models[full_model_id]
                result.metadata = self.model_registry.get(full_model_id)
                result.loading_time_seconds = time.time() - start_time
                return result
            
            # Try primary model first
            result = self._attempt_model_loading(full_model_id, **kwargs)
            
            # If primary model fails, try fallbacks
            if not result.success and full_model_id in self.fallback_models:
                logger.warning(f"Primary model {full_model_id} failed, trying fallbacks")
                
                for fallback_id in self.fallback_models[full_model_id]:
                    fallback_result = self._attempt_model_loading(fallback_id, **kwargs)
                    if fallback_result.success:
                        result = fallback_result
                        result.fallback_applied = True
                        logger.info(f"Successfully loaded fallback model: {fallback_id}")
                        break
            
            result.loading_time_seconds = time.time() - start_time
            return result
    
    def _attempt_model_loading(self, model_id: str, **kwargs) -> ModelLoadingResult:
        """Attempt to load a specific model with error handling"""
        result = ModelLoadingResult(success=False)
        
        try:
            # Validate model availability
            status = self.validate_model_availability(model_id)
            if status not in [ModelStatus.AVAILABLE, ModelStatus.LOADED]:
                result.error_message = f"Model not available: {status.value}"
                return result
            
            # Check VRAM before loading
            vram_info = self._get_vram_info()
            if model_id in self.model_registry:
                required_vram = self.model_registry[model_id].min_vram_mb
                if vram_info["free_mb"] < required_vram:
                    # Try to free memory
                    self._free_memory()
                    vram_info = self._get_vram_info()
                    
                    if vram_info["free_mb"] < required_vram:
                        result.error_message = f"Insufficient VRAM: {vram_info['free_mb']:.0f}MB free, {required_vram:.0f}MB required"
                        return result
            
            # Ensure model is downloaded
            model_path = self._ensure_model_downloaded(model_id)
            
            # Load the model
            logger.info(f"Loading model from: {model_path}")
            
            # Apply default optimizations
            loading_kwargs = {
                "torch_dtype": torch.bfloat16,
                "use_safetensors": True,
                **kwargs
            }
            
            pipeline = DiffusionPipeline.from_pretrained(model_path, **loading_kwargs)
            
            # Apply additional optimizations
            optimizations = self._apply_loading_optimizations(pipeline, model_id)
            
            # Calculate memory usage
            memory_usage = self._calculate_model_memory(pipeline)
            
            # Cache the loaded model
            self.loaded_models[model_id] = pipeline
            self.model_status[model_id] = ModelStatus.LOADED
            
            # Update result
            result.success = True
            result.model = pipeline
            result.metadata = self.model_registry.get(model_id)
            result.optimization_applied = optimizations
            result.memory_usage_mb = memory_usage
            
            logger.info(f"Successfully loaded model: {model_id} ({memory_usage:.1f} MB)")
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            log_error_with_context(e, "model_loading_oom", {"model_id": model_id, "vram_info": vram_info if 'vram_info' in locals() else None})
            result.error_message = f"Out of memory loading model: {str(e)}"
            return result
        except Exception as e:
            log_error_with_context(e, "model_loading", {"model_id": model_id})
            result.error_message = f"Failed to load model: {str(e)}"
            return result
    
    def _ensure_model_downloaded(self, model_id: str) -> str:
        """Ensure model is downloaded and return path"""
        model_path = self.cache_dir / model_id.replace("/", "_")
        
        # Check if model is already cached and valid
        if self._check_local_model(model_id) == ModelStatus.AVAILABLE:
            return str(model_path)
        
        # Download model
        logger.info(f"Downloading model: {model_id}")
        
        try:
            downloaded_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(self.cache_dir),
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            # Update model status
            self.model_status[model_id] = ModelStatus.AVAILABLE
            
            return downloaded_path
            
        except Exception as e:
            log_error_with_context(e, "model_download", {"model_id": model_id})
            raise
    
    def _apply_loading_optimizations(self, pipeline, model_id: str) -> Dict[str, Any]:
        """Apply optimizations during model loading"""
        optimizations = {}
        
        try:
            # Enable memory efficient attention if available
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
                optimizations["attention_slicing"] = True
            
            # Enable VAE tiling if supported
            if hasattr(pipeline, 'enable_vae_tiling'):
                pipeline.enable_vae_tiling()
                optimizations["vae_tiling"] = True
            
            # Enable CPU offload if VRAM is limited
            vram_info = self._get_vram_info()
            if model_id in self.model_registry:
                recommended_vram = self.model_registry[model_id].recommended_vram_mb
                if vram_info["total_mb"] < recommended_vram:
                    if hasattr(pipeline, 'enable_model_cpu_offload'):
                        pipeline.enable_model_cpu_offload()
                        optimizations["cpu_offload"] = True
            
            return optimizations
            
        except Exception as e:
            logger.warning(f"Failed to apply some optimizations: {e}")
            return optimizations
    
    def _calculate_model_memory(self, model) -> float:
        """Calculate approximate memory usage of a model in MB"""
        try:
            total_params = 0
            if hasattr(model, 'unet') and model.unet is not None:
                total_params += sum(p.numel() for p in model.unet.parameters())
            if hasattr(model, 'vae') and model.vae is not None:
                total_params += sum(p.numel() for p in model.vae.parameters())
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                total_params += sum(p.numel() for p in model.text_encoder.parameters())
            
            # Estimate memory usage (assuming bf16/fp16)
            memory_bytes = total_params * 2
            return memory_bytes / (1024 * 1024)
            
        except Exception as e:
            logger.warning(f"Failed to calculate model memory: {e}")
            return 0.0
    
    def _free_memory(self):
        """Free GPU memory by unloading least recently used models"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Could implement LRU eviction here if needed
        # For now, just clear GPU cache
    
    def _start_background_validation(self):
        """Start background thread for periodic model validation"""
        if not self.config.get("model_validation", {}).get("validate_on_startup", True):
            return
        
        def validation_worker():
            while True:
                try:
                    interval_hours = self.config.get("model_validation", {}).get("validation_interval_hours", 24)
                    time.sleep(interval_hours * 3600)  # Convert to seconds
                    
                    # Validate all known models
                    for model_id in self.model_registry.keys():
                        self.validate_model_availability(model_id, force_check=True)
                        
                except Exception as e:
                    logger.error(f"Background validation error: {e}")
                    time.sleep(3600)  # Wait 1 hour before retrying
        
        validation_thread = threading.Thread(target=validation_worker, daemon=True)
        validation_thread.start()
    
    def get_model_status_report(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive status report for a model"""
        full_model_id = self.get_model_id(model_id)
        
        report = {
            "model_id": full_model_id,
            "status": self.model_status.get(full_model_id, ModelStatus.UNKNOWN).value,
            "is_loaded": full_model_id in self.loaded_models,
            "metadata": None,
            "local_path": None,
            "size_mb": 0.0,
            "last_validated": None,
            "compatibility": {}
        }
        
        # Add metadata if available
        if full_model_id in self.model_registry:
            metadata = self.model_registry[full_model_id]
            report["metadata"] = {
                "model_type": metadata.model_type,
                "generation_modes": [mode.value for mode in metadata.generation_modes],
                "supported_resolutions": metadata.supported_resolutions,
                "min_vram_mb": metadata.min_vram_mb,
                "recommended_vram_mb": metadata.recommended_vram_mb,
                "model_size_mb": metadata.model_size_mb
            }
            report["last_validated"] = metadata.last_validated.isoformat() if metadata.last_validated else None
        
        # Check local path
        model_path = self.cache_dir / full_model_id.replace("/", "_")
        if model_path.exists():
            report["local_path"] = str(model_path)
            report["size_mb"] = self._get_directory_size_mb(model_path)
        
        # Add compatibility checks for common scenarios
        for mode in GenerationMode:
            for resolution in ["1280x720", "1920x1080"]:
                compat = self.check_model_compatibility(full_model_id, mode, resolution)
                report["compatibility"][f"{mode.value}_{resolution}"] = {
                    "compatibility": compat.compatibility.value,
                    "issues": compat.issues,
                    "recommendations": compat.recommendations
                }
        
        return report
    
    def _get_directory_size_mb(self, path: Path) -> float:
        """Get directory size in MB"""
        try:
            total_size = 0
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def repair_corrupted_model(self, model_id: str) -> bool:
        """Attempt to repair a corrupted model"""
        full_model_id = self.get_model_id(model_id)
        
        try:
            logger.info(f"Attempting to repair corrupted model: {full_model_id}")
            
            # Remove corrupted local copy
            model_path = self.cache_dir / full_model_id.replace("/", "_")
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Re-download model
            self._ensure_model_downloaded(full_model_id)
            
            # Validate repair
            status = self.validate_model_availability(full_model_id, force_check=True)
            if status == ModelStatus.AVAILABLE:
                logger.info(f"Successfully repaired model: {full_model_id}")
                return True
            else:
                logger.error(f"Failed to repair model: {full_model_id}")
                return False
                
        except Exception as e:
            log_error_with_context(e, "model_repair", {"model_id": full_model_id})
            return False
    
    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        full_model_id = self.get_model_id(model_id)
        
        if full_model_id in self.loaded_models:
            del self.loaded_models[full_model_id]
            self.model_status[full_model_id] = ModelStatus.AVAILABLE
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model: {full_model_id}")
    
    def list_all_models(self) -> Dict[str, Dict[str, Any]]:
        """List all models with their status and metadata"""
        models = {}
        
        for model_id in self.model_registry.keys():
            models[model_id] = self.get_model_status_report(model_id)
        
        return models


# Global enhanced model manager instance
_enhanced_model_manager = None

def get_enhanced_model_manager() -> EnhancedModelManager:
    """Get the global enhanced model manager instance"""
    global _enhanced_model_manager
    if _enhanced_model_manager is None:
        _enhanced_model_manager = EnhancedModelManager()
    return _enhanced_model_manager

# Convenience functions
def validate_model_availability(model_id: str, force_check: bool = False) -> ModelStatus:
    """Validate model availability"""
    manager = get_enhanced_model_manager()
    return manager.validate_model_availability(model_id, force_check)

def check_model_compatibility(model_id: str, generation_mode: GenerationMode, 
                            resolution: str = "1280x720") -> CompatibilityCheck:
    """Check model compatibility"""
    manager = get_enhanced_model_manager()
    return manager.check_model_compatibility(model_id, generation_mode, resolution)

def load_model_with_fallback(model_id: str, **kwargs) -> ModelLoadingResult:
    """Load model with fallback strategies"""
    manager = get_enhanced_model_manager()
    return manager.load_model_with_fallback(model_id, **kwargs)

def get_model_status_report(model_id: str) -> Dict[str, Any]:
    """Get comprehensive model status report"""
    manager = get_enhanced_model_manager()
    return manager.get_model_status_report(model_id)