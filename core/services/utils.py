"""
Core utilities for Wan2.2 UI Variant
Handles model management, optimization, and generation workflows
"""

import os
import json
import hashlib
import logging
import threading
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from enum import Enum
import shutil
import subprocess

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
import psutil
import GPUtil
from PIL import Image
import cv2
import numpy as np

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
from core.services.wan_pipeline_loader import WanPipelineLoader, GenerationConfig
from core.services.optimization_manager import OptimizationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enumeration for task status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationTask:
    """Data structure for video generation tasks with enhanced image support"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str = ""  # 't2v-A14B', 'i2v-A14B', 'ti2v-5B'
    prompt: str = ""
    image: Optional[Image.Image] = None  # Start frame image for I2V/TI2V
    end_image: Optional[Image.Image] = None  # End frame image for I2V/TI2V
    resolution: str = "1280x720"
    steps: int = 50
    duration: int = 4  # Video duration in seconds
    fps: int = 24  # Frames per second
    lora_path: Optional[str] = None
    lora_strength: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0  # Progress percentage (0.0 to 100.0)
    
    # Enhanced image metadata for better tracking and validation
    image_metadata: Optional[Dict[str, Any]] = None
    end_image_metadata: Optional[Dict[str, Any]] = None
    
    # Image storage paths for queue persistence
    image_temp_path: Optional[str] = None
    end_image_temp_path: Optional[str] = None
    
    # Enhanced download support for larger files
    download_timeout: int = 300  # 5 minutes default timeout
    smart_download_enabled: bool = True  # Enable smart downloading features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            "id": self.id,
            "model_type": self.model_type,
            "prompt": self.prompt,
            "image": None if self.image is None else "Start frame image object",
            "end_image": None if self.end_image is None else "End frame image object",
            "resolution": self.resolution,
            "steps": self.steps,
            "duration": self.duration,
            "fps": self.fps,
            "lora_path": self.lora_path,
            "lora_strength": self.lora_strength,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "progress": self.progress,
            "image_metadata": self.image_metadata,
            "end_image_metadata": self.end_image_metadata,
            "image_temp_path": self.image_temp_path,
            "end_image_temp_path": self.end_image_temp_path
        }
    
    def update_status(self, status: TaskStatus, error_message: Optional[str] = None):
        """Update task status with optional error message"""
        self.status = status
        if error_message:
            self.error_message = error_message
        if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
            self.completed_at = datetime.now()
    
    def store_image_data(self, start_image: Optional[Image.Image] = None, 
                        end_image: Optional[Image.Image] = None) -> None:
        """Store image data with enhanced metadata and validation for generation pipeline"""
        import tempfile
        import os
        
        if start_image is not None:
            # Validate image before storing
            if not self._validate_image(start_image, "start"):
                logger.warning("Start image validation failed, storing anyway")
            
            self.image = start_image
            # Generate comprehensive metadata
            self.image_metadata = {
                "format": start_image.format or "PNG",
                "size": start_image.size,
                "mode": start_image.mode,
                "has_transparency": start_image.mode in ("RGBA", "LA") or "transparency" in start_image.info,
                "aspect_ratio": start_image.size[0] / start_image.size[1] if start_image.size[1] > 0 else 1.0,
                "pixel_count": start_image.size[0] * start_image.size[1],
                "stored_at": datetime.now().isoformat(),
                "validation_passed": True,
                "file_size_estimate": self._estimate_image_size(start_image)
            }
            
            # Save temporary copy for queue persistence with better error handling
            try:
                # Create dedicated temp directory for WAN22 images
                temp_dir = os.path.join(tempfile.gettempdir(), "wan22_images")
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_path = os.path.join(temp_dir, f"start_{self.id}.png")
                # Save with optimal settings for queue persistence
                start_image.save(temp_path, "PNG", optimize=True)
                self.image_temp_path = temp_path
                
                # Verify the saved file
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    logger.debug(f"Saved start image to temporary path: {temp_path}")
                    self.image_metadata["temp_file_size"] = os.path.getsize(temp_path)
                else:
                    raise Exception("Temporary file was not created properly")
                    
            except Exception as e:
                logger.error(f"Failed to save temporary start image: {e}")
                self.image_metadata["temp_storage_error"] = str(e)
                # Set temp path to None to indicate failure
                self.image_temp_path = None
        
        if end_image is not None:
            # Validate image before storing
            if not self._validate_image(end_image, "end"):
                logger.warning("End image validation failed, storing anyway")
            
            self.end_image = end_image
            # Generate comprehensive metadata
            self.end_image_metadata = {
                "format": end_image.format or "PNG",
                "size": end_image.size,
                "mode": end_image.mode,
                "has_transparency": end_image.mode in ("RGBA", "LA") or "transparency" in end_image.info,
                "aspect_ratio": end_image.size[0] / end_image.size[1] if end_image.size[1] > 0 else 1.0,
                "pixel_count": end_image.size[0] * end_image.size[1],
                "stored_at": datetime.now().isoformat(),
                "validation_passed": True,
                "file_size_estimate": self._estimate_image_size(end_image)
            }
            
            # Validate aspect ratio compatibility if both images exist
            if self.image is not None and self.image_metadata is not None:
                start_ratio = self.image_metadata["aspect_ratio"]
                end_ratio = self.end_image_metadata["aspect_ratio"]
                ratio_diff = abs(start_ratio - end_ratio)
                
                if ratio_diff > 0.1:  # Allow 10% difference
                    logger.warning(f"Aspect ratio mismatch: start={start_ratio:.3f}, end={end_ratio:.3f}")
                    self.end_image_metadata["aspect_ratio_warning"] = f"Differs from start image by {ratio_diff:.3f}"
            
            # Save temporary copy for queue persistence
            try:
                # Create dedicated temp directory for WAN22 images
                temp_dir = os.path.join(tempfile.gettempdir(), "wan22_images")
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_path = os.path.join(temp_dir, f"end_{self.id}.png")
                # Save with optimal settings for queue persistence
                end_image.save(temp_path, "PNG", optimize=True)
                self.end_image_temp_path = temp_path
                
                # Verify the saved file
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    logger.debug(f"Saved end image to temporary path: {temp_path}")
                    self.end_image_metadata["temp_file_size"] = os.path.getsize(temp_path)
                else:
                    raise Exception("Temporary file was not created properly")
                    
            except Exception as e:
                logger.error(f"Failed to save temporary end image: {e}")
                self.end_image_metadata["temp_storage_error"] = str(e)
                # Set temp path to None to indicate failure
                self.end_image_temp_path = None
    
    def _validate_image(self, image: Image.Image, image_type: str) -> bool:
        """Validate image for generation pipeline compatibility"""
        try:
            # Check minimum dimensions
            min_size = 256
            if image.size[0] < min_size or image.size[1] < min_size:
                logger.warning(f"{image_type} image too small: {image.size}, minimum: {min_size}x{min_size}")
                return False
            
            # Check maximum dimensions to prevent memory issues
            max_size = 2048
            if image.size[0] > max_size or image.size[1] > max_size:
                logger.warning(f"{image_type} image too large: {image.size}, maximum: {max_size}x{max_size}")
                return False
            
            # Check if image is valid
            image.verify()
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed for {image_type}: {e}")
            return False
    
    def _estimate_image_size(self, image: Image.Image) -> int:
        """Estimate the file size of an image in bytes"""
        # Rough estimation based on dimensions and mode
        pixel_count = image.size[0] * image.size[1]
        
        if image.mode == "RGB":
            return pixel_count * 3
        elif image.mode == "RGBA":
            return pixel_count * 4
        elif image.mode == "L":
            return pixel_count
        else:
            return pixel_count * 3  # Default estimate
    
    def restore_image_data(self) -> bool:
        """Restore image data from temporary paths for queue processing"""
        restoration_success = True
        
        # Handle start image restoration
        if self.image_temp_path:
            if os.path.exists(self.image_temp_path):
                try:
                    # Verify file integrity before loading
                    file_size = os.path.getsize(self.image_temp_path)
                    if file_size == 0:
                        raise Exception("Temporary start image file is empty")
                    
                    self.image = Image.open(self.image_temp_path)
                    # Verify the image can be loaded properly
                    self.image.load()
                    
                    logger.debug(f"Restored start image from: {self.image_temp_path} ({file_size} bytes)")
                    
                    # Update metadata with restoration info
                    if self.image_metadata:
                        self.image_metadata["restored_at"] = datetime.now().isoformat()
                        self.image_metadata["restored_file_size"] = file_size
                        
                except Exception as e:
                    logger.error(f"Failed to restore start image from {self.image_temp_path}: {e}")
                    self.image = None
                    restoration_success = False
                    
                    # Update metadata with error info
                    if self.image_metadata:
                        self.image_metadata["restoration_error"] = str(e)
            else:
                logger.warning(f"Start image temp file not found: {self.image_temp_path}")
                restoration_success = False
                if self.image_metadata:
                    self.image_metadata["restoration_error"] = "Temp file not found"
        elif self.image is None:
            # No temp path and no image - this might be expected for some model types
            logger.debug("No start image temp path available for restoration")
        
        # Handle end image restoration
        if self.end_image_temp_path:
            if os.path.exists(self.end_image_temp_path):
                try:
                    # Verify file integrity before loading
                    file_size = os.path.getsize(self.end_image_temp_path)
                    if file_size == 0:
                        raise Exception("Temporary end image file is empty")
                    
                    self.end_image = Image.open(self.end_image_temp_path)
                    # Verify the image can be loaded properly
                    self.end_image.load()
                    
                    logger.debug(f"Restored end image from: {self.end_image_temp_path} ({file_size} bytes)")
                    
                    # Update metadata with restoration info
                    if self.end_image_metadata:
                        self.end_image_metadata["restored_at"] = datetime.now().isoformat()
                        self.end_image_metadata["restored_file_size"] = file_size
                        
                except Exception as e:
                    logger.error(f"Failed to restore end image from {self.end_image_temp_path}: {e}")
                    self.end_image = None
                    restoration_success = False
                    
                    # Update metadata with error info
                    if self.end_image_metadata:
                        self.end_image_metadata["restoration_error"] = str(e)
            else:
                logger.warning(f"End image temp file not found: {self.end_image_temp_path}")
                restoration_success = False
                if self.end_image_metadata:
                    self.end_image_metadata["restoration_error"] = "Temp file not found"
        elif self.end_image is None:
            # No temp path and no end image - this is often expected
            logger.debug("No end image temp path available for restoration")
        
        return restoration_success
    
    def cleanup_temp_images(self) -> None:
        """Clean up temporary image files"""
        import os
        
        for temp_path in [self.image_temp_path, self.end_image_temp_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"Cleaned up temporary image: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary image {temp_path}: {e}")
        
        # Always reset paths regardless of cleanup success
        self.image_temp_path = None
        self.end_image_temp_path = None
    
    def has_images(self) -> bool:
        """Check if task includes image data"""
        return self.image is not None or self.end_image is not None
    
    def get_image_summary(self) -> str:
        """Get summary of included images"""
        parts = []
        if self.image is not None:
            size_str = f"{self.image.size[0]}x{self.image.size[1]}" if self.image.size else "unknown"
            parts.append(f"start image ({size_str})")
        if self.end_image is not None:
            size_str = f"{self.end_image.size[0]}x{self.end_image.size[1]}" if self.end_image.size else "unknown"
            parts.append(f"end image ({size_str})")
        
        return ", ".join(parts) if parts else "no images"

@dataclass
class ModelInfo:
    """Information about a loaded model"""
    model_type: str
    model_path: str
    model_id: str
    loaded_at: datetime
    memory_usage_mb: float
    is_quantized: bool = False
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
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache info: {e}")
        return {}
    
    def _save_cache_info(self):
        """Save cache information to disk"""
        try:
            with open(self.cache_info_file, 'w') as f:
                json.dump(self.cache_info, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save cache info: {e}")
    
    def get_model_path(self, model_id: str) -> Path:
        """Get the local path for a model"""
        # Create a safe directory name from model ID
        safe_name = model_id.replace("/", "_").replace(":", "_")
        return self.cache_dir / safe_name
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is cached locally"""
        model_path = self.get_model_path(model_id)
        return model_path.exists() and (model_path / "config.json").exists()
    
    def get_cache_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get cached model information"""
        return self.cache_info.get(model_id)
    
    def update_cache_info(self, model_id: str, info: Dict[str, Any]):
        """Update cache information for a model"""
        self.cache_info[model_id] = {
            **info,
            "last_accessed": datetime.now().isoformat(),
            "cache_path": str(self.get_model_path(model_id))
        }
        self._save_cache_info()
    
    def validate_cached_model(self, model_id: str) -> bool:
        """Validate that a cached model is complete and not corrupted"""
        if not self.is_model_cached(model_id):
            return False
        
        model_path = self.get_model_path(model_id)
        required_files = ["config.json"]
        
        # Check for required files
        for file in required_files:
            if not (model_path / file).exists():
                logger.warning(f"Missing required file {file} for model {model_id}")
                return False
        
        # Check if model has pytorch_model.bin or model.safetensors
        has_weights = any([
            (model_path / "pytorch_model.bin").exists(),
            (model_path / "model.safetensors").exists(),
            any(model_path.glob("pytorch_model-*.bin")),
            any(model_path.glob("model-*.safetensors"))
        ])
        
        if not has_weights:
            logger.warning(f"No model weights found for {model_id}")
            return False
        
        return True

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
        
        # Model ID mappings - Updated to use correct Hugging Face repository names
        self.model_mappings = {
            "t2v-A14B": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "i2v-A14B": "Wan-AI/Wan2.2-I2V-A14B-Diffusers", 
            "ti2v-5B": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
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
  
    @handle_error_with_recovery
    def download_model(self, model_id: str, force_download: bool = False, timeout: int = 300) -> str:
        """Download a model from Hugging Face Hub with enhanced error handling"""
        full_model_id = self.get_model_id(model_id)
        
        # Check if already cached and valid
        if not force_download and self.cache.is_model_cached(full_model_id):
            if self.cache.validate_cached_model(full_model_id):
                logger.info(f"Using cached model: {full_model_id}")
                return str(self.cache.get_model_path(full_model_id))
            else:
                logger.warning(f"Cached model {full_model_id} is invalid, re-downloading")
        
        logger.info(f"Downloading model: {full_model_id} (timeout: {timeout}s)")
        
        try:
            # Check available disk space before download
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 10:  # Require at least 10GB free space
                raise RuntimeError(f"Insufficient disk space: {free_gb:.1f}GB free, need at least 10GB")
            
            # Fix tqdm compatibility issue by disabling progress bars and using fallback
            import os
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            
            # Try to fix tqdm lock issue
            try:
                import tqdm
                if not hasattr(tqdm.tqdm, '_lock'):
                    import threading
                    tqdm.tqdm._lock = threading.RLock()
            except Exception as tqdm_fix_error:
                logger.debug(f"Could not fix tqdm: {tqdm_fix_error}")
            
            # Use snapshot_download for complete model download with multiple fallback strategies
            download_strategies = [
                # Strategy 1: Standard download with reduced workers
                {
                    "repo_id": full_model_id,
                    "cache_dir": str(self.cache.cache_dir),
                    "local_dir": str(self.cache.get_model_path(full_model_id)),
                    "local_dir_use_symlinks": False,
                    "resume_download": True,
                    "max_workers": 1
                },
                # Strategy 2: Disable tqdm completely
                {
                    "repo_id": full_model_id,
                    "cache_dir": str(self.cache.cache_dir),
                    "local_dir": str(self.cache.get_model_path(full_model_id)),
                    "local_dir_use_symlinks": False,
                    "resume_download": True,
                    "max_workers": 1,
                    "tqdm_class": None
                },
                # Strategy 3: Use cache_dir only (no local_dir)
                {
                    "repo_id": full_model_id,
                    "cache_dir": str(self.cache.cache_dir),
                    "resume_download": True,
                    "max_workers": 1
                }
            ]
            
            model_path = None
            last_error = None
            
            for i, strategy in enumerate(download_strategies):
                try:
                    logger.info(f"Trying download strategy {i+1}/{len(download_strategies)}")
                    model_path = snapshot_download(**strategy)
                    logger.info(f"Download successful with strategy {i+1}")
                    break
                except Exception as download_error:
                    last_error = download_error
                    logger.warning(f"Download strategy {i+1} failed: {download_error}")
                    if i < len(download_strategies) - 1:
                        logger.info(f"Trying next strategy...")
                    continue
            
            if model_path is None:
                raise Exception(f"All download strategies failed. Last error: {last_error}")
            
            # Update cache info
            model_type = self.detect_model_type(full_model_id)
            self.cache.update_cache_info(full_model_id, {
                "model_type": model_type,
                "download_date": datetime.now().isoformat(),
                "model_id": full_model_id,
                "local_path": model_path
            })
            
            logger.info(f"Successfully downloaded model: {full_model_id}")
            return model_path
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                error_msg = f"Model {full_model_id} not found on Hugging Face Hub"
                log_error_with_context(e, "model_download", {"model_id": full_model_id, "status_code": 404})
                raise ValueError(error_msg)
            else:
                log_error_with_context(e, "model_download", {"model_id": full_model_id, "status_code": e.response.status_code})
                raise
        except Exception as e:
            log_error_with_context(e, "model_download", {"model_id": full_model_id, "disk_free_gb": free_gb if 'free_gb' in locals() else "unknown"})
            raise
    
    @handle_error_with_recovery
    def load_model(self, model_type: str, progress_callback: Optional[Callable[[str, float], None]] = None, **kwargs) -> Tuple[Any, ModelInfo]:
        """Load a model with compatibility detection and optimization"""
        model_id = self.get_model_id(model_type)
        
        # Check if model is already loaded
        if model_id in self.loaded_models:
            logger.info(f"Using already loaded model: {model_id}")
            if progress_callback:
                progress_callback("Using cached model", 100.0)
            return self.loaded_models[model_id], self.model_info[model_id]
        
        if progress_callback:
            progress_callback("Checking system resources", 5.0)
        
        # Check VRAM availability before loading
        if torch.cuda.is_available():
            vram_info = self._get_vram_info()
            if vram_info["free_mb"] < 2000:  # Need at least 2GB free
                logger.warning(f"Low VRAM: {vram_info['free_mb']:.0f}MB free")
                # Try to free some memory
                torch.cuda.empty_cache()
        
        if progress_callback:
            progress_callback("Downloading model if needed", 15.0)
        
        # Ensure model is downloaded
        model_path = self.download_model(model_id)
        
        logger.info(f"Loading model from: {model_path}")
        
        try:
            if progress_callback:
                progress_callback("Running compatibility check", 25.0)
            
            # Detect model architecture first with progress reporting
            def compatibility_progress(stage: str, percent: float):
                # Map compatibility check progress to overall progress (25-45%)
                overall_percent = 25.0 + (percent / 100.0) * 20.0
                if progress_callback:
                    progress_callback(f"Compatibility: {stage}", overall_percent)
            
            compatibility_status = self.check_model_compatibility(model_path, compatibility_progress)
            
            if progress_callback:
                progress_callback("Loading pipeline", 50.0)
            
            # Load pipeline based on enhanced compatibility detection
            if compatibility_status["is_wan_model"]:
                logger.info(f"Detected Wan model ({compatibility_status['architecture_type']}) - using WanPipelineLoader")
                
                def wan_loading_progress(stage: str, percent: float):
                    # Map Wan loading progress to overall progress (50-90%)
                    overall_percent = 50.0 + (percent / 100.0) * 40.0
                    if progress_callback:
                        progress_callback(f"Wan Pipeline: {stage}", overall_percent)
                
                # Use enhanced Wan model loading with architecture-specific optimizations
                pipeline_wrapper = self._load_wan_model_with_enhanced_detection(
                    model_path, compatibility_status, wan_loading_progress, **kwargs
                )
                pipeline = pipeline_wrapper.pipeline
                memory_usage = compatibility_status.get("estimated_memory_mb", 8192)
                
                # Update compatibility status with actual loading results
                compatibility_status["optimization_applied"] = True
                compatibility_status["applied_optimizations"] = getattr(pipeline_wrapper, 'applied_optimizations', [])
                
            else:
                logger.info(f"Detected standard model ({compatibility_status['architecture_type']}) - using fallback loading")
                if progress_callback:
                    progress_callback("Loading standard pipeline", 70.0)
                
                # Use architecture-aware fallback loading
                pipeline = self._load_pipeline_with_architecture_detection(
                    model_path, compatibility_status, **kwargs
                )
                memory_usage = self._calculate_model_memory(pipeline)
            
            if progress_callback:
                progress_callback("Finalizing model setup", 95.0)
            
            # Create model info with compatibility data
            model_info = ModelInfo(
                model_type=compatibility_status.get("architecture_type", self.detect_model_type(model_id)),
                model_path=model_path,
                model_id=model_id,
                loaded_at=datetime.now(),
                memory_usage_mb=memory_usage
            )
            
            # Store compatibility information in model info
            if hasattr(model_info, '__dict__'):
                model_info.__dict__.update({
                    'compatibility_status': compatibility_status,
                    'optimization_applied': compatibility_status.get("optimization_applied", False),
                    'applied_optimizations': compatibility_status.get("applied_optimizations", [])
                })
            
            # Cache the loaded model
            self.loaded_models[model_id] = pipeline
            self.model_info[model_id] = model_info
            
            if progress_callback:
                progress_callback("Model loaded successfully", 100.0)
            
            logger.info(f"Successfully loaded model: {model_id} ({memory_usage:.1f} MB)")
            return pipeline, model_info
            
        except torch.cuda.OutOfMemoryError as e:
            log_error_with_context(e, "model_loading", {"model_id": model_id, "vram_info": vram_info if 'vram_info' in locals() else None})
            if progress_callback:
                progress_callback("Failed: Out of memory", 100.0)
            raise
        except Exception as e:
            log_error_with_context(e, "model_loading", {"model_id": model_id, "model_path": model_path})
            if progress_callback:
                progress_callback(f"Failed: {str(e)}", 100.0)
            raise
    
    def _load_pipeline_with_architecture_detection(self, model_path: str, compatibility_status: Dict[str, Any], **kwargs):
        """Load pipeline with architecture-aware fallback"""
        
        # Force local files only to prevent downloading
        kwargs['local_files_only'] = True
        
        # Clean up kwargs to remove problematic parameters
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['boundary_ratio', 'custom_pipeline', 'progress_callback']}
        
        # Get architecture information for better loading decisions
        architecture_type = compatibility_status.get("architecture_type", "unknown")
        model_architecture = compatibility_status.get("model_architecture")
        
        # Add parameters based on architecture detection
        model_loading_params = {
            'trust_remote_code': compatibility_status.get("requires_trust_remote_code", True),
            'low_cpu_mem_usage': False,
            'ignore_mismatched_sizes': True,
            'safety_checker': None,
            'requires_safety_checker': False
        }
        
        # Apply architecture-specific optimizations
        if model_architecture and hasattr(model_architecture, 'requirements'):
            if model_architecture.requirements.supports_mixed_precision:
                model_loading_params['torch_dtype'] = torch.bfloat16
            
        # Don't override existing parameters
        for key, value in model_loading_params.items():
            if key not in clean_kwargs:
                clean_kwargs[key] = value
        
        logger.info(f"Attempting to load pipeline from: {model_path} (architecture: {architecture_type})")
        
        try:
            # Try architecture-specific loading first
            if architecture_type.startswith("wan"):
                try:
                    from diffusers import WanPipeline
                    logger.info("Using WanPipeline for Wan architecture")
                    return WanPipeline.from_pretrained(model_path, **clean_kwargs)
                except (ImportError, AttributeError) as e:
                    logger.warning(f"WanPipeline not available: {e}")
                    compatibility_status["compatibility_warnings"].append("WanPipeline not available, using fallback")
            
            # Try standard DiffusionPipeline
            logger.info("Trying standard DiffusionPipeline")
            return DiffusionPipeline.from_pretrained(model_path, **clean_kwargs)
            
        except Exception as e2:
            logger.warning(f"Standard DiffusionPipeline failed: {e2}")
            compatibility_status["compatibility_warnings"].append(f"Standard pipeline loading failed: {str(e2)}")
            
            try:
                # Final fallback to StableDiffusionPipeline
                from diffusers import StableDiffusionPipeline
                logger.info("Using StableDiffusionPipeline as final fallback")
                compatibility_status["compatibility_warnings"].append("Using StableDiffusion fallback - functionality may be limited")
                return StableDiffusionPipeline.from_pretrained(model_path, **clean_kwargs)
                
            except Exception as e3:
                logger.error(f"All pipeline loading methods failed: {e3}")
                compatibility_status["compatibility_errors"].append(f"All loading methods failed: {str(e3)}")
                raise e3
    
    def _load_pipeline_with_fallback(self, model_path: str, **kwargs):
        """Legacy fallback method - kept for backward compatibility"""
        # Create minimal compatibility status for legacy calls
        compatibility_status = {
            "architecture_type": "unknown",
            "requires_trust_remote_code": True,
            "compatibility_warnings": [],
            "compatibility_errors": []
        }
        
        return self._load_pipeline_with_architecture_detection(model_path, compatibility_status, **kwargs)
    
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
            
            # Estimate memory usage (parameters * 4 bytes for fp32, adjust for actual dtype)
            memory_bytes = total_params * 2  # Assuming bf16/fp16 (2 bytes per parameter)
            return memory_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Failed to calculate model memory: {e}")
            return 0.0
    
    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        full_model_id = self.get_model_id(model_id)
        
        if full_model_id in self.loaded_models:
            del self.loaded_models[full_model_id]
            del self.model_info[full_model_id]
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model: {full_model_id}")
    
    def list_cached_models(self) -> Dict[str, Dict[str, Any]]:
        """List all cached models with their information"""
        cached_models = {}
        
        for model_id, info in self.cache.cache_info.items():
            if self.cache.validate_cached_model(model_id):
                cached_models[model_id] = {
                    **info,
                    "is_loaded": model_id in self.loaded_models,
                    "size_mb": self._get_model_size_mb(model_id)
                }
        
        return cached_models
    
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
    
    def clear_cache(self, model_id: Optional[str] = None):
        """Clear model cache (specific model or all models)"""
        if model_id:
            full_model_id = self.get_model_id(model_id)
            model_path = self.cache.get_model_path(full_model_id)
            
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
                logger.info(f"Cleared cache for model: {full_model_id}")
            
            # Remove from cache info
            if full_model_id in self.cache.cache_info:
                del self.cache.cache_info[full_model_id]
                self.cache._save_cache_info()
        else:
            # Clear all cache
            import shutil
            if self.cache.cache_dir.exists():
                shutil.rmtree(self.cache.cache_dir)
                self.cache.cache_dir.mkdir(exist_ok=True)
            
            self.cache.cache_info = {}
            self.cache._save_cache_info()
            logger.info("Cleared all model cache")
    
    def check_model_compatibility(self, model_path: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """Check model compatibility and return detailed status with progress reporting"""
        try:
            # Check cache first
            if model_path in self._compatibility_cache:
                logger.debug(f"Using cached compatibility status for {model_path}")
                if progress_callback:
                    progress_callback("Using cached compatibility data", 100.0)
                return self._compatibility_cache[model_path]
            
            logger.info(f"Checking compatibility for model at: {model_path}")
            if progress_callback:
                progress_callback("Starting compatibility check", 10.0)
            
            # Detect model architecture using enhanced detection system
            if progress_callback:
                progress_callback("Detecting model architecture", 30.0)
            model_architecture = self.architecture_detector.detect_model_architecture(model_path)
            
            # Determine if this is a Wan model
            is_wan_model = model_architecture.architecture_type in [
                ArchitectureType.WAN_T2V, ArchitectureType.WAN_T2I, ArchitectureType.WAN_I2V
            ]
            
            if progress_callback:
                progress_callback("Analyzing system resources", 50.0)
            
            # Get optimization recommendations
            system_resources = self.optimization_manager.analyze_system_resources()
            
            if progress_callback:
                progress_callback("Generating optimization recommendations", 70.0)
            
            optimization_plan = self.optimization_manager.recommend_optimizations(
                model_architecture.requirements, system_resources
            )
            
            if progress_callback:
                progress_callback("Finalizing compatibility report", 90.0)
            
            # Create enhanced compatibility status with UI integration data
            compatibility_status = {
                "is_wan_model": is_wan_model,
                "architecture_type": model_architecture.architecture_type.value,
                "model_architecture": model_architecture,
                "requires_trust_remote_code": model_architecture.requirements.requires_trust_remote_code,
                "min_vram_mb": model_architecture.requirements.min_vram_mb,
                "recommended_vram_mb": model_architecture.requirements.recommended_vram_mb,
                "estimated_memory_mb": getattr(model_architecture.requirements, 'model_size_mb', 8192),
                "supports_optimizations": {
                    "mixed_precision": model_architecture.requirements.supports_mixed_precision,
                    "cpu_offload": model_architecture.requirements.supports_cpu_offload,
                    "chunked_processing": True  # Wan models support chunked processing
                },
                "optimization_plan": optimization_plan,
                "optimization_applied": False,
                "applied_optimizations": [],
                "compatibility_warnings": [],
                "compatibility_errors": [],
                "system_resources": {
                    "available_vram_mb": system_resources.available_vram_mb,
                    "total_vram_mb": system_resources.total_vram_mb,
                    "available_ram_mb": system_resources.available_ram_mb,
                    "gpu_name": getattr(system_resources, 'gpu_name', 'Unknown')
                },
                # UI integration data
                "ui_status": {
                    "compatibility_level": self._get_compatibility_level(model_architecture, system_resources),
                    "user_friendly_status": self._get_user_friendly_status(is_wan_model, model_architecture, system_resources),
                    "progress_indicators": self._get_progress_indicators(optimization_plan),
                    "recommended_actions": self._get_recommended_actions(model_architecture, system_resources, optimization_plan)
                }
            }
            
            # Add warnings based on system resources
            if system_resources.available_vram_mb < model_architecture.requirements.min_vram_mb:
                compatibility_status["compatibility_warnings"].append(
                    f"Insufficient VRAM: {system_resources.available_vram_mb}MB available, "
                    f"{model_architecture.requirements.min_vram_mb}MB required"
                )
            
            if is_wan_model and not model_architecture.requirements.requires_trust_remote_code:
                compatibility_status["compatibility_warnings"].append(
                    "Wan model may require trust_remote_code=True for proper loading"
                )
            
            # Add optimization recommendations as warnings
            if optimization_plan:
                if optimization_plan.use_mixed_precision:
                    compatibility_status["compatibility_warnings"].append(
                        f"Recommended: Use {optimization_plan.precision_type} precision for better performance"
                    )
                if optimization_plan.enable_cpu_offload:
                    compatibility_status["compatibility_warnings"].append(
                        "Recommended: Enable CPU offloading to reduce VRAM usage"
                    )
                if optimization_plan.chunk_frames:
                    compatibility_status["compatibility_warnings"].append(
                        f"Recommended: Use chunked processing (max {optimization_plan.max_chunk_size} frames)"
                    )
            
            # Cache the result for future use
            self._compatibility_cache[model_path] = compatibility_status
            
            if progress_callback:
                progress_callback("Compatibility check complete", 100.0)
            
            return compatibility_status
            
        except Exception as e:
            logger.error(f"Compatibility check failed for {model_path}: {e}")
            error_status = {
                "is_wan_model": False,
                "architecture_type": "unknown",
                "compatibility_errors": [f"Compatibility check failed: {str(e)}"],
                "compatibility_warnings": [],
                "ui_status": {
                    "compatibility_level": "error",
                    "user_friendly_status": f"❌ Compatibility check failed: {str(e)}",
                    "progress_indicators": [],
                    "recommended_actions": ["Check model files", "Verify model format", "Try re-downloading model"]
                }
            }
            if progress_callback:
                progress_callback("Compatibility check failed", 100.0)
            return error_status
            
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            if progress_callback:
                progress_callback(f"Compatibility check failed: {str(e)}", 100.0)
            
            # Return fallback compatibility status
            return {
                "is_wan_model": False,
                "architecture_type": "unknown",
                "model_architecture": None,
                "requires_trust_remote_code": False,
                "min_vram_mb": 4096,
                "recommended_vram_mb": 8192,
                "estimated_memory_mb": 8192,
                "supports_optimizations": {
                    "mixed_precision": True,
                    "cpu_offload": True,
                    "chunked_processing": False
                },
                "optimization_plan": None,
                "optimization_applied": False,
                "applied_optimizations": [],
                "compatibility_warnings": [f"Compatibility check failed: {str(e)}"],
                "compatibility_errors": [str(e)],
                "system_resources": {
                    "available_vram_mb": 0,
                    "total_vram_mb": 0,
                    "available_ram_mb": 0,
                    "gpu_name": "Unknown"
                },
                "ui_status": {
                    "compatibility_level": "error",
                    "user_friendly_status": f"❌ Compatibility check failed: {str(e)}",
                    "progress_indicators": [],
                    "recommended_actions": ["Check model files", "Verify model format", "Try re-downloading model"]
                }
            }
    
    def _load_wan_model_with_enhanced_detection(self, model_path: str, 
                                        compatibility_status: Dict[str, Any],
                                        progress_callback: Optional[Callable[[str, float], None]] = None,
                                        **kwargs) -> Any:
        """Load Wan model using WanPipelineLoader with enhanced architecture detection"""
        try:
            if progress_callback:
                progress_callback("Preparing optimization config", 10.0)
            
            # Get model architecture for enhanced loading
            model_architecture = compatibility_status.get("model_architecture")
            
            # Prepare optimization config from compatibility status and architecture
            optimization_config = {}
            
            if compatibility_status.get("optimization_plan"):
                plan = compatibility_status["optimization_plan"]
                optimization_config = {
                    "precision": plan.precision_type if plan.use_mixed_precision else "fp32",
                    "enable_cpu_offload": plan.enable_cpu_offload,
                    "chunk_size": plan.max_chunk_size if plan.chunk_frames else None,
                    "architecture_type": compatibility_status["architecture_type"]
                }
            
            if progress_callback:
                progress_callback("Loading Wan pipeline", 30.0)
            
            # Load pipeline with architecture-specific optimizations
            pipeline_wrapper = self.wan_pipeline_loader.load_wan_pipeline(
                model_path=model_path,
                trust_remote_code=compatibility_status.get("requires_trust_remote_code", True),
                apply_optimizations=True,
                optimization_config=optimization_config,
                model_architecture=model_architecture,
                progress_callback=progress_callback,
                **kwargs
            )
            
            if progress_callback:
                progress_callback("Applying optimizations", 80.0)
            
            # Update compatibility status with applied optimizations
            if hasattr(pipeline_wrapper, 'optimization_result'):
                optimization_result = pipeline_wrapper.optimization_result
                compatibility_status["optimization_applied"] = optimization_result.success
                compatibility_status["applied_optimizations"] = optimization_result.applied_optimizations
                
                if optimization_result.warnings:
                    compatibility_status["compatibility_warnings"].extend(optimization_result.warnings)
                
                if optimization_result.errors:
                    compatibility_status["compatibility_errors"].extend(optimization_result.errors)
            
            if progress_callback:
                progress_callback("Wan model loaded successfully", 100.0)
            
            return pipeline_wrapper
            
        except Exception as e:
            logger.error(f"Failed to load Wan model with enhanced detection: {e}")
            compatibility_status["compatibility_errors"].append(f"Wan model loading failed: {str(e)}")
            if progress_callback:
                progress_callback(f"Loading failed: {str(e)}", 100.0)
            raise
    
    def get_optimization_recommendations(self, model_id: str) -> List[str]:
        """Get optimization recommendations for a model"""
        try:
            full_model_id = self.get_model_id(model_id)
            model_path = self.cache.get_model_path(full_model_id)
            
            if not model_path.exists():
                return ["Model not found - download required"]
            
            # Check compatibility to get optimization recommendations
            compatibility_status = self.check_model_compatibility(str(model_path))
            
            recommendations = []
            
            if compatibility_status.get("optimization_plan"):
                plan = compatibility_status["optimization_plan"]
                
                if plan.use_mixed_precision:
                    recommendations.append(f"Enable {plan.precision_type} mixed precision for {plan.estimated_vram_reduction*100:.0f}% VRAM reduction")
                
                if plan.enable_cpu_offload:
                    recommendations.append(f"Enable {plan.offload_strategy} CPU offloading")
                
                if plan.chunk_frames:
                    recommendations.append(f"Use chunked processing with {plan.max_chunk_size} frames per chunk")
                
                for warning in plan.warnings:
                    recommendations.append(f"⚠️ {warning}")
            
            # Add general recommendations
            if compatibility_status["is_wan_model"]:
                recommendations.append("Use WanPipeline for optimal performance")
                
                if compatibility_status["requires_trust_remote_code"]:
                    recommendations.append("Enable trust_remote_code=True for proper loading")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return [f"Error getting recommendations: {str(e)}"]
    
    def _get_compatibility_level(self, model_architecture: 'ModelArchitecture', system_resources: 'SystemResources') -> str:
        """Get compatibility level for UI display"""
        if system_resources.available_vram_mb >= model_architecture.requirements.recommended_vram_mb:
            return "excellent"
        elif system_resources.available_vram_mb >= model_architecture.requirements.min_vram_mb:
            return "good"
        elif system_resources.available_vram_mb >= model_architecture.requirements.min_vram_mb * 0.7:
            return "limited"
        else:
            return "insufficient"
    
    def get_compatibility_status_for_ui(self, model_id: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """Get compatibility status formatted for UI display with progress reporting"""
        try:
            full_model_id = self.get_model_id(model_id)
            
            if progress_callback:
                progress_callback("Checking model cache", 10.0)
            
            # Check if model is cached
            if not self.cache.is_model_cached(full_model_id):
                return {
                    "status": "not_cached",
                    "message": "Model not downloaded",
                    "level": "info",
                    "actions": ["Download model"],
                    "progress_indicators": [],
                    "compatibility_details": {}
                }
            
            if progress_callback:
                progress_callback("Validating model files", 20.0)
            
            # Validate cached model
            if not self.cache.validate_cached_model(full_model_id):
                return {
                    "status": "invalid",
                    "message": "Model files are corrupted or incomplete",
                    "level": "error",
                    "actions": ["Re-download model", "Clear cache and retry"],
                    "progress_indicators": [],
                    "compatibility_details": {}
                }
            
            if progress_callback:
                progress_callback("Running compatibility check", 40.0)
            
            # Get detailed compatibility status
            model_path = self.cache.get_model_path(full_model_id)
            compatibility_status = self.check_model_compatibility(str(model_path), progress_callback)
            
            if progress_callback:
                progress_callback("Formatting UI status", 90.0)
            
            # Format for UI display
            ui_status = {
                "status": "compatible" if not compatibility_status.get("compatibility_errors") else "error",
                "message": compatibility_status.get("ui_status", {}).get("user_friendly_status", "Unknown status"),
                "level": compatibility_status.get("ui_status", {}).get("compatibility_level", "unknown"),
                "actions": compatibility_status.get("ui_status", {}).get("recommended_actions", []),
                "progress_indicators": compatibility_status.get("ui_status", {}).get("progress_indicators", []),
                "compatibility_details": {
                    "is_wan_model": compatibility_status.get("is_wan_model", False),
                    "architecture_type": compatibility_status.get("architecture_type", "unknown"),
                    "requires_trust_remote_code": compatibility_status.get("requires_trust_remote_code", False),
                    "min_vram_mb": compatibility_status.get("min_vram_mb", 0),
                    "recommended_vram_mb": compatibility_status.get("recommended_vram_mb", 0),
                    "system_vram_mb": compatibility_status.get("system_resources", {}).get("available_vram_mb", 0),
                    "supports_optimizations": compatibility_status.get("supports_optimizations", {}),
                    "optimization_applied": compatibility_status.get("optimization_applied", False),
                    "applied_optimizations": compatibility_status.get("applied_optimizations", []),
                    "warnings": compatibility_status.get("compatibility_warnings", []),
                    "errors": compatibility_status.get("compatibility_errors", [])
                }
            }
            
            if progress_callback:
                progress_callback("Compatibility check complete", 100.0)
            
            return ui_status
            
        except Exception as e:
            logger.error(f"Failed to get UI compatibility status: {e}")
            if progress_callback:
                progress_callback("Compatibility check failed", 100.0)
            
            return {
                "status": "error",
                "message": f"Compatibility check failed: {str(e)}",
                "level": "error",
                "actions": ["Check model files", "Try re-downloading"],
                "progress_indicators": [],
                "compatibility_details": {"errors": [str(e)]}
            }
    
    def get_optimization_status_for_ui(self, model_id: str) -> Dict[str, Any]:
        """Get optimization status formatted for UI display"""
        try:
            full_model_id = self.get_model_id(model_id)
            
            # Check if model is loaded
            if full_model_id not in self.loaded_models:
                return {
                    "status": "not_loaded",
                    "message": "Model not loaded",
                    "optimizations": [],
                    "recommendations": []
                }
            
            # Get model info with optimization details
            model_info = self.model_info.get(full_model_id)
            if not model_info:
                return {
                    "status": "no_info",
                    "message": "No optimization information available",
                    "optimizations": [],
                    "recommendations": []
                }
            
            # Extract optimization information
            compatibility_status = getattr(model_info, 'compatibility_status', {})
            applied_optimizations = getattr(model_info, 'applied_optimizations', [])
            
            # Format optimization status
            optimization_status = []
            for opt in applied_optimizations:
                optimization_status.append({
                    "name": opt,
                    "status": "active",
                    "description": self._get_optimization_description(opt)
                })
            
            # Get recommendations
            recommendations = self.get_optimization_recommendations(model_id)
            
            return {
                "status": "optimized" if applied_optimizations else "not_optimized",
                "message": f"{len(applied_optimizations)} optimizations active" if applied_optimizations else "No optimizations applied",
                "optimizations": optimization_status,
                "recommendations": recommendations,
                "memory_usage_mb": model_info.memory_usage_mb,
                "is_quantized": getattr(model_info, 'is_quantized', False),
                "is_offloaded": getattr(model_info, 'is_offloaded', False)
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}")
            return {
                "status": "error",
                "message": f"Error getting optimization status: {str(e)}",
                "optimizations": [],
                "recommendations": []
            }
    
    def _get_optimization_description(self, optimization_name: str) -> str:
        """Get user-friendly description for optimization"""
        descriptions = {
            "mixed_precision": "Using reduced precision (fp16/bf16) to save VRAM",
            "cpu_offload": "Offloading model components to CPU to reduce VRAM usage",
            "chunked_processing": "Processing video in smaller chunks to manage memory",
            "sequential_offload": "Moving model components between GPU and CPU as needed",
            "model_offload": "Keeping model on CPU until needed for generation",
            "vae_tiling": "Processing VAE in tiles to reduce memory usage",
            "attention_slicing": "Using attention slicing to reduce memory usage"
        }
        return descriptions.get(optimization_name, f"Applied {optimization_name} optimization")
    
    def apply_optimization_recommendations(self, model_id: str, optimizations: List[str], progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """Apply optimization recommendations to a loaded model"""
        try:
            full_model_id = self.get_model_id(model_id)
            
            if progress_callback:
                progress_callback("Checking model status", 10.0)
            
            # Check if model is loaded
            if full_model_id not in self.loaded_models:
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "applied_optimizations": []
                }
            
            pipeline = self.loaded_models[full_model_id]
            model_info = self.model_info[full_model_id]
            
            if progress_callback:
                progress_callback("Applying optimizations", 30.0)
            
            applied_optimizations = []
            errors = []
            
            # Apply each optimization
            for i, optimization in enumerate(optimizations):
                try:
                    if progress_callback:
                        progress = 30.0 + (i / len(optimizations)) * 60.0
                        progress_callback(f"Applying {optimization}", progress)
                    
                    if optimization == "mixed_precision":
                        if hasattr(pipeline, 'to'):
                            pipeline.to(dtype=torch.bfloat16)
                            applied_optimizations.append("mixed_precision")
                    
                    elif optimization == "cpu_offload":
                        if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                            pipeline.enable_sequential_cpu_offload()
                            applied_optimizations.append("cpu_offload")
                    
                    elif optimization == "attention_slicing":
                        if hasattr(pipeline, 'enable_attention_slicing'):
                            pipeline.enable_attention_slicing()
                            applied_optimizations.append("attention_slicing")
                    
                    elif optimization == "vae_tiling":
                        if hasattr(pipeline, 'enable_vae_tiling'):
                            pipeline.enable_vae_tiling()
                            applied_optimizations.append("vae_tiling")
                    
                except Exception as e:
                    logger.warning(f"Failed to apply {optimization}: {e}")
                    errors.append(f"Failed to apply {optimization}: {str(e)}")
            
            if progress_callback:
                progress_callback("Updating model info", 95.0)
            
            # Update model info with applied optimizations
            if hasattr(model_info, '__dict__'):
                current_optimizations = getattr(model_info, 'applied_optimizations', [])
                model_info.__dict__['applied_optimizations'] = list(set(current_optimizations + applied_optimizations))
                
                # Update optimization status
                if applied_optimizations:
                    model_info.__dict__['optimization_applied'] = True
            
            if progress_callback:
                progress_callback("Optimizations applied", 100.0)
            
            return {
                "success": True,
                "applied_optimizations": applied_optimizations,
                "errors": errors,
                "total_applied": len(applied_optimizations)
            }
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            if progress_callback:
                progress_callback("Optimization failed", 100.0)
            
            return {
                "success": False,
                "error": str(e),
                "applied_optimizations": []
            }
    
    def _get_user_friendly_status(self, is_wan_model: bool, model_architecture: 'ModelArchitecture', system_resources: 'SystemResources') -> str:
        """Get user-friendly status message"""
        if not is_wan_model:
            return "✅ Standard model - compatible with basic pipeline"
        
        compatibility_level = self._get_compatibility_level(model_architecture, system_resources)
        
        if compatibility_level == "excellent":
            return "🚀 Wan model - excellent compatibility, full performance expected"
        elif compatibility_level == "good":
            return "✅ Wan model - good compatibility, optimizations recommended"
        elif compatibility_level == "limited":
            return "⚠️ Wan model - limited compatibility, optimizations required"
        else:
            return "❌ Wan model - insufficient resources, may not work properly"
    
    def _get_progress_indicators(self, optimization_plan: Optional['OptimizationPlan']) -> List[Dict[str, Any]]:
        """Get progress indicators for UI display"""
        indicators = []
        
        if optimization_plan:
            if optimization_plan.use_mixed_precision:
                indicators.append({
                    "name": "Mixed Precision",
                    "status": "recommended",
                    "description": f"Use {optimization_plan.precision_type} for {optimization_plan.estimated_vram_reduction*100:.0f}% VRAM reduction"
                })
            
            if optimization_plan.enable_cpu_offload:
                indicators.append({
                    "name": "CPU Offload",
                    "status": "recommended", 
                    "description": f"Enable {optimization_plan.offload_strategy} offloading"
                })
            
            if optimization_plan.chunk_frames:
                indicators.append({
                    "name": "Chunked Processing",
                    "status": "required",
                    "description": f"Process in chunks of {optimization_plan.max_chunk_size} frames"
                })
        
        return indicators
    
    def _get_recommended_actions(self, model_architecture: 'ModelArchitecture', system_resources: 'SystemResources', optimization_plan: Optional['OptimizationPlan']) -> List[str]:
        """Get recommended actions for the user"""
        actions = []
        
        if system_resources.available_vram_mb < model_architecture.requirements.min_vram_mb:
            actions.append("Close other GPU applications to free VRAM")
            actions.append("Enable CPU offloading in optimization settings")
        
        if optimization_plan and optimization_plan.use_mixed_precision:
            actions.append(f"Enable {optimization_plan.precision_type} precision for better performance")
        
        if model_architecture.requirements.requires_trust_remote_code:
            actions.append("Ensure trust_remote_code is enabled for custom pipeline loading")
        
        if not actions:
            actions.append("Model is ready to use with current settings")
        
        return actions

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
            "optimization_recommendations": []
        }
        
        if status["is_cached"]:
            status["is_valid"] = self.cache.validate_cached_model(full_model_id)
            status["cache_info"] = self.cache.get_cache_info(full_model_id)
            status["size_mb"] = self._get_model_size_mb(full_model_id)
            
            # Get compatibility status for cached models
            model_path = self.cache.get_model_path(full_model_id)
            if model_path.exists():
                status["compatibility_status"] = self.check_model_compatibility(str(model_path))
                status["optimization_recommendations"] = self.get_optimization_recommendations(model_id)
        
        if status["is_loaded"]:
            status["model_info"] = self.model_info[full_model_id]
        
        return status


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

# UI Integration Functions
def get_compatibility_status_for_ui(model_id: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """Get compatibility status formatted for UI display with progress reporting"""
    manager = get_model_manager()
    return manager.get_compatibility_status_for_ui(model_id, progress_callback)

def get_optimization_status_for_ui(model_id: str) -> Dict[str, Any]:
    """Get optimization status formatted for UI display"""
    manager = get_model_manager()
    return manager.get_optimization_status_for_ui(model_id)

def apply_optimization_recommendations(model_id: str, optimizations: List[str], progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """Apply optimization recommendations to a loaded model"""
    manager = get_model_manager()
    return manager.apply_optimization_recommendations(model_id, optimizations, progress_callback)

def check_model_compatibility_for_ui(model_id: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """Check model compatibility with progress reporting for UI"""
    manager = get_model_manager()
    full_model_id = manager.get_model_id(model_id)
    
    # Check if model is cached
    if not manager.cache.is_model_cached(full_model_id):
        if progress_callback:
            progress_callback("Model not cached", 100.0)
        return {
            "is_wan_model": False,
            "architecture_type": "unknown",
            "compatibility_status": "not_cached",
            "ui_message": "Model needs to be downloaded first"
        }
    
    # Get model path and check compatibility
    model_path = manager.cache.get_model_path(full_model_id)
    return manager.check_model_compatibility(str(model_path), progress_callback)

def get_model_loading_progress_info(model_id: str) -> Dict[str, Any]:
    """Get information for displaying model loading progress"""
    manager = get_model_manager()
    full_model_id = manager.get_model_id(model_id)
    
    # Check current status
    is_cached = manager.cache.is_model_cached(full_model_id)
    is_loaded = full_model_id in manager.loaded_models
    
    progress_info = {
        "model_id": full_model_id,
        "is_cached": is_cached,
        "is_loaded": is_loaded,
        "estimated_steps": [],
        "current_step": 0,
        "total_steps": 0
    }
    
    # Define loading steps based on model status
    if not is_cached:
        progress_info["estimated_steps"] = [
            "Download model files",
            "Validate model integrity", 
            "Check compatibility",
            "Load pipeline",
            "Apply optimizations",
            "Finalize setup"
        ]
    elif not is_loaded:
        progress_info["estimated_steps"] = [
            "Check compatibility",
            "Load pipeline", 
            "Apply optimizations",
            "Finalize setup"
        ]
    else:
        progress_info["estimated_steps"] = ["Model already loaded"]
        progress_info["current_step"] = 1
    
    progress_info["total_steps"] = len(progress_info["estimated_steps"])
    
    return progress_info

# Enhanced Generation Pipeline Integration
def get_enhanced_generation_pipeline(config: Dict[str, Any] = None):
    """Get the enhanced generation pipeline instance"""
    if config is None:
        # Load default config
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            config = {
                "directories": {"models_directory": "models", "outputs_directory": "outputs", "loras_directory": "loras"},
                "optimization": {"max_vram_usage_gb": 12},
                "generation": {"max_retry_attempts": 3, "enable_auto_optimization": True}
            }
    
    try:
        from enhanced_generation_pipeline import get_enhanced_pipeline
        return get_enhanced_pipeline(config)
    except ImportError as e:
        logger.warning(f"Enhanced pipeline not available: {e}")
        return None

def get_generation_mode_router(config: Dict[str, Any] = None):
    """Get the generation mode router instance"""
    if config is None:
        # Load default config
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            config = {"generation": {"max_prompt_length": 512}}
    
    try:
        from generation_mode_router import get_generation_mode_router as get_router
        return get_router(config)
    except ImportError as e:
        logger.warning(f"Generation mode router not available: {e}")
        return None

# Enhanced generation function with pipeline integration
@handle_error_with_recovery
def generate_video_enhanced(model_type: str, prompt: str, image: Optional[Any] = None,
                          end_image: Optional[Any] = None, resolution: str = "720p", steps: int = 50, 
                          guidance_scale: float = 7.5, strength: float = 0.8, seed: int = -1, 
                          fps: int = 24, duration: int = 4, lora_config: Optional[Dict[str, float]] = None,
                          progress_callback: Optional[Callable] = None, download_timeout: int = 300) -> Dict[str, Any]:
    """
    Enhanced video generation with full pipeline validation and error handling
    Supports both start and end images for I2V and TI2V modes with smart downloading
    """
    try:
        # Log image information for debugging
        if image is not None:
            logger.info(f"Enhanced generation with start image: {type(image)}, size: {getattr(image, 'size', 'unknown')}")
        if end_image is not None:
            logger.info(f"Enhanced generation with end image: {type(end_image)}, size: {getattr(end_image, 'size', 'unknown')}")
        
        # Get enhanced pipeline with extended timeout for model downloads
        pipeline = get_enhanced_generation_pipeline(download_timeout=download_timeout)
        if pipeline is None:
            logger.warning("Enhanced pipeline not available, falling back to legacy generation")
            # Fallback to legacy generation with image support
            return generate_video_legacy(model_type, prompt, image, end_image, resolution, steps, 
                                        guidance_scale, strength, seed, fps, duration, lora_config, progress_callback)
        
        # Create generation request with enhanced image support
        from generation_orchestrator import GenerationRequest
        request = GenerationRequest(
            model_type=model_type,
            prompt=prompt,
            image=image,
            end_image=end_image,
            resolution=resolution,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
            fps=fps,
            duration=duration,
            lora_config=lora_config or {}
        )
        
        # Validate image compatibility with model type
        validation_result = _validate_images_for_model_type(model_type, image, end_image)
        if not validation_result["valid"]:
            logger.warning(f"Image validation warning: {validation_result['message']}")
            # Continue with generation but log the warning
        
        # Add progress callback if provided
        if progress_callback:
            def enhanced_progress_callback(stage, progress):
                try:
                    progress_callback(int(progress), 100)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
            
            pipeline.add_progress_callback(enhanced_progress_callback)
        
        # Execute generation asynchronously with timeout handling
        import asyncio
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create a task with timeout
            task = loop.create_task(pipeline.generate_video(request))
            # Wait for completion with extended timeout for larger files
            result = asyncio.run_coroutine_threadsafe(task, loop).result(timeout=download_timeout + 600)  # Extra 10 minutes for generation
        except RuntimeError:
            # No event loop running, create one with timeout
            result = asyncio.wait_for(pipeline.generate_video(request), timeout=download_timeout + 600)
            result = asyncio.run(result)
        except asyncio.TimeoutError:
            logger.error(f"Generation timed out after {download_timeout + 600} seconds")
            return {
                "success": False,
                "error": f"Generation timed out after {download_timeout + 600} seconds",
                "recovery_suggestions": [
                    "Try reducing the number of steps or duration",
                    "Check system resources and free up memory",
                    "Increase timeout for larger models"
                ]
            }
        
        if result.success:
            logger.info(f"Enhanced generation completed successfully: {result.output_path}")
            return {
                "success": True,
                "output_path": result.output_path,
                "generation_time": result.generation_time,
                "retry_count": result.retry_count,
                "metadata": {
                    **(result.context.metadata if result.context else {}),
                    "image_used": image is not None,
                    "end_image_used": end_image is not None,
                    "model_type": model_type,
                    "enhanced_pipeline": True
                }
            }
        else:
            error_msg = result.error.message if result.error else "Unknown error"
            logger.error(f"Enhanced generation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "retry_count": result.retry_count,
                "recovery_suggestions": result.error.recovery_suggestions if result.error else []
            }
            
    except Exception as e:
        log_error_with_context(e, "enhanced_generation", {
            "model_type": model_type, 
            "prompt": prompt[:100], 
            "resolution": resolution,
            "has_image": image is not None,
            "has_end_image": end_image is not None
        })
        # Fallback to legacy generation with full image support
        logger.warning("Enhanced generation failed, falling back to legacy generation")
        return generate_video_legacy(model_type, prompt, image, end_image, resolution, steps, 
                                    guidance_scale, strength, seed, fps, duration, lora_config, progress_callback)

def _validate_images_for_model_type(model_type: str, image: Optional[Any], end_image: Optional[Any]) -> Dict[str, Any]:
    """Validate that images are appropriate for the selected model type"""
    if model_type == "t2v-A14B":
        if image is not None or end_image is not None:
            return {
                "valid": False,
                "message": "T2V model does not use images, but images were provided"
            }
    elif model_type == "i2v-A14B":
        if image is None:
            return {
                "valid": False,
                "message": "I2V model requires a start image, but none was provided"
            }
        if end_image is not None:
            return {
                "valid": True,
                "message": "I2V model typically uses only start image, end image will be ignored"
            }
    elif model_type == "ti2v-5B":
        if image is None:
            return {
                "valid": False,
                "message": "TI2V model requires a start image, but none was provided"
            }
        # TI2V can use both start and end images
    
    return {"valid": True, "message": "Image configuration is valid for model type"}

# Main generation function with enhanced pipeline integration
@handle_error_with_recovery
def generate_video(model_type: str, prompt: str, image: Optional[Any] = None,
                  end_image: Optional[Any] = None, resolution: str = "720p", steps: int = 50, 
                  guidance_scale: float = 7.5, strength: float = 0.8, seed: int = -1, 
                  fps: int = 24, duration: int = 4, lora_config: Optional[Dict[str, float]] = None,
                  progress_callback: Optional[Callable] = None, download_timeout: int = 300) -> Dict[str, Any]:
    """
    Main video generation function with enhanced pipeline integration
    
    This function now uses the enhanced generation pipeline with:
    - Pre-flight validation and checks
    - Automatic retry mechanisms with optimization
    - Generation mode routing (T2V, I2V, TI2V)
    - Comprehensive error handling and recovery
    - Enhanced support for start and end images
    - Smart downloading with configurable timeouts for larger models
    """
    logger.info(f"Starting video generation: {model_type}, resolution: {resolution}, steps: {steps}")
    
    # Log image information for debugging and validation
    image_info = []
    if image is not None:
        image_size = getattr(image, 'size', 'unknown')
        image_mode = getattr(image, 'mode', 'unknown')
        image_info.append(f"start image ({image_size}, {image_mode})")
        logger.info(f"Start image provided: {type(image)}, size: {image_size}, mode: {image_mode}")
    
    if end_image is not None:
        end_image_size = getattr(end_image, 'size', 'unknown')
        end_image_mode = getattr(end_image, 'mode', 'unknown')
        image_info.append(f"end image ({end_image_size}, {end_image_mode})")
        logger.info(f"End image provided: {type(end_image)}, size: {end_image_size}, mode: {end_image_mode}")
    
    if image_info:
        logger.info(f"Generation with images: {', '.join(image_info)}")
    
    try:
        # Validate image-model compatibility before generation
        validation_result = _validate_images_for_model_type(model_type, image, end_image)
        if not validation_result["valid"]:
            logger.error(f"Image validation failed: {validation_result['message']}")
            return {
                "success": False,
                "error": f"Image validation failed: {validation_result['message']}",
                "recovery_suggestions": [
                    "Check that the correct model type is selected for your images",
                    "T2V models don't use images",
                    "I2V models require a start image",
                    "TI2V models require a start image and optionally an end image"
                ]
            }
        
        # First try enhanced pipeline with extended timeout support
        result = generate_video_enhanced(
            model_type=model_type,
            prompt=prompt,
            image=image,
            end_image=end_image,
            resolution=resolution,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
            fps=fps,
            duration=duration,
            lora_config=lora_config,
            progress_callback=progress_callback,
            download_timeout=download_timeout
        )
        
        # If enhanced pipeline succeeded, return result
        if result.get("success", False):
            logger.info(f"Enhanced pipeline generation completed: {result.get('output_path')}")
            # Add image information to result metadata
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["images_used"] = image_info
            return result
        
        # If enhanced pipeline failed but provided recovery suggestions, log them
        if result.get("recovery_suggestions"):
            logger.info(f"Enhanced pipeline provided recovery suggestions: {result['recovery_suggestions']}")
        
        # Fall back to legacy generation if enhanced pipeline failed
        logger.warning("Enhanced pipeline failed, falling back to legacy generation")
        legacy_result = generate_video_legacy(
            model_type=model_type,
            prompt=prompt,
            image=image,
            end_image=end_image,
            resolution=resolution,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
            fps=fps,
            duration=duration,
            lora_config=lora_config,
            progress_callback=progress_callback
        )
        
        # Add image information to legacy result metadata
        if legacy_result.get("success") and "metadata" not in legacy_result:
            legacy_result["metadata"] = {}
        if legacy_result.get("metadata") is not None:
            legacy_result["metadata"]["images_used"] = image_info
            legacy_result["metadata"]["fallback_used"] = True
        
        return legacy_result
        
    except Exception as e:
        log_error_with_context(e, "video_generation", {
            "model_type": model_type,
            "resolution": resolution,
            "steps": steps,
            "has_image": image is not None,
            "has_end_image": end_image is not None,
            "has_lora": bool(lora_config),
            "download_timeout": download_timeout
        })
        
        # Return error result with enhanced information
        return {
            "success": False,
            "error": f"Video generation failed: {str(e)}",
            "recovery_suggestions": [
                "Check model availability and system resources",
                "Try reducing generation parameters (steps, resolution)",
                "Ensure sufficient VRAM is available",
                "Verify input prompt and images are valid",
                "Increase download timeout for larger models",
                "Check internet connection for model downloads"
            ],
            "metadata": {
                "error_context": {
                    "model_type": model_type,
                    "has_images": image is not None or end_image is not None,
                    "images_used": image_info
                }
            }
        }

# Legacy generation function (renamed from original generate_video)
@handle_error_with_recovery  
def generate_video_legacy(model_type: str, prompt: str, image: Optional[Any] = None,
                         end_image: Optional[Any] = None, resolution: str = "720p", steps: int = 50, 
                         guidance_scale: float = 7.5, strength: float = 0.8, seed: int = -1, 
                         fps: int = 24, duration: int = 4, lora_config: Optional[Dict[str, float]] = None,
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Legacy video generation function (original implementation)
    Used as fallback when enhanced pipeline is not available
    Now supports both start and end images for I2V and TI2V modes
    """
    logger.info(f"Using legacy generation pipeline for {model_type}")
    
    try:
        # Load model using model manager
        model_manager = get_model_manager()
        pipeline, model_info = model_manager.load_model(model_type)
        
        # Move pipeline to GPU if available
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        
        # Set up generation parameters
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)
        
        # Parse resolution
        if resolution == "720p":
            width, height = 1280, 720
        elif resolution == "1080p":
            width, height = 1920, 1080
        elif resolution == "480p":
            width, height = 854, 480
        else:
            width, height = 1280, 720  # Default to 720p
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{model_type}_{timestamp}.mp4"
        
        # Progress tracking
        def internal_progress(step, total_steps):
            if progress_callback:
                progress_callback(step, total_steps)
        
        # Execute generation based on model type
        if model_type in ["t2v-A14B", "text-to-video"]:
            # Text-to-Video generation
            video_frames = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=internal_progress,
                callback_steps=1
            ).frames[0]
            
        elif model_type in ["i2v-A14B", "image-to-video"] and image is not None:
            # Image-to-Video generation
            video_frames = pipeline(
                prompt=prompt if prompt else "",
                image=image,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                callback=internal_progress,
                callback_steps=1
            ).frames[0]
            
        elif model_type in ["ti2v-5B", "text-image-to-video"] and image is not None:
            # Text+Image-to-Video generation
            video_frames = pipeline(
                prompt=prompt,
                image=image,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                callback=internal_progress,
                callback_steps=1
            ).frames[0]
            
        else:
            raise ValueError(f"Unsupported model type or missing required inputs: {model_type}")
        
        # Save video frames to file
        if video_frames:
            # Convert frames to video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for frame in video_frames:
                if isinstance(frame, Image.Image):
                    frame_array = np.array(frame)
                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
            
            out.release()
            
            logger.info(f"Legacy generation completed: {output_path}")
            return {
                "success": True,
                "output_path": str(output_path),
                "generation_time": None,  # Not tracked in legacy mode
                "retry_count": 0
            }
        else:
            raise RuntimeError("No video frames generated")
            
    except Exception as e:
        log_error_with_context(e, "legacy_generation", {
            "model_type": model_type,
            "resolution": resolution,
            "steps": steps
        })
        
        return {
            "success": False,
            "error": f"Legacy generation failed: {str(e)}",
            "recovery_suggestions": [
                "Check model is properly loaded",
                "Verify CUDA availability for GPU acceleration",
                "Try reducing generation parameters",
                "Check available system memory"
            ]
        }

# VRAM Optimization Functions
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
        import time
        
        # Simple timeout tracking for logging purposes
        start_time = time.time()
        
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
            try:
                if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                    model.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.debug(f"xformers not available: {e}")
            
            # Apply gradient checkpointing for training/fine-tuning scenarios
            if optimization_config.get('enable_gradient_checkpointing', False):
                if hasattr(model, 'enable_gradient_checkpointing'):
                    model.enable_gradient_checkpointing()
                    logger.info("Enabled gradient checkpointing")
            
            # Set memory format optimization
            if torch.cuda.is_available() and optimization_config.get('optimize_memory_format', True):
                try:
                    if hasattr(model, 'unet') and model.unet is not None:
                        model.unet = model.unet.to(memory_format=torch.channels_last)
                    logger.info("Applied channels_last memory format optimization")
                except Exception as e:
                    logger.debug(f"Memory format optimization failed: {e}")
            
            return model
            
        except Exception as e:
            log_error_with_context(e, "advanced_optimizations", {"optimization_config": optimization_config})
            logger.error(f"Failed to apply advanced optimizations: {e}")
            return model
    
    def optimize_for_inference(self, model):
        """Optimize model specifically for inference"""
        try:
            # Set model to evaluation mode
            if hasattr(model, 'eval'):
                model.eval()
            
            # Disable gradient computation
            if hasattr(model, 'requires_grad_'):
                model.requires_grad_(False)
            
            # Apply torch.jit.script optimization if possible
            try:
                if hasattr(model, 'unet') and model.unet is not None:
                    # Only script if it's a simple model
                    if hasattr(model.unet, 'forward') and not hasattr(model.unet, 'enable_gradient_checkpointing'):
                        model.unet = torch.jit.script(model.unet)
                        logger.info("Applied TorchScript optimization to UNet")
            except Exception as e:
                logger.debug(f"TorchScript optimization failed: {e}")
            
            # Compile model with torch.compile if available (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile') and hasattr(model, 'unet'):
                    model.unet = torch.compile(model.unet, mode="reduce-overhead")
                    logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.debug(f"torch.compile optimization failed: {e}")
            
            return model
            
        except Exception as e:
            log_error_with_context(e, "inference_optimization", {})
            logger.error(f"Failed to optimize for inference: {e}")
            return model
        
        vae.encode = tiled_encode
        vae.decode = tiled_decode
    
    def _tiled_operation(self, x, operation, tile_size: int):
        """Perform tiled operation on input tensor"""
        if x.shape[-1] <= tile_size and x.shape[-2] <= tile_size:
            # Input is smaller than tile size, process normally
            return operation(x)
        
        # Split input into tiles
        batch_size, channels, height, width = x.shape
        
        # Calculate number of tiles
        tiles_h = (height + tile_size - 1) // tile_size
        tiles_w = (width + tile_size - 1) // tile_size
        
        # Process tiles
        results = []
        for i in range(tiles_h):
            row_results = []
            for j in range(tiles_w):
                # Extract tile
                start_h = i * tile_size
                end_h = min((i + 1) * tile_size, height)
                start_w = j * tile_size
                end_w = min((j + 1) * tile_size, width)
                
                tile = x[:, :, start_h:end_h, start_w:end_w]
                
                # Process tile
                tile_result = operation(tile)
                row_results.append(tile_result)
            
            # Concatenate row results
            row_result = torch.cat(row_results, dim=-1)
            results.append(row_result)
        
        # Concatenate all results
        return torch.cat(results, dim=-2)
    
    def optimize_model(self, model, quantization_level: str = "bf16", 
                      enable_offload: bool = True, vae_tile_size: int = 256, skip_large_components: bool = False):
        """Apply comprehensive VRAM optimizations to a model"""
        logger.info(f"Optimizing model with: quant={quantization_level}, offload={enable_offload}, tile_size={vae_tile_size}, skip_large={skip_large_components}")
        
        # Apply quantization
        model = self.apply_quantization(model, quantization_level, skip_large_components=skip_large_components)
        
        # Enable CPU offload if requested
        if enable_offload:
            model = self.enable_cpu_offload(model, enable_sequential=True)
        
        # Enable VAE tiling
        model = self.enable_vae_tiling(model, vae_tile_size)
        
        # Enable memory efficient attention if available
        try:
            if hasattr(model, 'unet') and model.unet is not None:
                model.unet.set_attn_processor({})  # Use default efficient attention
            logger.info("Enabled memory efficient attention")
        except Exception as e:
            logger.warning(f"Failed to enable memory efficient attention: {e}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model optimization completed")
        return model
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage statistics"""
        if not torch.cuda.is_available():
            return {"used_mb": 0, "total_mb": 0, "free_mb": 0, "usage_percent": 0}
        
        try:
            # Get VRAM info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            cached_memory = torch.cuda.memory_reserved(0)
            
            total_mb = total_memory / (1024 * 1024)
            allocated_mb = allocated_memory / (1024 * 1024)
            cached_mb = cached_memory / (1024 * 1024)
            free_mb = total_mb - cached_mb
            usage_percent = (cached_mb / total_mb) * 100
            
            return {
                "used_mb": allocated_mb,
                "cached_mb": cached_mb,
                "total_mb": total_mb,
                "free_mb": free_mb,
                "usage_percent": usage_percent
            }
            
        except Exception as e:
            logger.error(f"Failed to get VRAM usage: {e}")
            return {"used_mb": 0, "total_mb": 0, "free_mb": 0, "usage_percent": 0}
    
    def check_vram_availability(self, required_mb: float) -> bool:
        """Check if enough VRAM is available for an operation"""
        vram_info = self.get_vram_usage()
        available_mb = vram_info["free_mb"]
        
        return available_mb >= required_mb
    
    def estimate_model_vram_usage(self, model_type: str, quantization_level: str = "bf16") -> float:
        """Estimate VRAM usage for a model type and quantization level"""
        # Base estimates in MB for different model types (bf16)
        base_estimates = {
            "t2v-A14B": 8000,   # ~8GB for T2V model
            "i2v-A14B": 8500,   # ~8.5GB for I2V model  
            "ti2v-5B": 6000,    # ~6GB for TI2V model (smaller)
        }
        
        base_usage = base_estimates.get(model_type, 8000)  # Default to 8GB
        
        # Adjust for quantization
        if quantization_level == "fp16":
            multiplier = 1.0  # Same as bf16
        elif quantization_level == "bf16":
            multiplier = 1.0  # Base estimate
        elif quantization_level == "int8":
            multiplier = 0.6  # ~40% reduction
        else:
            multiplier = 1.0
        
        return base_usage * multiplier


# Global optimizer instance
_vram_optimizer = None

def get_vram_optimizer() -> VRAMOptimizer:
    """Get the global VRAM optimizer instance"""
    global _vram_optimizer
    if _vram_optimizer is None:
        manager = get_model_manager()
        _vram_optimizer = VRAMOptimizer(manager.config)
    return _vram_optimizer


# Convenience functions for VRAM optimization
def optimize_model(model, quantization_level: str = "bf16", 
                  enable_offload: bool = True, vae_tile_size: int = 256, skip_large_components: bool = False):
    """Apply VRAM optimizations to a model"""
    optimizer = get_vram_optimizer()
    return optimizer.optimize_model(model, quantization_level, enable_offload, vae_tile_size, skip_large_components)

def get_vram_usage() -> Dict[str, float]:
    """Get current VRAM usage statistics"""
    optimizer = get_vram_optimizer()
    return optimizer.get_vram_usage()

def check_vram_availability(required_mb: float) -> bool:
    """Check if enough VRAM is available for an operation"""
    optimizer = get_vram_optimizer()
    return optimizer.check_vram_availability(required_mb)


# Prompt Enhancement System
class PromptEnhancer:
    """Handles prompt enhancement, validation, and VACE aesthetic detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enhancement_config = config.get("prompt_enhancement", {})
        self.max_prompt_length = self.enhancement_config.get("max_prompt_length", 500)
        self.min_prompt_length = self.enhancement_config.get("min_prompt_length", 3)
        
        # Quality enhancement keywords
        self.quality_keywords = [
            "high quality", "detailed", "sharp focus", "professional",
            "cinematic lighting", "vibrant colors", "masterpiece",
            "ultra detailed", "8k resolution", "photorealistic"
        ]
        
        # VACE aesthetic keywords
        self.vace_keywords = [
            "vace", "aesthetic", "experimental", "artistic", "avant-garde",
            "abstract", "surreal", "dreamlike", "ethereal", "atmospheric",
            "moody", "stylized", "creative", "unique", "innovative"
        ]
        
        # Cinematic enhancement keywords
        self.cinematic_keywords = [
            "cinematic", "film grain", "depth of field", "bokeh",
            "dramatic lighting", "golden hour", "volumetric lighting",
            "lens flare", "wide angle", "close-up", "establishing shot",
            "color grading", "film noir", "epic", "sweeping camera movement"
        ]
        
        # Style detection patterns
        self.style_patterns = {
            "cinematic": ["cinematic", "film", "movie", "camera", "shot", "scene"],
            "artistic": ["art", "painting", "drawing", "sketch", "illustration"],
            "photographic": ["photo", "photograph", "camera", "lens", "exposure"],
            "fantasy": ["fantasy", "magical", "mystical", "enchanted", "fairy"],
            "sci-fi": ["futuristic", "sci-fi", "cyberpunk", "robot", "space", "alien"],
            "nature": ["landscape", "forest", "mountain", "ocean", "sunset", "sunrise"]
        }
        
        # Invalid characters for prompt validation
        self.invalid_chars = set(['<', '>', '{', '}', '[', ']', '|', '\\', '^', '~'])
    
    def enhance_prompt(self, prompt: str, apply_vace: bool = None, apply_cinematic: bool = None) -> str:
        """Enhance a prompt with quality keywords and style improvements"""
        if not prompt or not prompt.strip():
            return prompt
        
        enhanced_prompt = prompt.strip()
        
        # Auto-detect VACE if not specified
        if apply_vace is None:
            apply_vace = self.detect_vace_aesthetics(prompt)
        
        # Auto-detect cinematic style if not specified
        if apply_cinematic is None:
            apply_cinematic = self._detect_cinematic_style(prompt)
        
        # Add quality keywords if not already present
        quality_additions = []
        prompt_lower = enhanced_prompt.lower()
        
        for keyword in self.quality_keywords[:3]:  # Add top 3 quality keywords
            if keyword.lower() not in prompt_lower:
                quality_additions.append(keyword)
        
        if quality_additions:
            enhanced_prompt += ", " + ", ".join(quality_additions)
        
        # Add VACE enhancements if detected or requested
        if apply_vace:
            vace_additions = []
            for keyword in self.vace_keywords[:2]:  # Add top 2 VACE keywords
                if keyword.lower() not in prompt_lower:
                    vace_additions.append(keyword)
            
            if vace_additions:
                enhanced_prompt += ", " + ", ".join(vace_additions)
        
        # Add cinematic enhancements if detected or requested
        if apply_cinematic:
            cinematic_additions = []
            for keyword in self.cinematic_keywords[:2]:  # Add top 2 cinematic keywords
                if keyword.lower() not in prompt_lower:
                    cinematic_additions.append(keyword)
            
            if cinematic_additions:
                enhanced_prompt += ", " + ", ".join(cinematic_additions)
        
        # Ensure we don't exceed max length
        if len(enhanced_prompt) > self.max_prompt_length:
            # Truncate while preserving the original prompt
            original_length = len(prompt)
            if original_length < self.max_prompt_length:
                # Keep original + as much enhancement as possible
                available_space = self.max_prompt_length - original_length - 2  # -2 for ", "
                if available_space > 0:
                    enhancement_part = enhanced_prompt[original_length + 2:]
                    truncated_enhancement = enhancement_part[:available_space]
                    # Find last complete keyword
                    last_comma = truncated_enhancement.rfind(', ')
                    if last_comma > 0:
                        truncated_enhancement = truncated_enhancement[:last_comma]
                    enhanced_prompt = prompt + ", " + truncated_enhancement
                else:
                    enhanced_prompt = prompt[:self.max_prompt_length]
            else:
                enhanced_prompt = enhanced_prompt[:self.max_prompt_length]
        
        return enhanced_prompt
    
    def detect_vace_aesthetics(self, prompt: str) -> bool:
        """Detect if a prompt contains VACE aesthetic keywords or concepts"""
        if not prompt:
            return False
        
        prompt_lower = prompt.lower()
        
        # Check for explicit VACE keywords
        for keyword in self.vace_keywords:
            if keyword in prompt_lower:
                return True
        
        # Check for aesthetic-related terms
        aesthetic_terms = [
            "aesthetic", "aesthetics", "artistic", "experimental", "avant-garde",
            "abstract", "surreal", "dreamlike", "ethereal", "atmospheric",
            "moody", "stylized", "creative composition", "unique style",
            "innovative", "conceptual", "expressive", "evocative"
        ]
        
        for term in aesthetic_terms:
            if term in prompt_lower:
                return True
        
        return False
    
    def _detect_cinematic_style(self, prompt: str) -> bool:
        """Detect if a prompt suggests cinematic style"""
        if not prompt:
            return False
        
        prompt_lower = prompt.lower()
        
        cinematic_indicators = [
            "cinematic", "film", "movie", "camera", "shot", "scene",
            "dramatic", "epic", "sweeping", "lens", "lighting",
            "depth of field", "bokeh", "film grain", "color grading"
        ]
        
        for indicator in cinematic_indicators:
            if indicator in prompt_lower:
                return True
        
        return False
    
    def detect_style(self, prompt: str) -> str:
        """Detect the primary style of a prompt"""
        if not prompt:
            return "general"
        
        prompt_lower = prompt.lower()
        style_scores = {}
        
        # Score each style based on keyword matches
        for style, keywords in self.style_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1
            style_scores[style] = score
        
        # Return the style with the highest score, or "general" if no clear match
        if style_scores:
            max_score = max(style_scores.values())
            if max_score > 0:
                return max(style_scores, key=style_scores.get)
        
        return "general"
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Validate a prompt for length and content requirements"""
        if not prompt:
            return False, "Prompt cannot be empty"
        
        prompt = prompt.strip()
        
        # Check minimum length
        if len(prompt) < self.min_prompt_length:
            return False, f"Prompt must be at least {self.min_prompt_length} characters long"
        
        # Check maximum length
        if len(prompt) > self.max_prompt_length:
            return False, f"Prompt must be no more than {self.max_prompt_length} characters long"
        
        # Check for invalid characters
        invalid_found = []
        for char in prompt:
            if char in self.invalid_chars:
                invalid_found.append(char)
        
        if invalid_found:
            unique_invalid = list(set(invalid_found))
            return False, f"Prompt contains invalid characters: {', '.join(unique_invalid)}"
        
        # Check for potentially problematic content
        problematic_terms = ["nsfw", "explicit", "adult", "inappropriate"]
        prompt_lower = prompt.lower()
        
        for term in problematic_terms:
            if term in prompt_lower:
                return False, f"Prompt contains potentially inappropriate content: '{term}'"
        
        return True, "Prompt is valid"
    
    def get_enhancement_preview(self, prompt: str) -> Dict[str, Any]:
        """Get a preview of how a prompt would be enhanced"""
        original_prompt = prompt
        original_length = len(prompt) if prompt else 0
        
        # Validate original prompt
        is_valid, validation_message = self.validate_prompt(prompt)
        
        # Detect characteristics
        detected_vace = self.detect_vace_aesthetics(prompt)
        detected_style = self.detect_style(prompt)
        detected_cinematic = self._detect_cinematic_style(prompt)
        
        # Generate suggested enhancements
        suggested_enhancements = []
        
        if prompt:
            prompt_lower = prompt.lower()
            
            # Suggest quality keywords
            quality_suggestions = []
            for keyword in self.quality_keywords[:3]:
                if keyword.lower() not in prompt_lower:
                    quality_suggestions.append(keyword)
            
            if quality_suggestions:
                suggested_enhancements.append({
                    "type": "quality",
                    "keywords": quality_suggestions,
                    "description": "Quality improvement keywords"
                })
            
            # Suggest VACE enhancements if detected
            if detected_vace:
                vace_suggestions = []
                for keyword in self.vace_keywords[:2]:
                    if keyword.lower() not in prompt_lower:
                        vace_suggestions.append(keyword)
                
                if vace_suggestions:
                    suggested_enhancements.append({
                        "type": "vace",
                        "keywords": vace_suggestions,
                        "description": "VACE aesthetic enhancements"
                    })
            
            # Suggest cinematic enhancements if detected
            if detected_cinematic:
                cinematic_suggestions = []
                for keyword in self.cinematic_keywords[:2]:
                    if keyword.lower() not in prompt_lower:
                        cinematic_suggestions.append(keyword)
                
                if cinematic_suggestions:
                    suggested_enhancements.append({
                        "type": "cinematic",
                        "keywords": cinematic_suggestions,
                        "description": "Cinematic style enhancements"
                    })
        
        # Estimate final length
        estimated_additions = 0
        for enhancement in suggested_enhancements:
            for keyword in enhancement["keywords"]:
                estimated_additions += len(keyword) + 2  # +2 for ", "
        
        estimated_final_length = original_length + estimated_additions
        
        # Check if enhancement would exceed limit
        would_exceed_limit = estimated_final_length > self.max_prompt_length
        
        return {
            "original_prompt": original_prompt,
            "original_length": original_length,
            "is_valid": is_valid,
            "validation_message": validation_message,
            "detected_vace": detected_vace,
            "detected_style": detected_style,
            "detected_cinematic": detected_cinematic,
            "suggested_enhancements": suggested_enhancements,
            "estimated_final_length": estimated_final_length,
            "would_exceed_limit": would_exceed_limit,
            "max_length": self.max_prompt_length
        }
    
    def clean_prompt(self, prompt: str) -> str:
        """Clean a prompt by removing invalid characters and normalizing whitespace"""
        if not prompt:
            return ""
        
        # Remove invalid characters
        cleaned = ""
        for char in prompt:
            if char not in self.invalid_chars:
                cleaned += char
        
        # Normalize whitespace
        cleaned = " ".join(cleaned.split())
        
        # Ensure length limits
        if len(cleaned) > self.max_prompt_length:
            cleaned = cleaned[:self.max_prompt_length].strip()
        
        return cleaned


# Global prompt enhancer instance
_prompt_enhancer = None

def get_prompt_enhancer() -> PromptEnhancer:
    """Get the global prompt enhancer instance"""
    global _prompt_enhancer
    if _prompt_enhancer is None:
        manager = get_model_manager()
        _prompt_enhancer = PromptEnhancer(manager.config)
    return _prompt_enhancer


# Convenience functions for prompt enhancement
def enhance_prompt(prompt: str, apply_vace: bool = None, apply_cinematic: bool = None) -> str:
    """Enhance a prompt with quality keywords and style improvements"""
    enhancer = get_prompt_enhancer()
    return enhancer.enhance_prompt(prompt, apply_vace, apply_cinematic)

def detect_vace_aesthetics(prompt: str) -> bool:
    """Detect if a prompt contains VACE aesthetic keywords or concepts"""
    enhancer = get_prompt_enhancer()
    return enhancer.detect_vace_aesthetics(prompt)

def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """Validate a prompt for length and content requirements"""
    enhancer = get_prompt_enhancer()
    return enhancer.validate_prompt(prompt)

def get_enhancement_preview(prompt: str) -> Dict[str, Any]:
    """Get a preview of how a prompt would be enhanced"""
    enhancer = get_prompt_enhancer()
    return enhancer.get_enhancement_preview(prompt)

def clean_prompt(prompt: str) -> str:
    """Clean a prompt by removing invalid characters and normalizing whitespace"""
    enhancer = get_prompt_enhancer()
    return enhancer.clean_prompt(prompt)

def check_vram_availability(required_mb: float) -> bool:
    """Check if enough VRAM is available for an operation"""
    optimizer = get_vram_optimizer()
    return optimizer.check_vram_availability(required_mb)


# Prompt Enhancement System
class PromptEnhancer:
    """Handles prompt enhancement for better video generation quality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enhancement_config = config.get("prompt_enhancement", {})
        self.max_prompt_length = self.enhancement_config.get("max_prompt_length", 500)
        
        # Quality keywords for basic enhancement
        self.quality_keywords = [
            "high quality", "detailed", "sharp focus", "professional",
            "cinematic lighting", "vibrant colors", "ultra detailed",
            "masterpiece", "best quality", "8k resolution"
        ]
        
        # VACE aesthetic keywords for detection
        self.vace_keywords = [
            "vace", "aesthetic", "artistic", "stylized", "creative",
            "experimental", "avant-garde", "abstract", "surreal",
            "dreamlike", "ethereal", "atmospheric", "moody"
        ]
        
        # Cinematic enhancement keywords
        self.cinematic_keywords = [
            "cinematic", "film grain", "depth of field", "bokeh",
            "dramatic lighting", "golden hour", "volumetric lighting",
            "lens flare", "motion blur", "camera movement", "tracking shot",
            "establishing shot", "close-up", "wide angle", "telephoto"
        ]
        
        # Style-specific enhancements
        self.style_enhancements = {
            "portrait": ["portrait photography", "shallow depth of field", "soft lighting"],
            "landscape": ["landscape photography", "wide angle", "golden hour", "dramatic sky"],
            "action": ["dynamic motion", "motion blur", "high energy", "fast paced"],
            "nature": ["natural lighting", "organic", "environmental", "wildlife"],
            "urban": ["urban environment", "city lights", "architectural", "street photography"],
            "fantasy": ["magical", "otherworldly", "fantastical", "mystical"],
            "sci-fi": ["futuristic", "technological", "cyberpunk", "neon lights"]
        }
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Validate prompt length and content"""
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty"
        
        prompt = prompt.strip()
        
        # Check length
        if len(prompt) > self.max_prompt_length:
            return False, f"Prompt exceeds maximum length of {self.max_prompt_length} characters"
        
        # Check for minimum length
        if len(prompt) < 3:
            return False, "Prompt must be at least 3 characters long"
        
        # Check for invalid characters (basic validation)
        invalid_chars = ['<', '>', '{', '}', '[', ']']
        for char in invalid_chars:
            if char in prompt:
                return False, f"Prompt contains invalid character: {char}"
        
        return True, "Prompt is valid"
    
    def manage_prompt_length(self, prompt: str) -> str:
        """Manage prompt length by truncating if necessary"""
        if len(prompt) <= self.max_prompt_length:
            return prompt
        
        # Truncate at word boundary near the limit
        truncated = prompt[:self.max_prompt_length]
        last_space = truncated.rfind(' ')
        
        if last_space > self.max_prompt_length * 0.8:  # If we can find a space reasonably close
            truncated = truncated[:last_space]
        
        logger.warning(f"Prompt truncated from {len(prompt)} to {len(truncated)} characters")
        return truncated
    
    def detect_vace_aesthetics(self, prompt: str) -> bool:
        """Detect if prompt contains VACE aesthetic keywords"""
        prompt_lower = prompt.lower()
        
        # Check for direct VACE keywords
        for keyword in self.vace_keywords:
            if keyword in prompt_lower:
                logger.info(f"Detected VACE keyword: {keyword}")
                return True
        
        # Check for aesthetic-related phrases
        aesthetic_phrases = [
            "artistic style", "creative vision", "experimental art",
            "visual aesthetics", "artistic expression", "stylized rendering"
        ]
        
        for phrase in aesthetic_phrases:
            if phrase in prompt_lower:
                logger.info(f"Detected VACE aesthetic phrase: {phrase}")
                return True
        
        return False
    
    def detect_style_category(self, prompt: str) -> Optional[str]:
        """Detect the style category of the prompt for targeted enhancement"""
        prompt_lower = prompt.lower()
        
        # Style detection keywords
        style_indicators = {
            "portrait": ["person", "face", "character", "portrait", "headshot", "close-up"],
            "landscape": ["landscape", "mountain", "forest", "ocean", "sky", "horizon", "nature scene"],
            "action": ["running", "jumping", "fighting", "racing", "sports", "movement", "dynamic"],
            "nature": ["animal", "wildlife", "plant", "tree", "flower", "natural", "organic"],
            "urban": ["city", "building", "street", "urban", "architecture", "downtown"],
            "fantasy": ["dragon", "magic", "wizard", "fantasy", "mythical", "enchanted"],
            "sci-fi": ["robot", "spaceship", "futuristic", "cyberpunk", "alien", "technology"]
        }
        
        for style, keywords in style_indicators.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    return style
        
        return None
    
    def enhance_basic_quality(self, prompt: str) -> str:
        """Add basic quality enhancement keywords to the prompt"""
        # Select quality keywords that aren't already in the prompt
        prompt_lower = prompt.lower()
        selected_keywords = []
        
        for keyword in self.quality_keywords:
            if keyword not in prompt_lower and len(selected_keywords) < 3:
                selected_keywords.append(keyword)
        
        if selected_keywords:
            enhanced_prompt = f"{prompt}, {', '.join(selected_keywords)}"
            logger.info(f"Added basic quality keywords: {selected_keywords}")
            return enhanced_prompt
        
        return prompt
    
    def enhance_vace_aesthetics(self, prompt: str) -> str:
        """Enhance prompt with VACE aesthetic improvements"""
        vace_enhancements = [
            "VACE experimental cocktail aesthetics",
            "artistic cinematography",
            "creative visual composition",
            "aesthetic enhancement",
            "stylized rendering"
        ]
        
        # Select enhancements not already in prompt
        prompt_lower = prompt.lower()
        selected_enhancements = []
        
        for enhancement in vace_enhancements:
            if enhancement.lower() not in prompt_lower and len(selected_enhancements) < 2:
                selected_enhancements.append(enhancement)
        
        if selected_enhancements:
            enhanced_prompt = f"{prompt}, {', '.join(selected_enhancements)}"
            logger.info(f"Added VACE aesthetic enhancements: {selected_enhancements}")
            return enhanced_prompt
        
        return prompt
    
    def enhance_cinematic_style(self, prompt: str) -> str:
        """Add cinematic style improvements to the prompt"""
        # Select cinematic keywords that aren't already in the prompt
        prompt_lower = prompt.lower()
        selected_keywords = []
        
        for keyword in self.cinematic_keywords:
            if keyword not in prompt_lower and len(selected_keywords) < 3:
                selected_keywords.append(keyword)
        
        if selected_keywords:
            enhanced_prompt = f"{prompt}, {', '.join(selected_keywords)}"
            logger.info(f"Added cinematic enhancements: {selected_keywords}")
            return enhanced_prompt
        
        return prompt
    
    def enhance_by_style(self, prompt: str, style: str) -> str:
        """Enhance prompt based on detected style category"""
        if style not in self.style_enhancements:
            return prompt
        
        style_keywords = self.style_enhancements[style]
        prompt_lower = prompt.lower()
        selected_keywords = []
        
        for keyword in style_keywords:
            if keyword not in prompt_lower and len(selected_keywords) < 2:
                selected_keywords.append(keyword)
        
        if selected_keywords:
            enhanced_prompt = f"{prompt}, {', '.join(selected_keywords)}"
            logger.info(f"Added {style} style enhancements: {selected_keywords}")
            return enhanced_prompt
        
        return prompt
    
    def enhance_prompt(self, prompt: str, enable_vace: bool = True, 
                      enable_cinematic: bool = True, enable_style: bool = True) -> str:
        """
        Comprehensive prompt enhancement
        
        Args:
            prompt: Original text prompt
            enable_vace: Whether to apply VACE aesthetic enhancements
            enable_cinematic: Whether to apply cinematic enhancements
            enable_style: Whether to apply style-specific enhancements
            
        Returns:
            Enhanced prompt string
        """
        # Validate input prompt
        is_valid, error_msg = self.validate_prompt(prompt)
        if not is_valid:
            logger.error(f"Prompt validation failed: {error_msg}")
            raise ValueError(error_msg)
        
        # Start with the original prompt
        enhanced_prompt = prompt.strip()
        
        # Apply basic quality enhancement
        enhanced_prompt = self.enhance_basic_quality(enhanced_prompt)
        
        # Apply VACE aesthetic enhancement if detected or enabled
        if enable_vace and self.detect_vace_aesthetics(enhanced_prompt):
            enhanced_prompt = self.enhance_vace_aesthetics(enhanced_prompt)
        
        # Apply cinematic enhancements
        if enable_cinematic:
            enhanced_prompt = self.enhance_cinematic_style(enhanced_prompt)
        
        # Apply style-specific enhancements
        if enable_style:
            detected_style = self.detect_style_category(enhanced_prompt)
            if detected_style:
                enhanced_prompt = self.enhance_by_style(enhanced_prompt, detected_style)
                logger.info(f"Applied {detected_style} style enhancements")
        
        # Manage final prompt length
        enhanced_prompt = self.manage_prompt_length(enhanced_prompt)
        
        logger.info(f"Prompt enhancement completed. Original: {len(prompt)} chars, Enhanced: {len(enhanced_prompt)} chars")
        return enhanced_prompt
    
    def get_enhancement_preview(self, prompt: str) -> Dict[str, Any]:
        """Get a preview of what enhancements would be applied without actually applying them"""
        preview = {
            "original_prompt": prompt,
            "original_length": len(prompt),
            "is_valid": False,
            "validation_error": None,
            "detected_vace": False,
            "detected_style": None,
            "suggested_enhancements": [],
            "estimated_final_length": len(prompt)
        }
        
        # Validate prompt
        is_valid, error_msg = self.validate_prompt(prompt)
        preview["is_valid"] = is_valid
        if not is_valid:
            preview["validation_error"] = error_msg
            return preview
        
        # Detect VACE aesthetics
        preview["detected_vace"] = self.detect_vace_aesthetics(prompt)
        
        # Detect style category
        preview["detected_style"] = self.detect_style_category(prompt)
        
        # Simulate enhancements to estimate final length
        temp_prompt = prompt
        enhancements = []
        
        # Basic quality enhancement
        enhanced = self.enhance_basic_quality(temp_prompt)
        if enhanced != temp_prompt:
            enhancements.append("Basic quality keywords")
            temp_prompt = enhanced
        
        # VACE enhancement
        if preview["detected_vace"]:
            enhanced = self.enhance_vace_aesthetics(temp_prompt)
            if enhanced != temp_prompt:
                enhancements.append("VACE aesthetic enhancements")
                temp_prompt = enhanced
        
        # Cinematic enhancement
        enhanced = self.enhance_cinematic_style(temp_prompt)
        if enhanced != temp_prompt:
            enhancements.append("Cinematic style improvements")
            temp_prompt = enhanced
        
        # Style-specific enhancement
        if preview["detected_style"]:
            enhanced = self.enhance_by_style(temp_prompt, preview["detected_style"])
            if enhanced != temp_prompt:
                enhancements.append(f"{preview['detected_style'].title()} style enhancements")
                temp_prompt = enhanced
        
        preview["suggested_enhancements"] = enhancements
        preview["estimated_final_length"] = len(temp_prompt)
        
        return preview


# Global prompt enhancer instance
_prompt_enhancer = None

def get_prompt_enhancer() -> PromptEnhancer:
    """Get the global prompt enhancer instance"""
    global _prompt_enhancer
    if _prompt_enhancer is None:
        manager = get_model_manager()
        _prompt_enhancer = PromptEnhancer(manager.config)
    return _prompt_enhancer


# Convenience functions for prompt enhancement
def enhance_prompt(prompt: str, enable_vace: bool = True, 
                  enable_cinematic: bool = True, enable_style: bool = True) -> str:
    """Enhance a text prompt for better video generation quality"""
    enhancer = get_prompt_enhancer()
    return enhancer.enhance_prompt(prompt, enable_vace, enable_cinematic, enable_style)

def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """Validate a text prompt"""
    enhancer = get_prompt_enhancer()
    return enhancer.validate_prompt(prompt)

def get_enhancement_preview(prompt: str) -> Dict[str, Any]:
    """Get a preview of prompt enhancements without applying them"""
    enhancer = get_prompt_enhancer()
    return enhancer.get_enhancement_preview(prompt)

def detect_vace_aesthetics(prompt: str) -> bool:
    """Detect if prompt contains VACE aesthetic keywords"""
    enhancer = get_prompt_enhancer()
    return enhancer.detect_vace_aesthetics(prompt)


# Video Generation Engine
class VideoGenerationEngine:
    """Core video generation engine for T2V, I2V, and TI2V modes"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.model_manager = get_model_manager()
        self.vram_optimizer = get_vram_optimizer()
        self.loaded_pipelines: Dict[str, Any] = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                "generation": {
                    "default_steps": 50,
                    "default_guidance_scale": 7.5,
                    "max_resolution": "1920x1080",
                    "supported_resolutions": ["854x480", "480x854", "1280x720", "1280x704", "1920x1080"]
                },
                "optimization": {
                    "default_quantization": "bf16",
                    "enable_offload": True,
                    "vae_tile_size": 256
                }
            }
    
    def _get_pipeline(self, model_type: str, **optimization_kwargs) -> Any:
        """Get or load a pipeline for the specified model type"""
        if model_type in self.loaded_pipelines:
            return self.loaded_pipelines[model_type]
        
        # Load the model
        pipeline, model_info = self.model_manager.load_model(model_type)
        
        # Apply optimizations
        quantization = optimization_kwargs.get('quantization_level', 
                                              self.config['optimization']['default_quantization'])
        enable_offload = optimization_kwargs.get('enable_offload', 
                                                self.config['optimization']['enable_offload'])
        vae_tile_size = optimization_kwargs.get('vae_tile_size', 
                                               self.config['optimization']['vae_tile_size'])
        
        # Check if we should skip large components (for WAN models that might hang)
        # Default to True for WAN models to prevent hanging during quantization
        is_wan_model = 'wan' in model_type.lower() or 't2v' in model_type.lower()
        default_skip_large = is_wan_model  # Skip large components by default for WAN models
        skip_large = optimization_kwargs.get('skip_large_components', default_skip_large)
        
        optimized_pipeline = self.vram_optimizer.optimize_model(
            pipeline, quantization, enable_offload, vae_tile_size, skip_large
        )
        
        # Move to GPU if available and not using CPU offloading
        if torch.cuda.is_available() and not enable_offload:
            optimized_pipeline = optimized_pipeline.to("cuda")
        elif enable_offload:
            logger.info("Skipping GPU move due to CPU offloading being enabled")
        
        self.loaded_pipelines[model_type] = optimized_pipeline
        return optimized_pipeline
    
    def generate_t2v(self, prompt: str, resolution: str = "1280x720", 
                     num_inference_steps: int = 50, guidance_scale: float = 7.5,
                     **kwargs) -> Dict[str, Any]:
        """
        Generate video from text prompt (Text-to-Video)
        
        Args:
            prompt: Text description for video generation
            resolution: Output resolution (e.g., "1280x720")
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated video frames and metadata
        """
        logger.info(f"Starting T2V generation: '{prompt[:50]}...' at {resolution}")
        
        try:
            # Get the T2V pipeline
            pipeline = self._get_pipeline("t2v-A14B", **kwargs)
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Set generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": kwargs.get("num_frames", 16),  # Default 16 frames
                # Filter out optimization and unsupported parameters
                **{k: v for k, v in kwargs.items() if k not in [
                    'quantization_level', 'enable_offload', 'vae_tile_size', 'skip_large_components',
                    'duration', 'fps'  # These are handled separately or not supported by WanPipeline
                ]}
            }
            
            # Generate video
            logger.info(f"Generating T2V with {num_inference_steps} steps...")
            result = pipeline(**generation_kwargs)
            
            # Extract frames
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # Get first (and typically only) video
            elif hasattr(result, 'videos'):
                frames = result.videos[0]
            else:
                # Fallback - assume result is the frames directly
                frames = result
            
            logger.info(f"T2V generation completed: {len(frames)} frames at {width}x{height}")
            
            return {
                "frames": frames,
                "metadata": {
                    "model_type": "t2v-A14B",
                    "prompt": prompt,
                    "resolution": resolution,
                    "width": width,
                    "height": height,
                    "num_frames": len(frames),
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"T2V generation failed: {e}")
            raise
    
    def generate_i2v(self, prompt: str, image: Image.Image, resolution: str = "1280x720",
                     num_inference_steps: int = 50, guidance_scale: float = 7.5,
                     **kwargs) -> Dict[str, Any]:
        """
        Generate video from image and text prompt (Image-to-Video)
        
        Args:
            prompt: Text description for video generation
            image: Input PIL Image to animate
            resolution: Output resolution (e.g., "1280x720")
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated video frames and metadata
        """
        logger.info(f"Starting I2V generation: '{prompt[:50]}...' with image at {resolution}")
        
        try:
            # Get the I2V pipeline
            pipeline = self._get_pipeline("i2v-A14B", **kwargs)
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Resize input image to match target resolution
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Set generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "image": resized_image,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": kwargs.get("num_frames", 16),  # Default 16 frames
                **{k: v for k, v in kwargs.items() if k not in ['quantization_level', 'enable_offload', 'vae_tile_size']}
            }
            
            # Generate video
            logger.info(f"Generating I2V with {num_inference_steps} steps...")
            result = pipeline(**generation_kwargs)
            
            # Extract frames
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # Get first (and typically only) video
            elif hasattr(result, 'videos'):
                frames = result.videos[0]
            else:
                # Fallback - assume result is the frames directly
                frames = result
            
            logger.info(f"I2V generation completed: {len(frames)} frames at {width}x{height}")
            
            return {
                "frames": frames,
                "metadata": {
                    "model_type": "i2v-A14B",
                    "prompt": prompt,
                    "resolution": resolution,
                    "width": width,
                    "height": height,
                    "num_frames": len(frames),
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "input_image_size": f"{image.width}x{image.height}",
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"I2V generation failed: {e}")
            raise
    
    def generate_ti2v(self, prompt: str, image: Image.Image, resolution: str = "1280x720",
                      num_inference_steps: int = 50, guidance_scale: float = 7.5,
                      **kwargs) -> Dict[str, Any]:
        """
        Generate video from text and image inputs (Text-Image-to-Video hybrid)
        
        Args:
            prompt: Text description for video generation
            image: Input PIL Image as visual reference
            resolution: Output resolution (e.g., "1280x720", "1920x1080")
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated video frames and metadata
        """
        logger.info(f"Starting TI2V generation: '{prompt[:50]}...' with image at {resolution}")
        
        try:
            # Get the TI2V pipeline
            pipeline = self._get_pipeline("ti2v-5B", **kwargs)
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Resize input image to match target resolution
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Set generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "image": resized_image,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": kwargs.get("num_frames", 16),  # Default 16 frames
                **{k: v for k, v in kwargs.items() if k not in ['quantization_level', 'enable_offload', 'vae_tile_size']}
            }
            
            # Generate video
            logger.info(f"Generating TI2V with {num_inference_steps} steps...")
            result = pipeline(**generation_kwargs)
            
            # Extract frames
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # Get first (and typically only) video
            elif hasattr(result, 'videos'):
                frames = result.videos[0]
            else:
                # Fallback - assume result is the frames directly
                frames = result
            
            logger.info(f"TI2V generation completed: {len(frames)} frames at {width}x{height}")
            
            return {
                "frames": frames,
                "metadata": {
                    "model_type": "ti2v-5B",
                    "prompt": prompt,
                    "resolution": resolution,
                    "width": width,
                    "height": height,
                    "num_frames": len(frames),
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "input_image_size": f"{image.width}x{image.height}",
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"TI2V generation failed: {e}")
            raise
    
    @handle_error_with_recovery
    def generate_video(self, model_type: str, prompt: str, image: Optional[Image.Image] = None,
                      end_image: Optional[Image.Image] = None, resolution: str = "1280x720", 
                      num_inference_steps: int = 50, guidance_scale: float = 7.5, **kwargs) -> Dict[str, Any]:
        """
        Universal video generation function that routes to appropriate generation method
        
        Args:
            model_type: Type of generation ("t2v", "i2v", "ti2v")
            prompt: Text description for video generation
            image: Optional input image (required for i2v and ti2v)
            resolution: Output resolution
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated video frames and metadata
        """
        try:
            # Validate inputs
            self._validate_generation_inputs(model_type, prompt, image, resolution, num_inference_steps)
            
            # Check system resources before generation
            self._check_system_resources(resolution)
            
            model_type = model_type.lower()
            
            # Normalize model type (handle variants like t2v-a14b)
            if model_type.startswith("t2v"):
                model_type = "t2v"
            elif model_type.startswith("i2v"):
                model_type = "i2v"
            elif model_type.startswith("ti2v"):
                model_type = "ti2v"
            
            # Log generation start
            logger.info(f"Starting {model_type.upper()} generation: prompt='{prompt[:50]}...', resolution={resolution}, steps={num_inference_steps}")
            
            if model_type == "t2v":
                if image is not None:
                    logger.warning("Image provided for T2V generation, ignoring image input")
                return self.generate_t2v(prompt, resolution, num_inference_steps, guidance_scale, **kwargs)
            
            elif model_type == "i2v":
                if image is None:
                    raise ValueError("Image input is required for I2V generation")
                return self.generate_i2v(prompt, image, resolution, num_inference_steps, guidance_scale, **kwargs)
            
            elif model_type == "ti2v":
                if image is None:
                    raise ValueError("Image input is required for TI2V generation")
                return self.generate_ti2v(prompt, image, resolution, num_inference_steps, guidance_scale, **kwargs)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}. Supported types: t2v, i2v, ti2v")
                
        except torch.cuda.OutOfMemoryError as e:
            log_error_with_context(e, "video_generation", {
                "model_type": model_type,
                "resolution": resolution,
                "steps": num_inference_steps,
                "vram_info": self.vram_optimizer.get_vram_usage()
            })
            raise
        except Exception as e:
            log_error_with_context(e, "video_generation", {
                "model_type": model_type,
                "prompt_length": len(prompt),
                "resolution": resolution,
                "steps": num_inference_steps,
                "has_image": image is not None
            })
            raise
    
    def _validate_generation_inputs(self, model_type: str, prompt: str, image: Optional[Image.Image], 
                                  resolution: str, num_inference_steps: int):
        """Validate generation inputs"""
        # Validate model type
        valid_types = ["t2v", "i2v", "ti2v", "t2v-a14b", "i2v-a14b", "ti2v-5b"]
        if model_type.lower() not in valid_types:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if len(prompt) > 500:
            raise ValueError(f"Prompt too long: {len(prompt)} characters (max 500)")
        
        # Validate resolution
        valid_resolutions = ["854x480", "480x854", "1280x720", "1280x704", "1920x1080"]
        if resolution not in valid_resolutions:
            raise ValueError(f"Invalid resolution: {resolution}. Valid options: {valid_resolutions}")
        
        # Validate steps
        if not (10 <= num_inference_steps <= 100):
            raise ValueError(f"Invalid number of steps: {num_inference_steps} (must be 10-100)")
        
        # Validate image for I2V/TI2V
        if model_type.lower() in ["i2v", "ti2v", "i2v-a14b", "ti2v-5b"]:
            if image is None:
                raise ValueError(f"Image input is required for {model_type.upper()} generation")
            
            # Validate image format and size
            if not isinstance(image, Image.Image):
                raise ValueError("Image must be a PIL Image object")
            
            # Check image size (not too large to avoid memory issues)
            max_pixels = 2048 * 2048
            if image.width * image.height > max_pixels:
                raise ValueError(f"Image too large: {image.width}x{image.height} (max {max_pixels} pixels)")
    
    def _check_system_resources(self, resolution: str):
        """Check system resources before generation"""
        # Check VRAM
        vram_info = self.vram_optimizer.get_vram_usage()
        
        # Estimate VRAM requirements based on resolution
        vram_requirements = {
            "854x480": 6000,    # 6GB (landscape 480p)
            "480x854": 6000,    # 6GB (portrait 480p)
            "1280x720": 8000,   # 8GB
            "1280x704": 8000,   # 8GB
            "1920x1080": 12000  # 12GB
        }
        
        required_vram = vram_requirements.get(resolution, 8000)
        if vram_info["free_mb"] < required_vram:
            raise RuntimeError(f"Insufficient VRAM: {vram_info['free_mb']:.0f}MB free, need {required_vram}MB for {resolution}")
        
        # Check system RAM
        ram_info = psutil.virtual_memory()
        if ram_info.percent > 90:
            raise RuntimeError(f"System RAM usage too high: {ram_info.percent:.1f}%")
        
        # Check disk space
        disk_info = psutil.disk_usage('.')
        free_gb = disk_info.free / (1024**3)
        if free_gb < 5:  # Need at least 5GB for output files
            raise RuntimeError(f"Insufficient disk space: {free_gb:.1f}GB free, need at least 5GB")
    
    def unload_pipeline(self, model_type: str):
        """Unload a specific pipeline from memory"""
        if model_type in self.loaded_pipelines:
            del self.loaded_pipelines[model_type]
            
            # Also unload from model manager
            self.model_manager.unload_model(model_type)
            
            logger.info(f"Unloaded pipeline: {model_type}")
    
    def unload_all_pipelines(self):
        """Unload all pipelines from memory"""
        for model_type in list(self.loaded_pipelines.keys()):
            self.unload_pipeline(model_type)
        
        logger.info("Unloaded all pipelines")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of all loaded pipelines"""
        status = {}
        
        for model_type, pipeline in self.loaded_pipelines.items():
            vram_info = self.vram_optimizer.get_vram_usage()
            status[model_type] = {
                "loaded": True,
                "model_type": model_type,
                "device": str(next(pipeline.parameters()).device) if hasattr(pipeline, 'parameters') else "unknown",
                "vram_usage_mb": vram_info.get("used_mb", 0)
            }
        
        return status


# Global generation engine instance
_generation_engine = None

def get_generation_engine() -> VideoGenerationEngine:
    """Get the global video generation engine instance"""
    global _generation_engine
    if _generation_engine is None:
        _generation_engine = VideoGenerationEngine()
    return _generation_engine


# Convenience functions for video generation
def generate_video(model_type: str, prompt: str, image: Optional[Image.Image] = None,
                  end_image: Optional[Image.Image] = None, resolution: str = "1280x720", 
                  num_inference_steps: int = 50, guidance_scale: float = 7.5, **kwargs) -> Dict[str, Any]:
    """Generate video using the specified model type and parameters with optional end frame"""
    engine = get_generation_engine()
    return engine.generate_video(model_type, prompt, image, end_image, resolution, 
                               num_inference_steps, guidance_scale, **kwargs)

def generate_t2v_video(prompt: str, resolution: str = "1280x720", 
                      num_inference_steps: int = 50, guidance_scale: float = 7.5,
                      **kwargs) -> Dict[str, Any]:
    """Generate Text-to-Video"""
    engine = get_generation_engine()
    return engine.generate_t2v(prompt, resolution, num_inference_steps, guidance_scale, **kwargs)

def generate_i2v_video(prompt: str, image: Image.Image, resolution: str = "1280x720",
                      num_inference_steps: int = 50, guidance_scale: float = 7.5,
                      **kwargs) -> Dict[str, Any]:
    """Generate Image-to-Video"""
    engine = get_generation_engine()
    return engine.generate_i2v(prompt, image, resolution, num_inference_steps, guidance_scale, **kwargs)

def generate_ti2v_video(prompt: str, image: Image.Image, resolution: str = "1280x720",
                       num_inference_steps: int = 50, guidance_scale: float = 7.5,
                       **kwargs) -> Dict[str, Any]:
    """Generate Text-Image-to-Video"""
    engine = get_generation_engine()
    return engine.generate_ti2v(prompt, image, resolution, num_inference_steps, guidance_scale, **kwargs)


# Input Validation and Preprocessing
class InputValidator:
    """Handles validation and preprocessing of user inputs for video generation"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.generation_config = self.config.get("generation", {})
        
        # Validation parameters
        self.max_prompt_length = self.generation_config.get("max_prompt_length", 500)
        self.supported_resolutions = self.generation_config.get("supported_resolutions", 
                                                               ["854x480", "480x854", "1280x720", "1280x704", "1920x1080"])
        self.supported_image_formats = ["PNG", "JPG", "JPEG", "WebP", "BMP", "TIFF"]
        self.max_image_size_mb = self.generation_config.get("max_image_size_mb", 10)
        self.min_image_dimension = 64
        self.max_image_dimension = 2048
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {
                "generation": {
                    "max_prompt_length": 500,
                    "supported_resolutions": ["854x480", "480x854", "1280x720", "1280x704", "1920x1080"],
                    "max_image_size_mb": 10
                }
            }
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, str, str]:
        """
        Validate and preprocess text prompt
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Tuple of (is_valid, processed_prompt, error_message)
        """
        if not prompt or not isinstance(prompt, str):
            return False, "", "Prompt cannot be empty"
        
        # Strip whitespace
        prompt = prompt.strip()
        
        if not prompt:
            return False, "", "Prompt cannot be empty after removing whitespace"
        
        # Check length
        if len(prompt) > self.max_prompt_length:
            return False, prompt[:self.max_prompt_length], f"Prompt exceeds maximum length of {self.max_prompt_length} characters"
        
        # Filter out potentially problematic characters
        # Remove control characters but keep newlines and tabs
        filtered_prompt = ''.join(char for char in prompt 
                                if ord(char) >= 32 or char in '\n\t')
        
        # Check for minimum meaningful content
        if len(filtered_prompt.strip()) < 3:
            return False, filtered_prompt, "Prompt must contain at least 3 meaningful characters"
        
        # Basic content validation - check for common issues
        if filtered_prompt.count('"') % 2 != 0:
            logger.warning("Unmatched quotes in prompt, this may cause issues")
        
        return True, filtered_prompt, ""
    
    def validate_image(self, image: Union[Image.Image, str, bytes]) -> Tuple[bool, Optional[Image.Image], str]:
        """
        Validate and preprocess input image
        
        Args:
            image: PIL Image, file path, or image bytes
            
        Returns:
            Tuple of (is_valid, processed_image, error_message)
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                # File path
                if not os.path.exists(image):
                    return False, None, f"Image file not found: {image}"
                
                # Check file size
                file_size_mb = os.path.getsize(image) / (1024 * 1024)
                if file_size_mb > self.max_image_size_mb:
                    return False, None, f"Image file too large: {file_size_mb:.1f}MB (max: {self.max_image_size_mb}MB)"
                
                # Load image
                image = Image.open(image)
                
            elif isinstance(image, bytes):
                # Image bytes
                if len(image) > self.max_image_size_mb * 1024 * 1024:
                    return False, None, f"Image data too large (max: {self.max_image_size_mb}MB)"
                
                from io import BytesIO
                image = Image.open(BytesIO(image))
                
            elif not isinstance(image, Image.Image):
                return False, None, "Invalid image type. Expected PIL Image, file path, or bytes"
            
            # Validate image format
            if image.format not in self.supported_image_formats:
                logger.warning(f"Unsupported image format: {image.format}, attempting conversion")
                # Try to convert to RGB
                if image.mode not in ['RGB', 'RGBA']:
                    image = image.convert('RGB')
            
            # Validate image dimensions
            width, height = image.size
            
            if width < self.min_image_dimension or height < self.min_image_dimension:
                return False, None, f"Image too small: {width}x{height} (minimum: {self.min_image_dimension}x{self.min_image_dimension})"
            
            if width > self.max_image_dimension or height > self.max_image_dimension:
                logger.warning(f"Large image detected: {width}x{height}, consider resizing for better performance")
            
            # Convert to RGB if necessary (remove alpha channel)
            if image.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            return True, image, ""
            
        except Exception as e:
            return False, None, f"Failed to process image: {str(e)}"
    
    def validate_resolution(self, resolution: str) -> Tuple[bool, Tuple[int, int], str]:
        """
        Validate and parse resolution string
        
        Args:
            resolution: Resolution string (e.g., "1280x720")
            
        Returns:
            Tuple of (is_valid, (width, height), error_message)
        """
        if not resolution or not isinstance(resolution, str):
            return False, (0, 0), "Resolution cannot be empty"
        
        # Check if resolution is in supported list
        if resolution not in self.supported_resolutions:
            return False, (0, 0), f"Unsupported resolution: {resolution}. Supported: {', '.join(self.supported_resolutions)}"
        
        try:
            # Parse resolution
            parts = resolution.split('x')
            if len(parts) != 2:
                return False, (0, 0), "Resolution must be in format 'WIDTHxHEIGHT'"
            
            width, height = int(parts[0]), int(parts[1])
            
            # Validate dimensions
            if width <= 0 or height <= 0:
                return False, (0, 0), "Resolution dimensions must be positive"
            
            # Check if dimensions are reasonable
            if width < 256 or height < 256:
                return False, (0, 0), "Resolution too small (minimum 256x256)"
            
            if width > 2048 or height > 2048:
                return False, (0, 0), "Resolution too large (maximum 2048x2048)"
            
            # Check aspect ratio (should be reasonable)
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                logger.warning(f"Unusual aspect ratio: {aspect_ratio:.2f}")
            
            return True, (width, height), ""
            
        except ValueError:
            return False, (0, 0), "Invalid resolution format. Use 'WIDTHxHEIGHT' (e.g., '1280x720')"
    
    def scale_image_to_resolution(self, image: Image.Image, target_resolution: str, 
                                 maintain_aspect: bool = True) -> Image.Image:
        """
        Scale image to target resolution
        
        Args:
            image: Input PIL Image
            target_resolution: Target resolution string (e.g., "1280x720")
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Scaled PIL Image
        """
        is_valid, (target_width, target_height), error = self.validate_resolution(target_resolution)
        if not is_valid:
            raise ValueError(f"Invalid target resolution: {error}")
        
        current_width, current_height = image.size
        
        if current_width == target_width and current_height == target_height:
            return image  # Already correct size
        
        if maintain_aspect:
            # Calculate scaling to fit within target dimensions
            scale_x = target_width / current_width
            scale_y = target_height / current_height
            scale = min(scale_x, scale_y)
            
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            # Resize image
            scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create canvas with target dimensions and center the scaled image
            canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            canvas.paste(scaled_image, (paste_x, paste_y))
            
            return canvas
        else:
            # Direct resize to target dimensions
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    def validate_generation_parameters(self, model_type: str, prompt: str, 
                                     image: Optional[Image.Image] = None,
                                     resolution: str = "1280x720",
                                     num_inference_steps: int = 50,
                                     guidance_scale: float = 7.5) -> Tuple[bool, Dict[str, Any], str]:
        """
        Validate all generation parameters
        
        Args:
            model_type: Generation model type
            prompt: Text prompt
            image: Optional input image
            resolution: Target resolution
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale value
            
        Returns:
            Tuple of (is_valid, validated_params, error_message)
        """
        errors = []
        validated_params = {}
        
        # Validate model type
        valid_model_types = ["t2v", "i2v", "ti2v"]
        if model_type.lower() not in valid_model_types:
            errors.append(f"Invalid model type: {model_type}. Supported: {', '.join(valid_model_types)}")
        else:
            validated_params["model_type"] = model_type.lower()
        
        # Validate prompt
        prompt_valid, processed_prompt, prompt_error = self.validate_prompt(prompt)
        if not prompt_valid:
            errors.append(f"Prompt validation failed: {prompt_error}")
        else:
            validated_params["prompt"] = processed_prompt
        
        # Validate image (if required)
        if model_type.lower() in ["i2v", "ti2v"]:
            if image is None:
                errors.append(f"Image input is required for {model_type.upper()} generation")
            else:
                image_valid, processed_image, image_error = self.validate_image(image)
                if not image_valid:
                    errors.append(f"Image validation failed: {image_error}")
                else:
                    validated_params["image"] = processed_image
        elif image is not None:
            logger.warning(f"Image provided for {model_type.upper()} generation but not required")
        
        # Validate resolution
        res_valid, (width, height), res_error = self.validate_resolution(resolution)
        if not res_valid:
            errors.append(f"Resolution validation failed: {res_error}")
        else:
            validated_params["resolution"] = resolution
            validated_params["width"] = width
            validated_params["height"] = height
        
        # Validate inference steps
        if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
            errors.append("Number of inference steps must be a positive integer")
        elif num_inference_steps > 200:
            errors.append("Number of inference steps too high (maximum: 200)")
        else:
            validated_params["num_inference_steps"] = num_inference_steps
        
        # Validate guidance scale
        if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0:
            errors.append("Guidance scale must be a non-negative number")
        elif guidance_scale > 20:
            errors.append("Guidance scale too high (maximum: 20)")
        else:
            validated_params["guidance_scale"] = float(guidance_scale)
        
        if errors:
            return False, {}, "; ".join(errors)
        
        return True, validated_params, ""
    
    def preprocess_for_generation(self, model_type: str, prompt: str,
                                image: Optional[Image.Image] = None,
                                resolution: str = "1280x720") -> Dict[str, Any]:
        """
        Preprocess all inputs for generation
        
        Args:
            model_type: Generation model type
            prompt: Text prompt
            image: Optional input image
            resolution: Target resolution
            
        Returns:
            Dictionary of preprocessed parameters
        """
        # Validate all parameters
        is_valid, validated_params, error = self.validate_generation_parameters(
            model_type, prompt, image, resolution
        )
        
        if not is_valid:
            raise ValueError(f"Input validation failed: {error}")
        
        # Additional preprocessing
        processed_params = validated_params.copy()
        
        # Scale image if provided
        if "image" in processed_params:
            processed_params["image"] = self.scale_image_to_resolution(
                processed_params["image"], resolution, maintain_aspect=True
            )
        
        return processed_params


# Global input validator instance
_input_validator = None

def get_input_validator() -> InputValidator:
    """Get the global input validator instance"""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


# Convenience functions for input validation
def validate_prompt(prompt: str) -> Tuple[bool, str, str]:
    """Validate text prompt"""
    validator = get_input_validator()
    return validator.validate_prompt(prompt)

def validate_image(image: Union[Image.Image, str, bytes]) -> Tuple[bool, Optional[Image.Image], str]:
    """Validate input image"""
    validator = get_input_validator()
    return validator.validate_image(image)

def validate_resolution(resolution: str) -> Tuple[bool, Tuple[int, int], str]:
    """Validate resolution string"""
    validator = get_input_validator()
    return validator.validate_resolution(resolution)

def validate_generation_parameters(model_type: str, prompt: str, 
                                 image: Optional[Image.Image] = None,
                                 resolution: str = "1280x720",
                                 num_inference_steps: int = 50,
                                 guidance_scale: float = 7.5) -> Tuple[bool, Dict[str, Any], str]:
    """Validate all generation parameters"""
    validator = get_input_validator()
    return validator.validate_generation_parameters(
        model_type, prompt, image, resolution, num_inference_steps, guidance_scale
    )

def preprocess_for_generation(model_type: str, prompt: str,
                            image: Optional[Image.Image] = None,
                            resolution: str = "1280x720") -> Dict[str, Any]:
    """Preprocess inputs for generation"""
    validator = get_input_validator()
    return validator.preprocess_for_generation(model_type, prompt, image, resolution)


# Error Handling and Recovery System
import time
import traceback
from enum import Enum
from typing import Callable, List

class ErrorType(Enum):
    """Types of errors that can occur during generation"""
    VRAM_OUT_OF_MEMORY = "vram_oom"
    MODEL_LOADING_ERROR = "model_loading"
    GENERATION_TIMEOUT = "generation_timeout"
    INPUT_VALIDATION_ERROR = "input_validation"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class GenerationError:
    """Represents an error that occurred during generation"""
    error_type: ErrorType
    message: str
    original_exception: Optional[Exception] = None
    timestamp: datetime = None
    recovery_suggestions: List[str] = None
    is_recoverable: bool = True
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []

class ErrorHandler:
    """Handles errors and implements recovery strategies for video generation"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.error_config = self.config.get("error_handling", {})
        self.vram_optimizer = get_vram_optimizer()
        
        # Recovery settings
        self.max_retries = self.error_config.get("max_retries", 3)
        self.retry_delay_seconds = self.error_config.get("retry_delay_seconds", 5)
        self.timeout_seconds = self.error_config.get("generation_timeout_seconds", 1800)  # 30 minutes
        self.vram_threshold_percent = self.error_config.get("vram_warning_threshold", 90)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {
                "error_handling": {
                    "max_retries": 3,
                    "retry_delay_seconds": 5,
                    "generation_timeout_seconds": 1800,
                    "vram_warning_threshold": 90
                }
            }
    
    def classify_error(self, exception: Exception) -> ErrorType:
        """Classify an exception into an error type"""
        error_message = str(exception).lower()
        
        # VRAM out of memory errors
        if any(keyword in error_message for keyword in [
            "out of memory", "cuda out of memory", "vram", "memory_limit"
        ]):
            return ErrorType.VRAM_OUT_OF_MEMORY
        
        # Model loading errors
        if any(keyword in error_message for keyword in [
            "model not found", "failed to load", "download", "huggingface", "repository"
        ]):
            return ErrorType.MODEL_LOADING_ERROR
        
        # Network errors
        if any(keyword in error_message for keyword in [
            "connection", "network", "timeout", "http", "ssl", "certificate"
        ]):
            return ErrorType.NETWORK_ERROR
        
        # Input validation errors
        if any(keyword in error_message for keyword in [
            "validation", "invalid input", "unsupported", "format"
        ]):
            return ErrorType.INPUT_VALIDATION_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def create_error(self, exception: Exception, context: str = "") -> GenerationError:
        """Create a GenerationError from an exception"""
        error_type = self.classify_error(exception)
        
        # Generate recovery suggestions based on error type
        suggestions = self._get_recovery_suggestions(error_type, exception)
        
        # Determine if error is recoverable
        is_recoverable = error_type in [
            ErrorType.VRAM_OUT_OF_MEMORY,
            ErrorType.MODEL_LOADING_ERROR,
            ErrorType.NETWORK_ERROR
        ]
        
        message = f"{context}: {str(exception)}" if context else str(exception)
        
        return GenerationError(
            error_type=error_type,
            message=message,
            original_exception=exception,
            recovery_suggestions=suggestions,
            is_recoverable=is_recoverable
        )
    
    def _get_recovery_suggestions(self, error_type: ErrorType, exception: Exception) -> List[str]:
        """Get recovery suggestions for a specific error type"""
        suggestions = []
        
        if error_type == ErrorType.VRAM_OUT_OF_MEMORY:
            suggestions = [
                "Try reducing the resolution (e.g., use 1280x720 instead of 1920x1080)",
                "Enable model offloading in optimization settings",
                "Use int8 quantization to reduce memory usage",
                "Reduce VAE tile size to 128 or 256",
                "Close other GPU-intensive applications",
                "Restart the application to clear GPU memory"
            ]
        
        elif error_type == ErrorType.MODEL_LOADING_ERROR:
            suggestions = [
                "Check your internet connection",
                "Verify the model name is correct",
                "Clear model cache and try downloading again",
                "Check available disk space",
                "Try downloading the model manually"
            ]
        
        elif error_type == ErrorType.GENERATION_TIMEOUT:
            suggestions = [
                "Reduce the number of inference steps",
                "Use a lower resolution",
                "Enable optimization settings (offloading, quantization)",
                "Check system resources and close unnecessary applications"
            ]
        
        elif error_type == ErrorType.NETWORK_ERROR:
            suggestions = [
                "Check your internet connection",
                "Try again in a few minutes",
                "Check if Hugging Face Hub is accessible",
                "Verify firewall settings"
            ]
        
        elif error_type == ErrorType.INPUT_VALIDATION_ERROR:
            suggestions = [
                "Check that your prompt is not empty and under 500 characters",
                "Ensure image is in a supported format (PNG, JPG, WebP)",
                "Verify image size is under 10MB",
                "Use a supported resolution (854x480, 480x854, 1280x720, 1280x704, 1920x1080)"
            ]
        
        return suggestions
    
    def handle_vram_oom_error(self, exception: Exception) -> Dict[str, Any]:
        """Handle VRAM out of memory errors with automatic recovery"""
        logger.error(f"VRAM out of memory error: {exception}")
        
        # Clear GPU cache immediately
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
        
        # Get current VRAM usage
        vram_info = self.vram_optimizer.get_vram_usage()
        
        recovery_actions = []
        
        # Suggest optimizations based on current state
        if vram_info["usage_percent"] > 80:
            recovery_actions.append("enable_offloading")
        
        if vram_info["usage_percent"] > 90:
            recovery_actions.append("use_int8_quantization")
            recovery_actions.append("reduce_vae_tile_size")
        
        return {
            "error_type": "vram_oom",
            "vram_info": vram_info,
            "recovery_actions": recovery_actions,
            "suggestions": self._get_recovery_suggestions(ErrorType.VRAM_OUT_OF_MEMORY, exception)
        }
    
    def handle_model_loading_error(self, exception: Exception, model_type: str) -> Dict[str, Any]:
        """Handle model loading errors with retry logic"""
        logger.error(f"Model loading error for {model_type}: {exception}")
        
        # Clear any partial downloads
        try:
            model_manager = get_model_manager()
            model_manager.clear_cache(model_type)
            logger.info(f"Cleared cache for {model_type}")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
        
        return {
            "error_type": "model_loading",
            "model_type": model_type,
            "suggestions": self._get_recovery_suggestions(ErrorType.MODEL_LOADING_ERROR, exception)
        }
    
    def handle_generation_timeout(self, timeout_seconds: int) -> Dict[str, Any]:
        """Handle generation timeout errors"""
        logger.error(f"Generation timed out after {timeout_seconds} seconds")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "error_type": "generation_timeout",
            "timeout_seconds": timeout_seconds,
            "suggestions": self._get_recovery_suggestions(ErrorType.GENERATION_TIMEOUT, 
                                                        Exception(f"Timeout after {timeout_seconds}s"))
        }
    
    def retry_with_backoff(self, func: Callable, *args, max_retries: int = None, 
                          delay_seconds: float = None, **kwargs) -> Any:
        """
        Retry a function with exponential backoff
        
        Args:
            func: Function to retry
            *args: Function arguments
            max_retries: Maximum number of retries
            delay_seconds: Initial delay between retries
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises the last exception
        """
        if max_retries is None:
            max_retries = self.max_retries
        if delay_seconds is None:
            delay_seconds = self.retry_delay_seconds
        
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                error_type = self.classify_error(e)
                
                # Don't retry non-recoverable errors
                if error_type == ErrorType.INPUT_VALIDATION_ERROR:
                    raise e
                
                if attempt < max_retries:
                    wait_time = delay_seconds * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    
                    # Special handling for VRAM errors
                    if error_type == ErrorType.VRAM_OUT_OF_MEMORY:
                        self.handle_vram_oom_error(e)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        # If we get here, all retries failed
        raise last_exception
    
    def with_timeout(self, func: Callable, timeout_seconds: float = None, 
                    *args, **kwargs) -> Any:
        """
        Execute a function with a timeout
        
        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises TimeoutError
        """
        if timeout_seconds is None:
            timeout_seconds = self.timeout_seconds
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
        
        # Set up timeout signal (Unix only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
        
        try:
            result = func(*args, **kwargs)
            return result
        
        except TimeoutError:
            # Handle timeout
            timeout_info = self.handle_generation_timeout(timeout_seconds)
            raise TimeoutError(f"Generation timed out: {timeout_info}")
        
        finally:
            # Clean up timeout signal
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    def safe_generate(self, generation_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Safely execute a generation function with comprehensive error handling
        
        Args:
            generation_func: Generation function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with result or error information
        """
        start_time = time.time()
        
        try:
            # Check VRAM before generation
            vram_info = self.vram_optimizer.get_vram_usage()
            if vram_info["usage_percent"] > self.vram_threshold_percent:
                logger.warning(f"High VRAM usage detected: {vram_info['usage_percent']:.1f}%")
            
            # Execute with timeout and retry logic
            result = self.retry_with_backoff(
                lambda: self.with_timeout(generation_func, self.timeout_seconds, *args, **kwargs)
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Generation completed successfully in {execution_time:.1f}s")
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "vram_info": self.vram_optimizer.get_vram_usage()
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            error = self.create_error(e, "Generation failed")
            
            logger.error(f"Generation failed after {execution_time:.1f}s: {error.message}")
            
            # Handle specific error types
            error_info = {"error_type": error.error_type.value}
            
            if error.error_type == ErrorType.VRAM_OUT_OF_MEMORY:
                error_info.update(self.handle_vram_oom_error(e))
            elif error.error_type == ErrorType.MODEL_LOADING_ERROR:
                model_type = kwargs.get('model_type', 'unknown')
                error_info.update(self.handle_model_loading_error(e, model_type))
            elif error.error_type == ErrorType.GENERATION_TIMEOUT:
                error_info.update(self.handle_generation_timeout(self.timeout_seconds))
            
            return {
                "success": False,
                "error": error,
                "error_info": error_info,
                "execution_time": execution_time,
                "vram_info": self.vram_optimizer.get_vram_usage()
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        vram_info = self.vram_optimizer.get_vram_usage()
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "vram": {
                "usage_percent": vram_info["usage_percent"],
                "used_mb": vram_info["used_mb"],
                "total_mb": vram_info["total_mb"],
                "status": "healthy" if vram_info["usage_percent"] < 80 else "warning" if vram_info["usage_percent"] < 95 else "critical"
            },
            "cpu": {
                "usage_percent": cpu_percent,
                "status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical"
            },
            "memory": {
                "usage_percent": memory.percent,
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3),
                "status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 95 else "critical"
            }
        }
        
        # Overall health status
        statuses = [health_status["vram"]["status"], health_status["cpu"]["status"], health_status["memory"]["status"]]
        if "critical" in statuses:
            health_status["overall_status"] = "critical"
        elif "warning" in statuses:
            health_status["overall_status"] = "warning"
        else:
            health_status["overall_status"] = "healthy"
        
        return health_status


# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


# Convenience functions for error handling
def safe_generate_video(model_type: str, prompt: str, image: Optional[Image.Image] = None,
                       resolution: str = "1280x720", **kwargs) -> Dict[str, Any]:
    """Safely generate video with comprehensive error handling"""
    error_handler = get_error_handler()
    generation_engine = get_generation_engine()
    
    return error_handler.safe_generate(
        generation_engine.generate_video,
        model_type, prompt, image, resolution, **kwargs
    )

def handle_generation_error(exception: Exception, context: str = "") -> GenerationError:
    """Handle and classify a generation error"""
    error_handler = get_error_handler()
    return error_handler.create_error(exception, context)

def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    error_handler = get_error_handler()
    return error_handler.get_system_health()

def retry_with_backoff(func: Callable, *args, max_retries: int = 3, **kwargs) -> Any:
    """Retry a function with exponential backoff"""
    error_handler = get_error_handler()
    return error_handler.retry_with_backoff(func, *args, max_retries=max_retries, **kwargs)
# LoRA Weight Management System
class LoRAManager:
    """Manages LoRA weights loading, application, and strength adjustment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loras_directory = Path(config["directories"]["loras_directory"])
        self.loras_directory.mkdir(exist_ok=True)
        
        # Track loaded LoRAs
        self.loaded_loras: Dict[str, Dict[str, Any]] = {}
        self.applied_loras: Dict[str, float] = {}  # LoRA name -> strength
        
    def list_available_loras(self) -> Dict[str, Dict[str, Any]]:
        """List all available LoRA files in the loras directory"""
        loras = {}
        
        # Supported LoRA file extensions
        lora_extensions = ['.safetensors', '.pt', '.pth', '.bin']
        
        for lora_file in self.loras_directory.iterdir():
            if lora_file.is_file() and lora_file.suffix.lower() in lora_extensions:
                lora_name = lora_file.stem
                
                # Get file info
                stat = lora_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                
                loras[lora_name] = {
                    "path": str(lora_file),
                    "filename": lora_file.name,
                    "size_mb": size_mb,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_loaded": lora_name in self.loaded_loras,
                    "is_applied": lora_name in self.applied_loras,
                    "current_strength": self.applied_loras.get(lora_name, 0.0)
                }
        
        return loras
    
    def load_lora(self, lora_name: str) -> Dict[str, Any]:
        """Load a LoRA from file"""
        if lora_name in self.loaded_loras:
            logger.info(f"LoRA {lora_name} already loaded")
            return self.loaded_loras[lora_name]
        
        # Find the LoRA file
        lora_file = None
        lora_extensions = ['.safetensors', '.pt', '.pth', '.bin']
        
        for ext in lora_extensions:
            potential_file = self.loras_directory / f"{lora_name}{ext}"
            if potential_file.exists():
                lora_file = potential_file
                break
        
        if not lora_file:
            raise FileNotFoundError(f"LoRA file not found: {lora_name}")
        
        logger.info(f"Loading LoRA: {lora_name} from {lora_file}")
        
        try:
            # Load LoRA weights
            if lora_file.suffix == '.safetensors':
                from safetensors.torch import load_file
                lora_weights = load_file(str(lora_file))
            else:
                lora_weights = torch.load(str(lora_file), map_location='cpu')
            
            # Validate LoRA structure
            if not self._validate_lora_weights(lora_weights):
                raise ValueError(f"Invalid LoRA structure in {lora_name}")
            
            # Store loaded LoRA
            lora_info = {
                "name": lora_name,
                "path": str(lora_file),
                "weights": lora_weights,
                "loaded_at": datetime.now(),
                "num_layers": len([k for k in lora_weights.keys() if 'lora_up' in k]),
                "size_mb": lora_file.stat().st_size / (1024 * 1024)
            }
            
            self.loaded_loras[lora_name] = lora_info
            logger.info(f"Successfully loaded LoRA: {lora_name} ({lora_info['num_layers']} layers)")
            
            return lora_info
            
        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_name}: {e}")
            raise
    
    def _validate_lora_weights(self, weights: Dict[str, torch.Tensor]) -> bool:
        """Validate that the loaded weights have proper LoRA structure"""
        try:
            # Check for LoRA-specific keys
            lora_keys = [k for k in weights.keys() if 'lora_up' in k or 'lora_down' in k]
            
            if not lora_keys:
                logger.warning("No LoRA keys found in weights")
                return False
            
            # Check that up and down weights are paired
            up_keys = [k for k in weights.keys() if 'lora_up' in k]
            down_keys = [k for k in weights.keys() if 'lora_down' in k]
            
            if len(up_keys) != len(down_keys):
                logger.warning("Mismatched LoRA up/down weights")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating LoRA weights: {e}")
            return False
    
    def apply_lora(self, model, lora_name: str, strength: float = 1.0):
        """Apply a LoRA to a model with specified strength"""
        if strength < 0.0 or strength > 2.0:
            raise ValueError("LoRA strength must be between 0.0 and 2.0")
        
        # Load LoRA if not already loaded
        if lora_name not in self.loaded_loras:
            self.load_lora(lora_name)
        
        lora_info = self.loaded_loras[lora_name]
        lora_weights = lora_info["weights"]
        
        logger.info(f"Applying LoRA {lora_name} with strength {strength}")
        
        try:
            # Check if model supports LoRA loading (diffusers pipeline)
            if hasattr(model, 'load_lora_weights'):
                # Use diffusers built-in LoRA loading
                model.load_lora_weights(lora_info["path"])
                
                # Set LoRA strength if supported
                if hasattr(model, 'set_adapters'):
                    model.set_adapters([lora_name], adapter_weights=[strength])
                elif hasattr(model, 'fuse_lora') and hasattr(model, 'unfuse_lora'):
                    # Fuse LoRA with specified strength
                    model.fuse_lora(lora_scale=strength)
                
                logger.info(f"Applied LoRA using diffusers built-in method")
                
            else:
                # Manual LoRA application
                self._apply_lora_manual(model, lora_weights, strength)
                logger.info(f"Applied LoRA using manual method")
            
            # Track applied LoRA
            self.applied_loras[lora_name] = strength
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA {lora_name}: {e}")
            raise
    
    def _apply_lora_manual(self, model, lora_weights: Dict[str, torch.Tensor], strength: float):
        """Manually apply LoRA weights to model"""
        # Get the UNet component (main target for LoRA)
        target_model = model.unet if hasattr(model, 'unet') else model
        
        # Apply LoRA weights
        for key, weight in lora_weights.items():
            if 'lora_up' in key or 'lora_down' in key:
                # Extract the base layer name
                base_key = key.replace('.lora_up.weight', '').replace('.lora_down.weight', '')
                
                # Find corresponding model parameter
                try:
                    # Navigate to the parameter
                    param_path = base_key.split('.')
                    current_module = target_model
                    
                    for attr in param_path[:-1]:
                        current_module = getattr(current_module, attr)
                    
                    param_name = param_path[-1]
                    
                    if hasattr(current_module, param_name):
                        original_param = getattr(current_module, param_name)
                        
                        # Apply LoRA modification
                        if 'lora_up' in key:
                            up_weight = weight
                            down_key = key.replace('lora_up', 'lora_down')
                            
                            if down_key in lora_weights:
                                down_weight = lora_weights[down_key]
                                
                                # Calculate LoRA delta: strength * (up @ down)
                                lora_delta = strength * torch.mm(up_weight, down_weight)
                                
                                # Add to original parameter
                                with torch.no_grad():
                                    original_param.data += lora_delta.to(original_param.device, original_param.dtype)
                
                except Exception as e:
                    logger.warning(f"Failed to apply LoRA weight {key}: {e}")
                    continue
    
    def remove_lora(self, model, lora_name: str):
        """Remove a LoRA from a model"""
        if lora_name not in self.applied_loras:
            logger.warning(f"LoRA {lora_name} is not currently applied")
            return model
        
        logger.info(f"Removing LoRA: {lora_name}")
        
        try:
            # Check if model supports LoRA removal (diffusers pipeline)
            if hasattr(model, 'unfuse_lora'):
                model.unfuse_lora()
                logger.info("Removed LoRA using diffusers built-in method")
            elif hasattr(model, 'unload_lora_weights'):
                model.unload_lora_weights()
                logger.info("Unloaded LoRA weights using diffusers method")
            else:
                # Manual LoRA removal would require storing original weights
                logger.warning("Manual LoRA removal not implemented - consider reloading the model")
            
            # Remove from tracking
            if lora_name in self.applied_loras:
                del self.applied_loras[lora_name]
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to remove LoRA {lora_name}: {e}")
            raise
    
    def adjust_lora_strength(self, model, lora_name: str, new_strength: float):
        """Adjust the strength of an applied LoRA"""
        if lora_name not in self.applied_loras:
            raise ValueError(f"LoRA {lora_name} is not currently applied")
        
        if new_strength < 0.0 or new_strength > 2.0:
            raise ValueError("LoRA strength must be between 0.0 and 2.0")
        
        logger.info(f"Adjusting LoRA {lora_name} strength from {self.applied_loras[lora_name]} to {new_strength}")
        
        try:
            # For diffusers models with adapter support
            if hasattr(model, 'set_adapters'):
                model.set_adapters([lora_name], adapter_weights=[new_strength])
                self.applied_loras[lora_name] = new_strength
                logger.info(f"Adjusted LoRA strength using diffusers method")
            else:
                # For manual adjustment, we need to remove and reapply
                self.remove_lora(model, lora_name)
                self.apply_lora(model, lora_name, new_strength)
                logger.info(f"Adjusted LoRA strength using remove/reapply method")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to adjust LoRA strength: {e}")
            raise
    
    def get_fallback_prompt_enhancement(self, base_prompt: str, lora_name: str) -> str:
        """Generate prompt-based enhancement when LoRA is unavailable"""
        logger.info(f"Generating fallback prompt enhancement for LoRA: {lora_name}")
        
        # Common LoRA types and their prompt enhancements
        enhancement_mappings = {
            # Style LoRAs
            "anime": "anime style, detailed anime art, vibrant colors",
            "realistic": "photorealistic, highly detailed, professional photography",
            "cartoon": "cartoon style, animated, colorful illustration",
            "oil_painting": "oil painting style, artistic brushstrokes, classical art",
            "watercolor": "watercolor painting, soft colors, artistic medium",
            
            # Quality LoRAs
            "detail": "extremely detailed, high quality, sharp focus, intricate details",
            "quality": "masterpiece, best quality, high resolution, detailed",
            "sharp": "sharp focus, crisp details, high definition",
            
            # Lighting LoRAs
            "dramatic_lighting": "dramatic lighting, cinematic lighting, moody atmosphere",
            "soft_lighting": "soft lighting, gentle illumination, warm tones",
            "neon": "neon lighting, cyberpunk style, glowing effects",
            
            # Character LoRAs (generic enhancements)
            "character": "detailed character design, expressive features",
            "portrait": "portrait style, detailed facial features, professional headshot"
        }
        
        # Try to match LoRA name to enhancement type
        lora_lower = lora_name.lower()
        enhancement = None
        
        for key, value in enhancement_mappings.items():
            if key in lora_lower:
                enhancement = value
                break
        
        # If no specific enhancement found, use generic quality enhancement
        if not enhancement:
            enhancement = "high quality, detailed, enhanced style"
        
        # Combine with base prompt
        if base_prompt.strip():
            enhanced_prompt = f"{base_prompt}, {enhancement}"
        else:
            enhanced_prompt = enhancement
        
        logger.info(f"Generated fallback enhancement: '{enhancement}'")
        return enhanced_prompt
    
    def unload_lora(self, lora_name: str):
        """Unload a LoRA from memory"""
        if lora_name in self.loaded_loras:
            del self.loaded_loras[lora_name]
            logger.info(f"Unloaded LoRA from memory: {lora_name}")
        
        if lora_name in self.applied_loras:
            del self.applied_loras[lora_name]
    
    def get_lora_status(self, lora_name: str) -> Dict[str, Any]:
        """Get comprehensive status information for a LoRA"""
        available_loras = self.list_available_loras()
        
        if lora_name not in available_loras:
            return {
                "name": lora_name,
                "exists": False,
                "is_loaded": False,
                "is_applied": False,
                "current_strength": 0.0
            }
        
        lora_info = available_loras[lora_name]
        
        return {
            "name": lora_name,
            "exists": True,
            "path": lora_info["path"],
            "size_mb": lora_info["size_mb"],
            "is_loaded": lora_info["is_loaded"],
            "is_applied": lora_info["is_applied"],
            "current_strength": lora_info["current_strength"],
            "modified_time": lora_info["modified_time"]
        }
    
    # UI Integration Methods
    def upload_lora_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Handle file uploads with validation for UI integration"""
        try:
            file_path_obj = Path(file_path)
            filename_obj = Path(filename)
            
            # Validate file extension
            valid_extensions = ['.safetensors', '.pt', '.pth', '.bin']
            if filename_obj.suffix.lower() not in valid_extensions:
                return {
                    "success": False,
                    "error": f"Invalid file format. Supported formats: {', '.join(valid_extensions)}"
                }
            
            # Check file size
            max_size_mb = self.config.get("lora_max_file_size_mb", 500)
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            
            if file_size_mb > max_size_mb:
                return {
                    "success": False,
                    "error": f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {max_size_mb}MB"
                }
            
            # Check if file already exists and handle duplicates
            target_path = self.loras_directory / filename
            original_filename = filename_obj.stem
            extension = filename_obj.suffix
            counter = 1
            
            while target_path.exists():
                new_filename = f"{original_filename}_{counter}{extension}"
                target_path = self.loras_directory / new_filename
                filename = new_filename
                counter += 1
            
            # Validate LoRA structure before copying
            try:
                if filename_obj.suffix == '.safetensors':
                    from safetensors.torch import load_file
                    test_weights = load_file(str(file_path_obj))
                else:
                    test_weights = torch.load(str(file_path_obj), map_location='cpu')
                
                if not self._validate_lora_weights(test_weights):
                    return {
                        "success": False,
                        "error": "Invalid LoRA structure. File does not contain valid LoRA weights."
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to validate LoRA file: {str(e)}"
                }
            
            # Copy file to LoRA directory
            shutil.copy2(file_path_obj, target_path)
            
            logger.info(f"Successfully uploaded LoRA: {filename} ({file_size_mb:.1f}MB)")
            
            return {
                "success": True,
                "message": f"Successfully uploaded LoRA: {filename}",
                "filename": filename,
                "file_path": str(target_path),
                "size_mb": file_size_mb
            }
            
        except Exception as e:
            logger.error(f"Error uploading LoRA file: {e}")
            return {
                "success": False,
                "error": f"Upload failed: {str(e)}"
            }
    
    def delete_lora_file(self, lora_name: str) -> Dict[str, Any]:
        """Delete a LoRA file from the filesystem"""
        try:
            # Find the LoRA file
            lora_file = None
            lora_extensions = ['.safetensors', '.pt', '.pth', '.bin']
            
            for ext in lora_extensions:
                potential_file = self.loras_directory / f"{lora_name}{ext}"
                if potential_file.exists():
                    lora_file = potential_file
                    break
            
            if not lora_file:
                return {
                    "success": False,
                    "error": f"LoRA file not found: {lora_name}"
                }
            
            # Check if LoRA is currently loaded or applied
            if lora_name in self.loaded_loras:
                logger.warning(f"Deleting loaded LoRA: {lora_name}")
                self.unload_lora(lora_name)
            
            # Delete the file
            lora_file.unlink()
            
            logger.info(f"Successfully deleted LoRA: {lora_name}")
            
            return {
                "success": True,
                "message": f"Successfully deleted LoRA: {lora_name}"
            }
            
        except Exception as e:
            logger.error(f"Error deleting LoRA {lora_name}: {e}")
            return {
                "success": False,
                "error": f"Delete failed: {str(e)}"
            }
    
    def rename_lora_file(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a LoRA file"""
        try:
            # Find the old LoRA file
            old_file = None
            lora_extensions = ['.safetensors', '.pt', '.pth', '.bin']
            
            for ext in lora_extensions:
                potential_file = self.loras_directory / f"{old_name}{ext}"
                if potential_file.exists():
                    old_file = potential_file
                    break
            
            if not old_file:
                return {
                    "success": False,
                    "error": f"LoRA file not found: {old_name}"
                }
            
            # Create new file path
            new_file = self.loras_directory / f"{new_name}{old_file.suffix}"
            
            # Check if new name already exists
            if new_file.exists():
                return {
                    "success": False,
                    "error": f"LoRA with name '{new_name}' already exists"
                }
            
            # Update loaded LoRA tracking if necessary
            if old_name in self.loaded_loras:
                lora_info = self.loaded_loras[old_name]
                del self.loaded_loras[old_name]
                lora_info["name"] = new_name
                lora_info["path"] = str(new_file)
                self.loaded_loras[new_name] = lora_info
            
            if old_name in self.applied_loras:
                strength = self.applied_loras[old_name]
                del self.applied_loras[old_name]
                self.applied_loras[new_name] = strength
            
            # Rename the file
            old_file.rename(new_file)
            
            logger.info(f"Successfully renamed LoRA: {old_name} -> {new_name}")
            
            return {
                "success": True,
                "message": f"Successfully renamed LoRA: {old_name} -> {new_name}",
                "old_name": old_name,
                "new_name": new_name,
                "new_path": str(new_file)
            }
            
        except Exception as e:
            logger.error(f"Error renaming LoRA {old_name} to {new_name}: {e}")
            return {
                "success": False,
                "error": f"Rename failed: {str(e)}"
            }
    
    def get_ui_display_data(self) -> Dict[str, Any]:
        """Get comprehensive data for UI display"""
        try:
            available_loras = self.list_available_loras()
            
            # Calculate statistics
            total_count = len(available_loras)
            total_size_mb = sum(lora["size_mb"] for lora in available_loras.values())
            loaded_count = sum(1 for lora in available_loras.values() if lora["is_loaded"])
            applied_count = sum(1 for lora in available_loras.values() if lora["is_applied"])
            
            # Group by categories
            categories = {
                "style": [],
                "character": [],
                "quality": [],
                "other": []
            }
            
            for name, info in available_loras.items():
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in ['anime', 'art', 'style', 'painting']):
                    categories["style"].append(info)
                elif any(keyword in name_lower for keyword in ['character', 'person', 'face']):
                    categories["character"].append(info)
                elif any(keyword in name_lower for keyword in ['detail', 'quality', 'hd']):
                    categories["quality"].append(info)
                else:
                    categories["other"].append(info)
            
            return {
                "loras": available_loras,
                "statistics": {
                    "total_count": total_count,
                    "total_size_mb": round(total_size_mb, 2),
                    "loaded_count": loaded_count,
                    "applied_count": applied_count
                },
                "categories": categories,
                "directory_path": str(self.loras_directory),
                "supported_formats": ['.safetensors', '.pt', '.pth', '.bin']
            }
            
        except Exception as e:
            logger.error(f"Error getting UI display data: {e}")
            return {
                "loras": {},
                "statistics": {
                    "total_count": 0,
                    "total_size_mb": 0,
                    "loaded_count": 0,
                    "applied_count": 0
                },
                "categories": {"style": [], "character": [], "quality": [], "other": []},
                "error": str(e)
            }
    
    def estimate_memory_impact(self, lora_name: str) -> Dict[str, Any]:
        """Estimate the memory impact of loading a LoRA"""
        try:
            lora_status = self.get_lora_status(lora_name)
            
            if not lora_status["exists"]:
                return {
                    "lora_name": lora_name,
                    "error": "LoRA not found"
                }
            
            file_size_mb = lora_status["size_mb"]
            
            # Estimate memory usage (typically 1.5-2x file size)
            estimated_memory_mb = file_size_mb * 1.8
            
            # Get current VRAM usage
            try:
                import psutil
                import GPUtil
                
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    vram_total_mb = gpu.memoryTotal
                    vram_used_mb = gpu.memoryUsed
                    vram_available_mb = vram_total_mb - vram_used_mb
                else:
                    # Fallback values
                    vram_total_mb = 8192  # 8GB default
                    vram_used_mb = 2048   # 2GB default usage
                    vram_available_mb = 6144
                    
            except ImportError:
                # Fallback if GPUtil not available
                vram_total_mb = 8192
                vram_used_mb = 2048
                vram_available_mb = 6144
            
            can_load = estimated_memory_mb < vram_available_mb * 0.8  # 80% threshold
            
            if estimated_memory_mb < 100:
                impact_level = "low"
            elif estimated_memory_mb < 300:
                impact_level = "medium"
            else:
                impact_level = "high"
            
            return {
                "lora_name": lora_name,
                "file_size_mb": file_size_mb,
                "estimated_memory_mb": round(estimated_memory_mb, 1),
                "vram_total_mb": vram_total_mb,
                "vram_used_mb": vram_used_mb,
                "vram_available_mb": vram_available_mb,
                "can_load": can_load,
                "impact_level": impact_level,
                "recommendation": self._get_memory_recommendation(estimated_memory_mb, vram_available_mb)
            }
            
        except Exception as e:
            logger.error(f"Error estimating memory impact for {lora_name}: {e}")
            return {
                "lora_name": lora_name,
                "error": str(e)
            }
    
    def _get_memory_recommendation(self, estimated_mb: float, available_mb: float) -> str:
        """Get memory usage recommendation"""
        if estimated_mb > available_mb:
            return "Not enough VRAM available. Consider freeing memory or using a smaller LoRA."
        elif estimated_mb > available_mb * 0.8:
            return "High VRAM usage expected. Monitor system performance."
        elif estimated_mb > available_mb * 0.5:
            return "Moderate VRAM usage. Should work well with current system."
        else:
            return "Low VRAM usage. Safe to load with current system."


# Global LoRA manager instance
_lora_manager = None

def get_lora_manager() -> LoRAManager:
    """Get the global LoRA manager instance"""
    global _lora_manager
    if _lora_manager is None:
        manager = get_model_manager()
        _lora_manager = LoRAManager(manager.config)
    return _lora_manager


# Convenience functions for LoRA operations
def apply_lora(model, lora_name: str, strength: float = 1.0):
    """Apply a LoRA to a model"""
    lora_manager = get_lora_manager()
    return lora_manager.apply_lora(model, lora_name, strength)

def remove_lora(model, lora_name: str):
    """Remove a LoRA from a model"""
    lora_manager = get_lora_manager()
    return lora_manager.remove_lora(model, lora_name)

def adjust_lora_strength(model, lora_name: str, new_strength: float):
    """Adjust LoRA strength"""
    lora_manager = get_lora_manager()
    return lora_manager.adjust_lora_strength(model, lora_name, new_strength)

def list_available_loras() -> Dict[str, Dict[str, Any]]:
    """List all available LoRAs"""
    lora_manager = get_lora_manager()
    return lora_manager.list_available_loras()

def get_fallback_prompt_enhancement(base_prompt: str, lora_name: str) -> str:
    """Get prompt-based enhancement when LoRA is unavailable"""
    lora_manager = get_lora_manager()
    return lora_manager.get_fallback_prompt_enhancement(base_prompt, lora_name)

def get_lora_status(lora_name: str) -> Dict[str, Any]:
    """Get LoRA status information"""
    lora_manager = get_lora_manager()
    return lora_manager.get_lora_status(lora_name)


class TaskQueue:
    """Thread-safe FIFO queue for managing generation tasks"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._queue = Queue(maxsize=max_size)
        self._tasks = {}  # Task ID -> GenerationTask mapping
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._current_task: Optional[GenerationTask] = None
        self._processing = False
        
    def add_task(self, task: GenerationTask) -> bool:
        """Add a task to the queue. Returns True if successful, False if queue is full"""
        with self._lock:
            try:
                # Check if queue is full
                if self._queue.full():
                    logger.warning(f"Queue is full (max size: {self.max_size}), cannot add task {task.id}")
                    return False
                
                # Add task to queue and tracking dict
                self._queue.put(task, block=False)
                self._tasks[task.id] = task
                
                logger.info(f"Added task {task.id} to queue (model: {task.model_type}, prompt: '{task.prompt[:50]}...')")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add task to queue: {e}")
                return False
    
    def get_next_task(self) -> Optional[GenerationTask]:
        """Get the next task from the queue (FIFO order). Returns None if queue is empty"""
        with self._lock:
            try:
                if self._queue.empty():
                    return None
                
                task = self._queue.get(block=False)
                self._current_task = task
                task.update_status(TaskStatus.PROCESSING)
                
                logger.info(f"Retrieved task {task.id} from queue")
                return task
                
            except Exception as e:
                logger.error(f"Failed to get next task from queue: {e}")
                return None
    
    def complete_task(self, task_id: str, output_path: Optional[str] = None, error_message: Optional[str] = None):
        """Mark a task as completed or failed"""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found in queue")
                return
            
            task = self._tasks[task_id]
            
            if error_message:
                task.update_status(TaskStatus.FAILED, error_message)
                logger.error(f"Task {task_id} failed: {error_message}")
            else:
                task.update_status(TaskStatus.COMPLETED)
                task.output_path = output_path
                logger.info(f"Task {task_id} completed successfully")
            
            # Clear current task if this was it
            if self._current_task and self._current_task.id == task_id:
                self._current_task = None
    
    def update_task_progress(self, task_id: str, progress: float):
        """Update the progress of a task (0.0 to 100.0)"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].progress = max(0.0, min(100.0, progress))
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task"""
        with self._lock:
            if task_id in self._tasks:
                return self._tasks[task_id].to_dict()
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status information"""
        with self._lock:
            pending_tasks = []
            completed_tasks = []
            failed_tasks = []
            
            for task in self._tasks.values():
                task_dict = task.to_dict()
                if task.status == TaskStatus.PENDING:
                    pending_tasks.append(task_dict)
                elif task.status == TaskStatus.COMPLETED:
                    completed_tasks.append(task_dict)
                elif task.status == TaskStatus.FAILED:
                    failed_tasks.append(task_dict)
            
            # Sort by creation time
            pending_tasks.sort(key=lambda x: x['created_at'])
            completed_tasks.sort(key=lambda x: x['created_at'], reverse=True)
            failed_tasks.sort(key=lambda x: x['created_at'], reverse=True)
            
            current_task_dict = None
            if self._current_task:
                current_task_dict = self._current_task.to_dict()
            
            return {
                "queue_size": self._queue.qsize(),
                "max_size": self.max_size,
                "is_processing": self._processing,
                "current_task": current_task_dict,
                "pending_tasks": pending_tasks,
                "completed_tasks": completed_tasks[:10],  # Limit to last 10
                "failed_tasks": failed_tasks[:10],  # Limit to last 10
                "total_pending": len(pending_tasks),
                "total_completed": len(completed_tasks),
                "total_failed": len(failed_tasks)
            }
    
    def clear_completed_tasks(self):
        """Remove completed and failed tasks from tracking"""
        with self._lock:
            tasks_to_remove = []
            for task_id, task in self._tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
            
            logger.info(f"Cleared {len(tasks_to_remove)} completed/failed tasks from tracking")
    
    def clear_all_tasks(self):
        """Clear all tasks from the queue and tracking"""
        with self._lock:
            # Clear the queue
            while not self._queue.empty():
                try:
                    self._queue.get(block=False)
                except:
                    break
            
            # Clear tracking
            self._tasks.clear()
            self._current_task = None
            self._processing = False
            
            logger.info("Cleared all tasks from queue")
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a specific task from the queue (only if pending)"""
        with self._lock:
            if task_id not in self._tasks:
                return False
            
            task = self._tasks[task_id]
            
            # Can only remove pending tasks
            if task.status != TaskStatus.PENDING:
                logger.warning(f"Cannot remove task {task_id} - status is {task.status.value}")
                return False
            
            # Remove from tracking
            del self._tasks[task_id]
            
            # Rebuild queue without this task
            temp_tasks = []
            while not self._queue.empty():
                try:
                    temp_task = self._queue.get(block=False)
                    if temp_task.id != task_id:
                        temp_tasks.append(temp_task)
                except:
                    break
            
            # Put remaining tasks back
            for temp_task in temp_tasks:
                self._queue.put(temp_task, block=False)
            
            logger.info(f"Removed task {task_id} from queue")
            return True
    
    def is_empty(self) -> bool:
        """Check if the queue is empty"""
        return self._queue.empty()
    
    def is_full(self) -> bool:
        """Check if the queue is full"""
        return self._queue.full()
    
    def size(self) -> int:
        """Get the current queue size"""
        return self._queue.qsize()
    
    def set_processing_status(self, processing: bool):
        """Set the processing status of the queue"""
        with self._lock:
            self._processing = processing


# Global task queue instance
_task_queue = None

def get_task_queue() -> TaskQueue:
    """Get the global task queue instance"""
    global _task_queue
    if _task_queue is None:
        manager = get_model_manager()
        max_queue_size = manager.config.get("queue", {}).get("max_size", 10)
        _task_queue = TaskQueue(max_size=max_queue_size)
    return _task_queue


# Convenience functions for queue operations
def add_generation_task(model_type: str, prompt: str, image: Optional[Image.Image] = None, 
                       resolution: str = "1280x720", steps: int = 50, 
                       lora_path: Optional[str] = None, lora_strength: float = 1.0) -> Optional[str]:
    """Add a generation task to the queue. Returns task ID if successful, None otherwise"""
    task = GenerationTask(
        model_type=model_type,
        prompt=prompt,
        image=image,
        resolution=resolution,
        steps=steps,
        lora_path=lora_path,
        lora_strength=lora_strength
    )
    
    queue = get_task_queue()
    if queue.add_task(task):
        return task.id
    return None

def get_next_generation_task() -> Optional[GenerationTask]:
    """Get the next task from the generation queue"""
    queue = get_task_queue()
    return queue.get_next_task()

def complete_generation_task(task_id: str, output_path: Optional[str] = None, error_message: Optional[str] = None):
    """Mark a generation task as completed or failed"""
    queue = get_task_queue()
    queue.complete_task(task_id, output_path, error_message)

def update_generation_task_progress(task_id: str, progress: float):
    """Update the progress of a generation task"""
    queue = get_task_queue()
    queue.update_task_progress(task_id, progress)

def get_generation_queue_status() -> Dict[str, Any]:
    """Get the status of the generation queue"""
    queue = get_task_queue()
    return queue.get_queue_status()

def clear_completed_generation_tasks():
    """Clear completed and failed tasks from the queue"""
    queue = get_task_queue()
    queue.clear_completed_tasks()

def clear_all_generation_tasks():
    """Clear all tasks from the generation queue"""
    queue = get_task_queue()
    queue.clear_all_tasks()

class QueueProcessor:
    """Background processor for handling generation tasks sequentially"""
    
    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._lock = threading.Lock()
        
        # Import generation functions (will be implemented in other tasks)
        self._generation_functions = {}
        
    def start_processing(self):
        """Start the background processing thread"""
        with self._lock:
            if self._is_running:
                logger.warning("Queue processor is already running")
                return
            
            self._stop_event.clear()
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                name="QueueProcessor",
                daemon=True
            )
            self._processing_thread.start()
            self._is_running = True
            
            logger.info("Started queue processing thread")
    
    def stop_processing(self):
        """Stop the background processing thread"""
        with self._lock:
            if not self._is_running:
                logger.warning("Queue processor is not running")
                return
            
            self._stop_event.set()
            self._is_running = False
            
            # Wait for thread to finish
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=5.0)
                if self._processing_thread.is_alive():
                    logger.warning("Processing thread did not stop gracefully")
            
            logger.info("Stopped queue processing thread")
    
    def _processing_loop(self):
        """Main processing loop that runs in the background thread"""
        logger.info("Queue processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get next task from queue
                task = self.task_queue.get_next_task()
                
                if task is None:
                    # No tasks available, wait a bit before checking again
                    self._stop_event.wait(timeout=1.0)
                    continue
                
                # Set processing status
                self.task_queue.set_processing_status(True)
                
                # Process the task
                self._process_task(task)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                # Continue processing other tasks
                
            finally:
                # Clear processing status
                self.task_queue.set_processing_status(False)
        
        logger.info("Queue processing loop stopped")
    
    def _process_task(self, task: GenerationTask):
        """Process a single generation task"""
        logger.info(f"Processing task {task.id}: {task.model_type} - '{task.prompt[:50]}...'")
        
        try:
            # Update progress to indicate processing started
            self.task_queue.update_task_progress(task.id, 5.0)
            
            # Load model manager and optimizer
            model_manager = get_model_manager()
            vram_optimizer = get_vram_optimizer()
            
            # Update progress
            self.task_queue.update_task_progress(task.id, 10.0)
            
            # Load the model
            logger.info(f"Loading model {task.model_type} for task {task.id}")
            model, model_info = model_manager.load_model(task.model_type)
            
            # Update progress
            self.task_queue.update_task_progress(task.id, 20.0)
            
            # Apply optimizations (using default config values for now)
            config = model_manager.config
            optimization_config = config.get("optimization", {})
            
            quantization_level = optimization_config.get("default_quantization", "bf16")
            enable_offload = optimization_config.get("enable_offload", True)
            vae_tile_size = optimization_config.get("vae_tile_size", 256)
            
            logger.info(f"Applying optimizations for task {task.id}")
            model = vram_optimizer.optimize_model(
                model, 
                quantization_level=quantization_level,
                enable_offload=enable_offload,
                vae_tile_size=vae_tile_size
            )
            
            # Update progress
            self.task_queue.update_task_progress(task.id, 30.0)
            
            # Apply LoRA if specified
            if task.lora_path:
                try:
                    logger.info(f"Applying LoRA {task.lora_path} for task {task.id}")
                    lora_manager = get_lora_manager()
                    model = lora_manager.apply_lora(model, task.lora_path, task.lora_strength)
                except Exception as e:
                    logger.warning(f"Failed to apply LoRA for task {task.id}: {e}")
                    # Continue without LoRA
            
            # Update progress
            self.task_queue.update_task_progress(task.id, 40.0)
            
            # Generate video (placeholder - actual generation will be implemented in other tasks)
            output_path = self._generate_video(task, model)
            
            # Update progress to completion
            self.task_queue.update_task_progress(task.id, 100.0)
            
            # Mark task as completed
            self.task_queue.complete_task(task.id, output_path=output_path)
            
            logger.info(f"Successfully completed task {task.id}")
            
        except Exception as e:
            error_message = f"Task processing failed: {str(e)}"
            logger.error(f"Failed to process task {task.id}: {error_message}")
            
            # Mark task as failed
            self.task_queue.complete_task(task.id, error_message=error_message)
    
    def _generate_video(self, task: GenerationTask, model) -> str:
        """Generate video for the task with enhanced image support and restoration"""
        logger.info(f"Generating video for task {task.id} with model type {task.model_type}")
        
        # Restore images from temporary storage with enhanced error handling
        restoration_success = task.restore_image_data()
        if not restoration_success:
            logger.warning(f"Image restoration had issues for task {task.id}, continuing with available images")
        
        # Get the restored images
        start_image = task.image
        end_image = task.end_image
        
        # Log image status for debugging
        if start_image is not None:
            logger.info(f"Using start image: {start_image.size} {start_image.mode}")
        if end_image is not None:
            logger.info(f"Using end image: {end_image.size} {end_image.mode}")
        
        # Validate images are appropriate for model type
        validation_result = _validate_images_for_model_type(task.model_type, start_image, end_image)
        if not validation_result["valid"]:
            logger.error(f"Image validation failed for task {task.id}: {validation_result['message']}")
            raise Exception(f"Image validation failed: {validation_result['message']}")
        
        # Update progress callback to report to queue with enhanced progress tracking
        def progress_callback(current_step, total_steps):
            if self._stop_event.is_set():
                raise Exception("Processing stopped by user")
            
            # Map generation progress to queue progress (50-95%)
            generation_progress = (current_step / total_steps) * 45  # 45% of total progress
            total_progress = 50 + generation_progress  # Start from 50%
            self.task_queue.update_task_progress(task.id, total_progress)
            
            # Log progress for debugging
            if current_step % 10 == 0 or current_step == total_steps:
                logger.debug(f"Task {task.id} progress: {current_step}/{total_steps} ({total_progress:.1f}%)")
        
        try:
            # Call the actual generation function with enhanced image support and timeout
            download_timeout = getattr(task, 'download_timeout', 300)
            smart_download = getattr(task, 'smart_download_enabled', True)
            
            logger.info(f"Starting generation for task {task.id} with timeout {download_timeout}s, smart_download: {smart_download}")
            
            result = generate_video(
                model_type=task.model_type,
                prompt=task.prompt,
                image=start_image,
                end_image=end_image,
                resolution=task.resolution,
                steps=task.steps,
                guidance_scale=7.5,  # Default value
                strength=0.8,  # Default value
                seed=-1,  # Random seed
                fps=task.fps,
                duration=task.duration,
                lora_config={"path": task.lora_path, "strength": task.lora_strength} if task.lora_path else None,
                progress_callback=progress_callback,
                download_timeout=download_timeout
            )
            
            if result.get("success", False):
                output_path = result.get("output_path")
                if output_path and Path(output_path).exists():
                    logger.info(f"Video generation completed for task {task.id}: {output_path}")
                    
                    # Log generation metadata if available
                    if result.get("metadata"):
                        metadata = result["metadata"]
                        logger.info(f"Generation metadata for task {task.id}: {metadata}")
                    
                    return output_path
                else:
                    raise Exception("Generation succeeded but output file not found")
            else:
                error_msg = result.get("error", "Unknown generation error")
                recovery_suggestions = result.get("recovery_suggestions", [])
                
                logger.error(f"Generation failed for task {task.id}: {error_msg}")
                if recovery_suggestions:
                    logger.info(f"Recovery suggestions: {'; '.join(recovery_suggestions)}")
                
                raise Exception(f"Generation failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Video generation failed for task {task.id}: {e}")
            # Add context about images to the error
            error_context = []
            if start_image is not None:
                error_context.append(f"start_image: {start_image.size}")
            if end_image is not None:
                error_context.append(f"end_image: {end_image.size}")
            
            if error_context:
                logger.error(f"Task {task.id} image context: {', '.join(error_context)}")
            
            raise
        
        finally:
            # Clean up temporary image files
            self._cleanup_temp_images(task)
    
    def _cleanup_temp_images(self, task: GenerationTask):
        """Clean up temporary image files after processing"""
        try:
            if task.image_temp_path and Path(task.image_temp_path).exists():
                Path(task.image_temp_path).unlink()
                logger.debug(f"Cleaned up temporary start image: {task.image_temp_path}")
            
            if task.end_image_temp_path and Path(task.end_image_temp_path).exists():
                Path(task.end_image_temp_path).unlink()
                logger.debug(f"Cleaned up temporary end image: {task.end_image_temp_path}")
                
        except Exception as e:
            logger.warning(f"Failed to clean up temporary images for task {task.id}: {e}")
    
    def is_running(self) -> bool:
        """Check if the processor is currently running"""
        return self._is_running
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get the current processing status"""
        return {
            "is_running": self._is_running,
            "is_processing": self.task_queue._processing,
            "current_task": self.task_queue._current_task.to_dict() if self.task_queue._current_task else None,
            "thread_alive": self._processing_thread.is_alive() if self._processing_thread else False
        }


class QueueManager:
    """High-level manager for queue operations and monitoring"""
    
    def __init__(self):
        self.task_queue = get_task_queue()
        self.processor = QueueProcessor(self.task_queue)
        self._stats_lock = threading.Lock()
        self._last_stats_update = datetime.now()
        self._cached_stats = {}
        
    def start_queue_processing(self):
        """Start automatic queue processing"""
        self.processor.start_processing()
        logger.info("Queue processing started")
    
    def stop_queue_processing(self):
        """Stop automatic queue processing"""
        self.processor.stop_processing()
        logger.info("Queue processing stopped")
    
    def add_task(self, model_type: str, prompt: str, image: Optional[Image.Image] = None,
                end_image: Optional[Image.Image] = None, resolution: str = "1280x720", steps: int = 50,
                lora_path: Optional[str] = None, lora_strength: float = 1.0, 
                download_timeout: int = 300, smart_download: bool = True) -> Optional[str]:
        """Add a new generation task to the queue with enhanced image support and smart downloading"""
        
        # Validate image-model compatibility before adding to queue
        validation_result = _validate_images_for_model_type(model_type, image, end_image)
        if not validation_result["valid"]:
            logger.error(f"Cannot add task: {validation_result['message']}")
            return None
        
        # Create task first
        task = GenerationTask(
            model_type=model_type,
            prompt=prompt,
            image=image,
            end_image=end_image,
            resolution=resolution,
            steps=steps,
            lora_path=lora_path,
            lora_strength=lora_strength,
            download_timeout=download_timeout,
            smart_download_enabled=smart_download
        )
        
        # Use the enhanced store_image_data method
        if image is not None or end_image is not None:
            try:
                task.store_image_data(image, end_image)
                logger.info(f"Stored image data for task {task.id}: {task.get_image_summary()}")
            except Exception as e:
                logger.error(f"Failed to store image data for task {task.id}: {e}")
                # Continue without images rather than failing completely
                task.image = None
                task.end_image = None
        
        if self.task_queue.add_task(task):
            logger.info(f"Added task {task.id} to queue")
            
            # Start processing if not already running
            if not self.processor.is_running():
                self.start_queue_processing()
            
            return task.id
        else:
            logger.error("Failed to add task to queue - queue may be full")
            return None
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status including queue and processing information"""
        with self._stats_lock:
            # Cache stats for 1 second to avoid excessive computation
            now = datetime.now()
            if (now - self._last_stats_update).total_seconds() < 1.0 and self._cached_stats:
                return self._cached_stats
            
            queue_status = self.task_queue.get_queue_status()
            processing_status = self.processor.get_processing_status()
            
            # Get system resource information
            try:
                vram_optimizer = get_vram_optimizer()
                vram_info = vram_optimizer.get_vram_usage()
            except Exception as e:
                logger.warning(f"Failed to get VRAM info: {e}")
                vram_info = {"used_mb": 0, "total_mb": 0, "free_mb": 0, "usage_percent": 0}
            
            comprehensive_status = {
                "queue": queue_status,
                "processing": processing_status,
                "system": {
                    "vram": vram_info,
                    "timestamp": now.isoformat()
                },
                "summary": {
                    "total_tasks": queue_status["total_pending"] + queue_status["total_completed"] + queue_status["total_failed"],
                    "success_rate": self._calculate_success_rate(queue_status),
                    "queue_utilization": (queue_status["queue_size"] / queue_status["max_size"]) * 100 if queue_status["max_size"] > 0 else 0
                }
            }
            
            self._cached_stats = comprehensive_status
            self._last_stats_update = now
            
            return comprehensive_status
    
    def _calculate_success_rate(self, queue_status: Dict[str, Any]) -> float:
        """Calculate the success rate of completed tasks"""
        total_completed = queue_status["total_completed"]
        total_failed = queue_status["total_failed"]
        
        if total_completed + total_failed == 0:
            return 100.0  # No completed tasks yet
        
        return (total_completed / (total_completed + total_failed)) * 100
    
    def pause_processing(self):
        """Pause queue processing"""
        self.stop_queue_processing()
        logger.info("Queue processing paused")
    
    def resume_processing(self):
        """Resume queue processing"""
        self.start_queue_processing()
        logger.info("Queue processing resumed")
    
    def clear_completed_tasks(self):
        """Clear completed and failed tasks"""
        self.task_queue.clear_completed_tasks()
        logger.info("Cleared completed tasks")
    
    def clear_all_tasks(self):
        """Clear all tasks and stop processing"""
        self.stop_queue_processing()
        self.task_queue.clear_all_tasks()
        logger.info("Cleared all tasks")
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a specific task from the queue"""
        return self.task_queue.remove_task(task_id)
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific task"""
        return self.task_queue.get_task_status(task_id)


# Global queue manager instance
_queue_manager = None

def get_queue_manager() -> QueueManager:
    """Get the global queue manager instance"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager


# Convenience functions for queue management
def start_queue_processing():
    """Start automatic queue processing"""
    manager = get_queue_manager()
    manager.start_queue_processing()

def stop_queue_processing():
    """Stop automatic queue processing"""
    manager = get_queue_manager()
    manager.stop_queue_processing()

def add_to_generation_queue(model_type: str, prompt: str, image: Optional[Image.Image] = None,
                           end_image: Optional[Image.Image] = None, resolution: str = "1280x720", 
                           steps: int = 50, lora_path: Optional[str] = None, 
                           lora_strength: float = 1.0, download_timeout: int = 300, 
                           smart_download: bool = True) -> Optional[str]:
    """Add a task to the generation queue with enhanced image support and smart downloading"""
    manager = get_queue_manager()
    return manager.add_task(model_type, prompt, image, end_image, resolution, steps, 
                           lora_path, lora_strength, download_timeout, smart_download)

def get_queue_comprehensive_status() -> Dict[str, Any]:
    """Get comprehensive queue and processing status"""
    manager = get_queue_manager()
    return manager.get_comprehensive_status()

def pause_queue_processing():
    """Pause queue processing"""
    manager = get_queue_manager()
    manager.pause_processing()

def resume_queue_processing():
    """Resume queue processing"""
    manager = get_queue_manager()
    manager.resume_processing()

def clear_completed_queue_tasks():
    """Clear completed and failed tasks from queue"""
    manager = get_queue_manager()
    manager.clear_completed_tasks()

def clear_all_queue_tasks():
    """Clear all tasks from queue"""
    manager = get_queue_manager()
    manager.clear_all_tasks()

def remove_queue_task(task_id: str) -> bool:
    """Remove a specific task from the queue"""
    manager = get_queue_manager()
    return manager.remove_task(task_id)

def get_queue_task_details(task_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific task"""
    manager = get_queue_manager()
    return manager.get_task_details(task_id)

# Resource Monitoring System
import time
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import pynvml

@dataclass
class ResourceStats:
    """Data structure for system resource statistics"""
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_percent: float
    vram_used_mb: float
    vram_total_mb: float
    vram_percent: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for serialization"""
        return {
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "gpu_percent": self.gpu_percent,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "vram_percent": self.vram_percent,
            "timestamp": self.timestamp.isoformat()
        }


class ResourceMonitor:
    """Monitors system resources including CPU, RAM, GPU, and VRAM"""
    
    def __init__(self, refresh_interval: int = 5):
        self.refresh_interval = refresh_interval
        self.monitoring_active = False
        self.monitor_thread = None
        self.current_stats = None
        self.stats_lock = threading.Lock()
        self.warning_callbacks: List[Callable[[str, float], None]] = []
        
        # Warning thresholds
        self.vram_warning_threshold = 90.0  # 90% VRAM usage
        self.ram_warning_threshold = 85.0   # 85% RAM usage
        self.cpu_warning_threshold = 90.0   # 90% CPU usage
        
        # Initialize NVIDIA ML
        self._init_nvidia_ml()
        
        logger.info(f"Resource monitor initialized with {refresh_interval}s refresh interval")
    
    def _init_nvidia_ml(self):
        """Initialize NVIDIA Management Library"""
        try:
            pynvml.nvmlInit()
            self.nvidia_ml_available = True
            logger.info("NVIDIA ML initialized successfully")
        except Exception as e:
            self.nvidia_ml_available = False
            logger.warning(f"Failed to initialize NVIDIA ML: {e}")
    
    def collect_system_stats(self) -> ResourceStats:
        """Collect comprehensive system resource statistics"""
        try:
            # CPU statistics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # RAM statistics
            ram_info = psutil.virtual_memory()
            ram_percent = ram_info.percent
            ram_used_gb = ram_info.used / (1024**3)
            ram_total_gb = ram_info.total / (1024**3)
            
            # GPU and VRAM statistics
            gpu_percent = 0.0
            vram_used_mb = 0.0
            vram_total_mb = 0.0
            vram_percent = 0.0
            
            if self.nvidia_ml_available:
                try:
                    # Get GPU utilization
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = gpu_util.gpu
                    
                    # Get VRAM information
                    vram_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_used_mb = vram_info.used / (1024**2)
                    vram_total_mb = vram_info.total / (1024**2)
                    vram_percent = (vram_info.used / vram_info.total) * 100
                    
                except Exception as e:
                    logger.warning(f"Failed to get GPU stats via NVIDIA ML: {e}")
                    # Fallback to GPUtil
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_percent = gpu.load * 100
                            vram_used_mb = gpu.memoryUsed
                            vram_total_mb = gpu.memoryTotal
                            vram_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    except Exception as e2:
                        logger.warning(f"Failed to get GPU stats via GPUtil: {e2}")
            
            # Create stats object
            stats = ResourceStats(
                cpu_percent=cpu_percent,
                ram_percent=ram_percent,
                ram_used_gb=ram_used_gb,
                ram_total_gb=ram_total_gb,
                gpu_percent=gpu_percent,
                vram_used_mb=vram_used_mb,
                vram_total_mb=vram_total_mb,
                vram_percent=vram_percent,
                timestamp=datetime.now()
            )
            
            # Check for warnings
            self._check_resource_warnings(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to collect system stats: {e}")
            # Return empty stats on error
            return ResourceStats(
                cpu_percent=0.0,
                ram_percent=0.0,
                ram_used_gb=0.0,
                ram_total_gb=0.0,
                gpu_percent=0.0,
                vram_used_mb=0.0,
                vram_total_mb=0.0,
                vram_percent=0.0,
                timestamp=datetime.now()
            )
    
    def _check_resource_warnings(self, stats: ResourceStats):
        """Check resource usage against warning thresholds"""
        warnings = []
        
        if stats.vram_percent >= self.vram_warning_threshold:
            warnings.append(("VRAM", stats.vram_percent))
        
        if stats.ram_percent >= self.ram_warning_threshold:
            warnings.append(("RAM", stats.ram_percent))
        
        if stats.cpu_percent >= self.cpu_warning_threshold:
            warnings.append(("CPU", stats.cpu_percent))
        
        # Trigger warning callbacks
        for resource_type, usage_percent in warnings:
            for callback in self.warning_callbacks:
                try:
                    callback(resource_type, usage_percent)
                except Exception as e:
                    logger.error(f"Error in warning callback: {e}")
    
    def start_monitoring(self):
        """Start real-time resource monitoring"""
        if self.monitoring_active:
            logger.warning("Resource monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started real-time resource monitoring")
    
    def stop_monitoring(self):
        """Stop real-time resource monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped real-time resource monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread"""
        while self.monitoring_active:
            try:
                # Collect current stats
                stats = self.collect_system_stats()
                
                # Update current stats with thread safety
                with self.stats_lock:
                    self.current_stats = stats
                
                # Wait for next refresh
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.refresh_interval)
    
    def get_current_stats(self) -> Optional[ResourceStats]:
        """Get the most recent resource statistics"""
        with self.stats_lock:
            return self.current_stats
    
    def refresh_stats_manually(self) -> ResourceStats:
        """Manually refresh and return current resource statistics"""
        stats = self.collect_system_stats()
        
        # Update current stats
        with self.stats_lock:
            self.current_stats = stats
        
        return stats
    
    def add_warning_callback(self, callback: Callable[[str, float], None]):
        """Add a callback function to be called when resource warnings occur"""
        self.warning_callbacks.append(callback)
    
    def remove_warning_callback(self, callback: Callable[[str, float], None]):
        """Remove a warning callback function"""
        if callback in self.warning_callbacks:
            self.warning_callbacks.remove(callback)
    
    def set_warning_thresholds(self, vram_threshold: float = None, 
                              ram_threshold: float = None, 
                              cpu_threshold: float = None):
        """Set custom warning thresholds for resource usage"""
        if vram_threshold is not None:
            self.vram_warning_threshold = max(0.0, min(100.0, vram_threshold))
        if ram_threshold is not None:
            self.ram_warning_threshold = max(0.0, min(100.0, ram_threshold))
        if cpu_threshold is not None:
            self.cpu_warning_threshold = max(0.0, min(100.0, cpu_threshold))
        
        logger.info(f"Updated warning thresholds: VRAM={self.vram_warning_threshold}%, "
                   f"RAM={self.ram_warning_threshold}%, CPU={self.cpu_warning_threshold}%")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get a formatted summary of current resource usage"""
        stats = self.get_current_stats()
        if not stats:
            stats = self.refresh_stats_manually()
        
        return {
            "cpu": {
                "usage_percent": round(stats.cpu_percent, 1),
                "status": "warning" if stats.cpu_percent >= self.cpu_warning_threshold else "normal"
            },
            "ram": {
                "usage_percent": round(stats.ram_percent, 1),
                "used_gb": round(stats.ram_used_gb, 2),
                "total_gb": round(stats.ram_total_gb, 2),
                "free_gb": round(stats.ram_total_gb - stats.ram_used_gb, 2),
                "status": "warning" if stats.ram_percent >= self.ram_warning_threshold else "normal"
            },
            "gpu": {
                "usage_percent": round(stats.gpu_percent, 1),
                "status": "normal"  # GPU usage warnings are less critical
            },
            "vram": {
                "usage_percent": round(stats.vram_percent, 1),
                "used_mb": round(stats.vram_used_mb, 1),
                "total_mb": round(stats.vram_total_mb, 1),
                "free_mb": round(stats.vram_total_mb - stats.vram_used_mb, 1),
                "status": "warning" if stats.vram_percent >= self.vram_warning_threshold else "normal"
            },
            "timestamp": stats.timestamp.isoformat(),
            "monitoring_active": self.monitoring_active
        }


# Global resource monitor instance
_resource_monitor = None

def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


# Convenience functions for resource monitoring
def start_resource_monitoring():
    """Start real-time resource monitoring with 5-second refresh intervals"""
    monitor = get_resource_monitor()
    monitor.start_monitoring()

def stop_resource_monitoring():
    """Stop real-time resource monitoring"""
    monitor = get_resource_monitor()
    monitor.stop_monitoring()

def get_system_stats() -> ResourceStats:
    """Get current system resource statistics"""
    monitor = get_resource_monitor()
    return monitor.collect_system_stats()

def get_current_resource_stats() -> Optional[ResourceStats]:
    """Get the most recent cached resource statistics"""
    monitor = get_resource_monitor()
    return monitor.get_current_stats()

def refresh_resource_stats() -> ResourceStats:
    """Manually refresh and return current resource statistics"""
    monitor = get_resource_monitor()
    return monitor.refresh_stats_manually()

def get_resource_summary() -> Dict[str, Any]:
    """Get a formatted summary of current resource usage"""
    monitor = get_resource_monitor()
    return monitor.get_resource_summary()

def add_resource_warning_callback(callback: Callable[[str, float], None]):
    """Add a callback function for resource usage warnings"""
    monitor = get_resource_monitor()
    monitor.add_warning_callback(callback)

def set_resource_warning_thresholds(vram_threshold: float = None, 
                                   ram_threshold: float = None, 
                                   cpu_threshold: float = None):
    """Set custom warning thresholds for resource usage"""
    monitor = get_resource_monitor()
    monitor.set_warning_thresholds(vram_threshold, ram_threshold, cpu_threshold)

def is_resource_monitoring_active() -> bool:
    """Check if real-time resource monitoring is currently active"""
    monitor = get_resource_monitor()
    return monitor.monitoring_active


# Output Management System
@dataclass
class VideoMetadata:
    """Metadata structure for generated videos"""
    id: str
    filename: str
    model_type: str
    prompt: str
    resolution: str
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float
    file_size_mb: float
    num_inference_steps: int
    guidance_scale: float
    lora_path: Optional[str] = None
    lora_strength: Optional[float] = None
    input_image_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    thumbnail_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization"""
        return {
            "id": self.id,
            "filename": self.filename,
            "model_type": self.model_type,
            "prompt": self.prompt,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "file_size_mb": self.file_size_mb,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "lora_path": self.lora_path,
            "lora_strength": self.lora_strength,
            "input_image_path": self.input_image_path,
            "created_at": self.created_at.isoformat(),
            "thumbnail_path": self.thumbnail_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoMetadata':
        """Create VideoMetadata from dictionary"""
        # Convert created_at string back to datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


class OutputManager:
    """Manages video output files, thumbnails, and metadata"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config["directories"]["output_directory"])
        self.thumbnails_dir = self.output_dir / "thumbnails"
        self.metadata_file = self.output_dir / "metadata.json"
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata_cache = self._load_metadata()
        
        # Video encoding settings
        self.video_codec = 'libx264'
        self.video_bitrate = '5000k'
        self.default_fps = 24
        self.thumbnail_size = self.config.get("ui", {}).get("gallery_thumbnail_size", 256)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                "directories": {"output_directory": "outputs"},
                "ui": {"gallery_thumbnail_size": 256}
            }
    
    def _load_metadata(self) -> Dict[str, VideoMetadata]:
        """Load video metadata from disk"""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata_cache = {}
            for video_id, metadata_dict in data.items():
                try:
                    metadata_cache[video_id] = VideoMetadata.from_dict(metadata_dict)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for video {video_id}: {e}")
            
            return metadata_cache
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save video metadata to disk"""
        try:
            data = {}
            for video_id, metadata in self.metadata_cache.items():
                data[video_id] = metadata.to_dict()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except IOError as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _generate_filename(self, model_type: str, prompt: str, resolution: str) -> str:
        """Generate a unique filename for the video"""
        # Create a safe filename from prompt (first 50 chars)
        safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        filename = f"{model_type}_{safe_prompt}_{resolution}_{timestamp}.mp4"
        
        return filename
    
    def save_video_frames(self, frames: List[Image.Image], metadata_dict: Dict[str, Any], 
                         fps: int = None) -> Tuple[str, VideoMetadata]:
        """
        Save video frames as MP4 file with metadata
        
        Args:
            frames: List of PIL Image frames
            metadata_dict: Dictionary containing generation metadata
            fps: Frames per second (defaults to 24)
            
        Returns:
            Tuple of (output_path, VideoMetadata)
        """
        if not frames:
            raise ValueError("No frames provided for video saving")
        
        fps = fps or self.default_fps
        
        # Generate unique filename
        filename = self._generate_filename(
            metadata_dict.get("model_type", "unknown"),
            metadata_dict.get("prompt", "")[:50],
            metadata_dict.get("resolution", "unknown")
        )
        
        output_path = self.output_dir / filename
        
        logger.info(f"Saving video with {len(frames)} frames to {output_path}")
        
        try:
            # Convert PIL images to numpy arrays
            frame_arrays = []
            for frame in frames:
                # Convert PIL image to RGB if not already
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                
                # Convert to numpy array (OpenCV uses BGR)
                frame_array = np.array(frame)
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                frame_arrays.append(frame_array)
            
            # Get video dimensions from first frame
            height, width = frame_arrays[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                raise RuntimeError("Failed to open video writer")
            
            # Write frames
            for frame_array in frame_arrays:
                video_writer.write(frame_array)
            
            # Release video writer
            video_writer.release()
            
            # Verify file was created and get size
            if not output_path.exists():
                raise RuntimeError("Video file was not created")
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            # Create metadata object
            video_id = str(uuid.uuid4())
            duration_seconds = len(frames) / fps
            
            metadata = VideoMetadata(
                id=video_id,
                filename=filename,
                model_type=metadata_dict.get("model_type", "unknown"),
                prompt=metadata_dict.get("prompt", ""),
                resolution=metadata_dict.get("resolution", f"{width}x{height}"),
                width=width,
                height=height,
                num_frames=len(frames),
                fps=fps,
                duration_seconds=duration_seconds,
                file_size_mb=file_size_mb,
                num_inference_steps=metadata_dict.get("num_inference_steps", 50),
                guidance_scale=metadata_dict.get("guidance_scale", 7.5),
                lora_path=metadata_dict.get("lora_path"),
                lora_strength=metadata_dict.get("lora_strength"),
                input_image_path=metadata_dict.get("input_image_path")
            )
            
            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail(frames[0], video_id)
            metadata.thumbnail_path = thumbnail_path
            
            # Cache metadata
            self.metadata_cache[video_id] = metadata
            self._save_metadata()
            
            logger.info(f"Successfully saved video: {filename} ({file_size_mb:.2f} MB, {duration_seconds:.1f}s)")
            
            return str(output_path), metadata
            
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            # Clean up partial file if it exists
            if output_path.exists():
                try:
                    output_path.unlink()
                except:
                    pass
            raise
    
    def _generate_thumbnail(self, first_frame: Image.Image, video_id: str) -> str:
        """Generate thumbnail from the first frame of the video"""
        try:
            # Create thumbnail filename
            thumbnail_filename = f"{video_id}_thumb.jpg"
            thumbnail_path = self.thumbnails_dir / thumbnail_filename
            
            # Resize image to thumbnail size while maintaining aspect ratio
            thumbnail = first_frame.copy()
            thumbnail.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.LANCZOS)
            
            # Create a square thumbnail with padding if needed
            square_thumbnail = Image.new('RGB', (self.thumbnail_size, self.thumbnail_size), (0, 0, 0))
            
            # Center the thumbnail
            x_offset = (self.thumbnail_size - thumbnail.width) // 2
            y_offset = (self.thumbnail_size - thumbnail.height) // 2
            square_thumbnail.paste(thumbnail, (x_offset, y_offset))
            
            # Save thumbnail
            square_thumbnail.save(thumbnail_path, 'JPEG', quality=85)
            
            logger.info(f"Generated thumbnail: {thumbnail_filename}")
            return str(thumbnail_path)
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            return None
    
    def get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Get metadata for a specific video"""
        return self.metadata_cache.get(video_id)
    
    def list_videos(self, sort_by: str = "created_at", reverse: bool = True) -> List[VideoMetadata]:
        """
        List all videos with their metadata
        
        Args:
            sort_by: Field to sort by ('created_at', 'filename', 'file_size_mb', etc.)
            reverse: Sort in descending order if True
            
        Returns:
            List of VideoMetadata objects sorted as requested
        """
        videos = list(self.metadata_cache.values())
        
        # Filter out videos whose files no longer exist
        existing_videos = []
        for video in videos:
            video_path = self.output_dir / video.filename
            if video_path.exists():
                existing_videos.append(video)
            else:
                logger.warning(f"Video file not found: {video.filename}")
        
        # Sort videos
        try:
            if sort_by == "created_at":
                existing_videos.sort(key=lambda x: x.created_at, reverse=reverse)
            elif sort_by == "filename":
                existing_videos.sort(key=lambda x: x.filename.lower(), reverse=reverse)
            elif sort_by == "file_size_mb":
                existing_videos.sort(key=lambda x: x.file_size_mb, reverse=reverse)
            elif sort_by == "duration_seconds":
                existing_videos.sort(key=lambda x: x.duration_seconds, reverse=reverse)
            elif sort_by == "resolution":
                existing_videos.sort(key=lambda x: (x.width * x.height), reverse=reverse)
            else:
                logger.warning(f"Unknown sort field: {sort_by}, using created_at")
                existing_videos.sort(key=lambda x: x.created_at, reverse=reverse)
        except Exception as e:
            logger.error(f"Failed to sort videos: {e}")
        
        return existing_videos
    
    def delete_video(self, video_id: str) -> bool:
        """
        Delete a video and its associated files
        
        Args:
            video_id: ID of the video to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if video_id not in self.metadata_cache:
            logger.warning(f"Video not found: {video_id}")
            return False
        
        metadata = self.metadata_cache[video_id]
        
        try:
            # Delete video file
            video_path = self.output_dir / metadata.filename
            if video_path.exists():
                video_path.unlink()
                logger.info(f"Deleted video file: {metadata.filename}")
            
            # Delete thumbnail
            if metadata.thumbnail_path:
                thumbnail_path = Path(metadata.thumbnail_path)
                if thumbnail_path.exists():
                    thumbnail_path.unlink()
                    logger.info(f"Deleted thumbnail: {thumbnail_path.name}")
            
            # Remove from metadata cache
            del self.metadata_cache[video_id]
            self._save_metadata()
            
            logger.info(f"Successfully deleted video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete video {video_id}: {e}")
            return False
    
    def get_video_path(self, video_id: str) -> Optional[str]:
        """Get the full path to a video file"""
        if video_id not in self.metadata_cache:
            return None
        
        metadata = self.metadata_cache[video_id]
        video_path = self.output_dir / metadata.filename
        
        if video_path.exists():
            return str(video_path)
        
        return None
    
    def get_thumbnail_path(self, video_id: str) -> Optional[str]:
        """Get the full path to a video's thumbnail"""
        if video_id not in self.metadata_cache:
            return None
        
        metadata = self.metadata_cache[video_id]
        if metadata.thumbnail_path and Path(metadata.thumbnail_path).exists():
            return metadata.thumbnail_path
        
        return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for the output directory"""
        try:
            total_size = 0
            video_count = 0
            thumbnail_count = 0
            
            # Calculate video files size
            for video_path in self.output_dir.glob("*.mp4"):
                if video_path.is_file():
                    total_size += video_path.stat().st_size
                    video_count += 1
            
            # Calculate thumbnail files size
            for thumb_path in self.thumbnails_dir.glob("*.jpg"):
                if thumb_path.is_file():
                    total_size += thumb_path.stat().st_size
                    thumbnail_count += 1
            
            total_size_mb = total_size / (1024 * 1024)
            
            return {
                "total_size_mb": round(total_size_mb, 2),
                "video_count": video_count,
                "thumbnail_count": thumbnail_count,
                "output_directory": str(self.output_dir),
                "thumbnails_directory": str(self.thumbnails_dir),
                "metadata_entries": len(self.metadata_cache)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                "total_size_mb": 0,
                "video_count": 0,
                "thumbnail_count": 0,
                "output_directory": str(self.output_dir),
                "thumbnails_directory": str(self.thumbnails_dir),
                "metadata_entries": len(self.metadata_cache)
            }
    
    def cleanup_orphaned_files(self) -> Dict[str, int]:
        """Clean up orphaned files (videos without metadata or thumbnails without videos)"""
        cleanup_stats = {
            "orphaned_videos_removed": 0,
            "orphaned_thumbnails_removed": 0,
            "missing_video_files_cleaned": 0
        }
        
        try:
            # Get all video IDs from metadata
            metadata_video_ids = set(self.metadata_cache.keys())
            
            # Get all video files
            video_files = list(self.output_dir.glob("*.mp4"))
            
            # Get all thumbnail files
            thumbnail_files = list(self.thumbnails_dir.glob("*.jpg"))
            
            # Find orphaned video files (files without metadata)
            for video_file in video_files:
                # Try to find corresponding metadata by filename
                found_metadata = False
                for metadata in self.metadata_cache.values():
                    if metadata.filename == video_file.name:
                        found_metadata = True
                        break
                
                if not found_metadata:
                    try:
                        video_file.unlink()
                        cleanup_stats["orphaned_videos_removed"] += 1
                        logger.info(f"Removed orphaned video: {video_file.name}")
                    except Exception as e:
                        logger.error(f"Failed to remove orphaned video {video_file.name}: {e}")
            
            # Find orphaned thumbnails (thumbnails without corresponding video metadata)
            for thumbnail_file in thumbnail_files:
                # Extract video ID from thumbnail filename (format: {video_id}_thumb.jpg)
                thumbnail_name = thumbnail_file.stem
                if thumbnail_name.endswith("_thumb"):
                    video_id = thumbnail_name[:-6]  # Remove "_thumb" suffix
                    
                    if video_id not in metadata_video_ids:
                        try:
                            thumbnail_file.unlink()
                            cleanup_stats["orphaned_thumbnails_removed"] += 1
                            logger.info(f"Removed orphaned thumbnail: {thumbnail_file.name}")
                        except Exception as e:
                            logger.error(f"Failed to remove orphaned thumbnail {thumbnail_file.name}: {e}")
            
            # Clean up metadata entries for missing video files
            missing_videos = []
            for video_id, metadata in self.metadata_cache.items():
                video_path = self.output_dir / metadata.filename
                if not video_path.exists():
                    missing_videos.append(video_id)
            
            for video_id in missing_videos:
                del self.metadata_cache[video_id]
                cleanup_stats["missing_video_files_cleaned"] += 1
                logger.info(f"Cleaned metadata for missing video: {video_id}")
            
            if missing_videos:
                self._save_metadata()
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
            return cleanup_stats
    
    def export_metadata(self, output_path: str = None) -> str:
        """Export all video metadata to a JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"metadata_export_{timestamp}.json"
        
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_videos": len(self.metadata_cache),
                "videos": {}
            }
            
            for video_id, metadata in self.metadata_cache.items():
                export_data["videos"][video_id] = metadata.to_dict()
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported metadata to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            raise


# Global output manager instance
_output_manager = None

def get_output_manager() -> OutputManager:
    """Get the global output manager instance"""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager()
    return _output_manager


# Convenience functions for output management
def save_video_frames(frames: List[Image.Image], metadata: Dict[str, Any], 
                     fps: int = 24) -> Tuple[str, VideoMetadata]:
    """Save video frames as MP4 file with metadata"""
    manager = get_output_manager()
    return manager.save_video_frames(frames, metadata, fps)

def list_generated_videos(sort_by: str = "created_at", reverse: bool = True) -> List[VideoMetadata]:
    """List all generated videos with their metadata"""
    manager = get_output_manager()
    return manager.list_videos(sort_by, reverse)

def get_video_metadata(video_id: str) -> Optional[VideoMetadata]:
    """Get metadata for a specific video"""
    manager = get_output_manager()
    return manager.get_video_metadata(video_id)

def delete_generated_video(video_id: str) -> bool:
    """Delete a generated video and its associated files"""
    manager = get_output_manager()
    return manager.delete_video(video_id)

def get_video_file_path(video_id: str) -> Optional[str]:
    """Get the full path to a video file"""
    manager = get_output_manager()
    return manager.get_video_path(video_id)

def get_video_thumbnail_path(video_id: str) -> Optional[str]:
    """Get the full path to a video's thumbnail"""
    manager = get_output_manager()
    return manager.get_thumbnail_path(video_id)

def get_output_storage_stats() -> Dict[str, Any]:
    """Get storage statistics for the output directory"""
    manager = get_output_manager()
    return manager.get_storage_stats()

def cleanup_orphaned_output_files() -> Dict[str, int]:
    """Clean up orphaned files in the output directory"""
    manager = get_output_manager()
    return manager.cleanup_orphaned_files()

def export_video_metadata(output_path: str = None) -> str:
    """Export all video metadata to a JSON file"""
    manager = get_output_manager()
    return manager.export_metadata(output_path)