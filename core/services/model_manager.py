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
from dataclasses import dataclass
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
from core.services.wan_pipeline_loader import WanPipelineLoader, GenerationConfig
from core.services.optimization_manager import OptimizationManager

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
        
        if status["is_loaded"]:
            status["model_info"] = self.model_info[full_model_id]
        
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
