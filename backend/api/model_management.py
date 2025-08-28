"""
Enhanced Model Management API
Provides model status and management endpoints using SystemIntegration and ConfigurationBridge
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio
from datetime import datetime

from backend.core.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class ModelManagementAPI:
    """Enhanced model management API using existing infrastructure"""
    
    def __init__(self):
        self.system_integration = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the model management API"""
        try:
            self.system_integration = await get_system_integration()
            self._initialized = True
            logger.info("Model Management API initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Model Management API: {e}")
            return False
    
    async def get_all_model_status(self) -> Dict[str, Any]:
        """Get status of all supported models using existing ModelManager"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get model paths from configuration bridge
            model_paths = self.system_integration.get_model_paths()
            
            # Get model status from existing ModelManager
            model_status = {}
            model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            
            for model_type in model_types:
                try:
                    status = await self._get_single_model_status(model_type)
                    model_status[model_type] = status
                except Exception as e:
                    logger.warning(f"Error getting status for {model_type}: {e}")
                    model_status[model_type] = {
                        "model_type": model_type,
                        "status": "error",
                        "error_message": str(e),
                        "is_available": False,
                        "is_loaded": False,
                        "size_mb": 0.0
                    }
            
            return {
                "models": model_status,
                "timestamp": datetime.now().isoformat(),
                "total_models": len(model_types),
                "available_models": sum(1 for status in model_status.values() if status.get("is_available", False))
            }
            
        except Exception as e:
            logger.error(f"Error getting all model status: {e}")
            raise
    
    async def _get_single_model_status(self, model_type: str) -> Dict[str, Any]:
        """Get status of a single model"""
        try:
            model_manager = self.system_integration.get_model_manager()
            model_paths = self.system_integration.get_model_paths()
            
            # Get model ID and path
            model_id = None
            model_path = None
            
            if model_manager and hasattr(model_manager, 'get_model_id'):
                model_id = model_manager.get_model_id(model_type)
            
            # Get model path from configuration
            model_path_key = f"{model_type.lower().replace('-', '_')}_model"
            if model_path_key in model_paths:
                model_path = Path(model_paths[model_path_key])
            
            # Check if model is available locally
            is_available = False
            size_mb = 0.0
            
            if model_path and model_path.exists():
                is_available = True
                # Calculate directory size
                try:
                    size_bytes = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                    size_mb = size_bytes / (1024 * 1024)
                except Exception:
                    size_mb = 0.0
            
            # Check if model is loaded
            is_loaded = False
            if model_manager and hasattr(model_manager, 'loaded_models') and model_id:
                is_loaded = model_id in getattr(model_manager, 'loaded_models', {})
            
            # Get hardware compatibility info
            optimization_settings = self.system_integration.get_optimization_settings()
            estimated_vram_mb = self._estimate_vram_usage(model_type, optimization_settings)
            
            return {
                "model_id": model_id or f"Wan2.2-{model_type}",
                "model_type": model_type,
                "status": "available" if is_available else "missing",
                "is_available": is_available,
                "is_loaded": is_loaded,
                "is_valid": is_available,  # Assume valid if available
                "size_mb": round(size_mb, 2),
                "model_path": str(model_path) if model_path else None,
                "estimated_vram_usage_mb": estimated_vram_mb,
                "hardware_compatible": True,  # Assume compatible for now
                "optimization_applied": False,
                "download_progress": None,
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"Error getting status for {model_type}: {e}")
            return {
                "model_type": model_type,
                "status": "error",
                "error_message": str(e),
                "is_available": False,
                "is_loaded": False,
                "size_mb": 0.0
            }
    
    def _estimate_vram_usage(self, model_type: str, optimization_settings: Dict[str, Any]) -> float:
        """Estimate VRAM usage for a model type"""
        # Base VRAM estimates for different models (in MB)
        base_estimates = {
            "t2v-A14B": 8000,   # ~8GB
            "i2v-A14B": 8500,   # ~8.5GB
            "ti2v-5B": 6000     # ~6GB (smaller model)
        }
        
        base_vram = base_estimates.get(model_type, 8000)
        
        # Apply optimization adjustments
        quantization = optimization_settings.get("quantization", "bf16")
        if quantization == "fp16":
            base_vram *= 0.8  # 20% reduction
        elif quantization == "int8":
            base_vram *= 0.6  # 40% reduction
        
        if optimization_settings.get("enable_offload", False):
            base_vram *= 0.7  # 30% reduction with offloading
        
        return round(base_vram, 2)
    
    async def get_model_status(self, model_type: str) -> Dict[str, Any]:
        """Get status of a specific model"""
        try:
            if not self._initialized:
                await self.initialize()
            
            return await self._get_single_model_status(model_type)
            
        except Exception as e:
            logger.error(f"Error getting model status for {model_type}: {e}")
            raise
    
    async def trigger_model_download(self, model_type: str, force_redownload: bool = False) -> Dict[str, Any]:
        """Trigger model download using existing ModelDownloader"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Check if model is already available
            if not force_redownload:
                status = await self._get_single_model_status(model_type)
                if status.get("is_available", False):
                    return {
                        "message": f"Model {model_type} is already available",
                        "model_type": model_type,
                        "status": "already_available",
                        "download_required": False
                    }
            
            # Use SystemIntegration's ensure_model_available method
            success, message = await self.system_integration.ensure_model_available(model_type)
            
            if success:
                return {
                    "message": f"Model {model_type} download completed successfully",
                    "model_type": model_type,
                    "status": "download_completed",
                    "download_required": True
                }
            else:
                return {
                    "message": f"Model {model_type} download failed: {message}",
                    "model_type": model_type,
                    "status": "download_failed",
                    "error_message": message,
                    "download_required": True
                }
            
        except Exception as e:
            logger.error(f"Error triggering model download for {model_type}: {e}")
            return {
                "message": f"Failed to trigger download for {model_type}: {str(e)}",
                "model_type": model_type,
                "status": "error",
                "error_message": str(e)
            }
    
    async def validate_model_integrity(self, model_type: str) -> Dict[str, Any]:
        """Validate model integrity using existing validation systems"""
        try:
            if not self._initialized:
                await self.initialize()
            
            status = await self._get_single_model_status(model_type)
            
            if not status.get("is_available", False):
                return {
                    "model_type": model_type,
                    "integrity_status": "not_available",
                    "is_valid": False,
                    "message": "Model is not available for validation"
                }
            
            # Basic integrity check - verify model files exist
            model_paths = self.system_integration.get_model_paths()
            model_path_key = f"{model_type.lower().replace('-', '_')}_model"
            
            if model_path_key in model_paths:
                model_path = Path(model_paths[model_path_key])
                
                # Check for essential model files
                essential_files = ["config.json", "model_index.json"]
                missing_files = []
                
                for file_name in essential_files:
                    if not (model_path / file_name).exists():
                        missing_files.append(file_name)
                
                if missing_files:
                    return {
                        "model_type": model_type,
                        "integrity_status": "corrupted",
                        "is_valid": False,
                        "message": f"Missing essential files: {', '.join(missing_files)}",
                        "missing_files": missing_files
                    }
                else:
                    return {
                        "model_type": model_type,
                        "integrity_status": "valid",
                        "is_valid": True,
                        "message": "Model integrity verified successfully"
                    }
            else:
                return {
                    "model_type": model_type,
                    "integrity_status": "path_not_found",
                    "is_valid": False,
                    "message": "Model path not found in configuration"
                }
            
        except Exception as e:
            logger.error(f"Error validating model integrity for {model_type}: {e}")
            return {
                "model_type": model_type,
                "integrity_status": "error",
                "is_valid": False,
                "message": f"Validation error: {str(e)}",
                "error_message": str(e)
            }
    
    async def get_system_optimization_status(self) -> Dict[str, Any]:
        """Get system optimization status using existing WAN22SystemOptimizer"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get system info from SystemIntegration
            system_info = self.system_integration.get_system_info()
            
            # Get optimization settings
            optimization_settings = self.system_integration.get_optimization_settings()
            
            # Get hardware profile if available
            wan22_optimizer = self.system_integration.get_wan22_system_optimizer()
            hardware_profile = None
            
            if wan22_optimizer:
                try:
                    hardware_profile = wan22_optimizer.get_hardware_profile()
                except Exception as e:
                    logger.warning(f"Could not get hardware profile: {e}")
            
            return {
                "system_integration": {
                    "initialized": system_info.get("initialized", False),
                    "components": system_info.get("components", {}),
                    "initialization_errors": system_info.get("initialization_errors", [])
                },
                "optimization_settings": optimization_settings,
                "hardware_profile": {
                    "cpu_model": hardware_profile.cpu_model if hardware_profile else "Unknown",
                    "cpu_cores": hardware_profile.cpu_cores if hardware_profile else 0,
                    "total_memory_gb": hardware_profile.total_memory_gb if hardware_profile else 0,
                    "gpu_model": hardware_profile.gpu_model if hardware_profile else "Unknown",
                    "vram_gb": hardware_profile.vram_gb if hardware_profile else 0
                } if hardware_profile else None,
                "model_paths": self.system_integration.get_model_paths(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system optimization status: {e}")
            raise

# Global instance
_model_management_api = None

async def get_model_management_api() -> ModelManagementAPI:
    """Get the global model management API instance"""
    global _model_management_api
    if _model_management_api is None:
        _model_management_api = ModelManagementAPI()
        await _model_management_api.initialize()
    return _model_management_api